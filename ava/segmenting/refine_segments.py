"""
Remove noise from segmenting files.

"""
__date__ = "August 2019 - January 2021"


from itertools import repeat
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
try: # Numba >= 0.52
	from numba.core.errors import NumbaPerformanceWarning
except ModuleNotFoundError:
	try: # Numba <= 0.45
		from numba.errors import NumbaPerformanceWarning
	except (NameError, ModuleNotFoundError):
		pass
import os
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning
import umap
import warnings

from ava.plotting.tooltip_plot import tooltip_plot
from ava.segmenting.utils import get_spec, get_audio_seg_filenames, \
		_read_onsets_offsets



def refine_segments_pre_vae(seg_dirs, audio_dirs, out_seg_dirs, p, \
	n_samples=10000, num_imgs=1000, verbose=True, img_fn='temp.pdf', \
	tooltip_output_dir='temp'):
	"""
	Manually remove noise by selecting regions of UMAP spectrogram projections.

	First, a tooltip plot of the UMAPed spectrograms will be made (using
	`ava.plotting.tooltip_plot`) and saved to `tooltip_output_dir`. You should
	open this plot and see which regions of the UMAP contain noise. Then, when
	prompted, press return to identify noise, Then enter the coordinates of a
	rectangle (x1, x2, y1, and y2) in the UMAP projection containing noise,
	following the prompts. You will be able to see the selected noise regions in
	the image save at `img_fn`, by default `'temp.pdf'`. When you are finished
	identifying noise regions, press `'q'` and the original segments from
	`seg_dirs` that aren't identified as noise (contained in one of the
	rectangles) are copied to segment files in `out_seg_dirs`.

	Doesn't support datasets that are too large to fit in memory.

	Parameters
	----------
	seg_dirs : list of str
		Directories containing segmenting information
	audio_dirs : list of str
		Directories containing audio files
	out_seg_dirs : list of str
		Directories to write updated segmenting information to
	p : dict
		Segmenting parameters: TO DO: ADD REFERENCE!
	n_samples : int, optional
		Number of spectrograms to feed to UMAP. Defaults to ``10000``.
	num_imgs : int, optional
		Number of images to embed in the tooltip plot. Defaults to ``1000``.
	verbose : bool, optional
		Defaults to ``True``.
	img_fn : str, optional
		Image filename. Defaults to ``'temp.pdf'``.
	tooltip_output_dir : str, optional
		Where to save tooltip plot. Defaults to ``'temp'``.
	"""
	if verbose:
		print("\nCleaning segments\n-----------------")
		print("Collecting spectrograms...")
	specs, max_len, _ = _get_specs(audio_dirs, seg_dirs, p, max_num_specs=n_samples)
	specs = np.stack(specs)
	if verbose:
		print("Running UMAP... n =", len(specs))
	transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, \
			metric='euclidean', random_state=42)
	with warnings.catch_warnings():
		try:
			warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
		except NameError:
			pass
		embed = transform.fit_transform(specs.reshape(len(specs), -1))
	if verbose:
		print("\tDone.")
	bounds = {'x1': [], 'x2': [], 'y1': [], 'y2': []}
	colors = ['b'] * len(embed)
	first_iteration = True

	# Keep drawing boxes around noise.
	while True:
		_plot_helper(embed, colors, verbose=verbose, filename=img_fn)
		if first_iteration:
			if verbose:
				print("Writing html plot:")
			first_iteration = False
			title = "Identify unwanted sounds:"
			tooltip_plot(embed, specs, num_imgs=num_imgs, title=title, \
					output_dir=tooltip_output_dir, grid=True)
			if verbose:
				print("\tDone.")
		if input("Press [q] to quit identifying noise or \
				[return] to continue: ") == 'q':
			break
		print("Enter the coordinates of a rectangle containing noise:")
		x1 = _get_input("x1: ")
		x2 = _get_input("x2: ")
		y1 = _get_input("y1: ")
		y2 = _get_input("y2: ")
		bounds['x1'].append(min(x1, x2))
		bounds['x2'].append(max(x1, x2))
		bounds['y1'].append(min(y1, y2))
		bounds['y2'].append(max(y1, y2))
		# Update scatter colors.
		colors = _update_colors(colors, embed, bounds)

	# Write files to out_seg_dirs.
	gen = zip(seg_dirs, audio_dirs, out_seg_dirs, repeat(p), repeat(max_len), \
			repeat(transform), repeat(bounds), repeat(verbose))
	n_jobs = min(len(seg_dirs), os.cpu_count()-1)
	Parallel(n_jobs=n_jobs)(delayed(_update_segs_helper)(*args) for args in gen)


def refine_segments_post_vae(dc, seg_dirs, audio_dirs, out_seg_dirs, \
	verbose=True, num_imgs=2000, tooltip_output_dir='temp', make_tooltip=True, \
	img_fn='temp.pdf'):
	"""
	Manually remove noise by selecting regions of UMAP latent mean projection.

	First, a tooltip plot of the spectrogram latent means will be made (using
	`ava.plotting.tooltip_plot`) and saved to `tooltip_output_dir`. You should
	open this plot and see which regions of the UMAP contain noise. Then, when
	prompted, press return to identify noise, Then enter the coordinates of a
	rectangle (x1, x2, y1, and y2) in the UMAP projection containing noise,
	following the prompts. You will be able to see the selected noise regions in
	the image save at `img_fn`, by default `'temp.pdf'`. When you are finished
	identifying noise regions, press `'q'` and the original segments from
	`seg_dirs` that aren't identified as noise (contained in one of the
	rectangles) are copied to segment files in `out_seg_dirs`.

	Doesn't support datasets that are too large to fit in memory.

	Parameters
	----------
	dc : ava.data.data_container.DataContainer
		DataContainer object
	seg_dirs : list of str
		Original segment directories.
	out_seg_dirs : list of str
		Output segment directories.
	verbose : bool, optional
		Defaults to ``True``.
	num_imgs : int, optional
		Number of images for tooltip plot. Defaults to ``2000``.
	tooltip_output_dir : str, optional
		Where to save tooltip plot. Defaults to ``'temp'``.
	make_tooltip : bool, optional
		Defaults to ``True``.
	img_fn : str, optional
		Where to save
	"""
	# Get UMAP embedding.
	embed = dc.request('latent_mean_umap')
	bounds = {'x1': [], 'x2': [], 'y1': [], 'y2': []}
	colors = ['b'] * len(embed)
	first_iteration = True
	# Keep drawing boxes around noise.
	while True:
		_plot_helper(embed, colors, filename=img_fn, verbose=verbose)
		if first_iteration and make_tooltip:
			if verbose:
				print("Writing html plot:")
			first_iteration = False
			title = "Identify unwanted sounds:"
			specs = dc.request('specs')
			tooltip_plot(embed, specs, num_imgs=num_imgs, title=title, \
					output_dir=tooltip_output_dir, grid=True)
			if verbose:
				print("\tDone.")
		if input("Press [q] to quit identifying noise or \
				[return] to continue: ") == 'q':
			break
		print("Enter the coordinates of a rectangle containing noise:")
		x1 = _get_input("x1: ")
		x2 = _get_input("x2: ")
		y1 = _get_input("y1: ")
		y2 = _get_input("y2: ")
		bounds['x1'].append(min(x1,x2))
		bounds['x2'].append(max(x1,x2))
		bounds['y1'].append(min(y1,y2))
		bounds['y2'].append(max(y1,y2))
		# Update scatter colors.
		colors = _update_colors(colors, embed, bounds)
	# Write files to out_seg_dirs.
	audio_fns = dc.request('audio_filenames')
	segs = np.zeros((len(audio_fns), 2))
	segs[:,0] = dc.request('onsets')
	segs[:,1] = dc.request('offsets')
	good_sylls = np.argwhere(colors == 'b').flatten()
	good_sylls = [i for i in range(len(colors)) if colors[i] == 'b']
	good_sylls = np.array(good_sylls, dtype='int')
	for fn in np.unique(audio_fns):
		# File stuff.
		index = [1 if a in fn else 0 for a in audio_dirs].index(1)
		seg_fn = os.path.split(fn)[-1][:-4] + '.txt'
		out_seg_fn = os.path.join(out_seg_dirs[index], seg_fn)
		seg_fn = os.path.join(seg_dirs[index], seg_fn)
		if not os.path.exists(out_seg_dirs[index]):
			os.makedirs(out_seg_dirs[index])
		# Collect indices of syllables to save.
		indices = np.argwhere(audio_fns == fn).flatten()
		indices = np.intersect1d(indices, good_sylls, assume_unique=True)
		header = "Cleaned onsets/offsets from: " + seg_fn
		np.savetxt(out_seg_fn, segs[indices], fmt='%.5f', header=header)
	# Write empty files if we don't have any syllables from them.
	for audio_dir, out_seg_dir in zip(audio_dirs, out_seg_dirs):
		for temp_fn in [os.path.join(audio_dir, i) for i in os.listdir(audio_dir)]:
			if _is_audio_file(temp_fn) and temp_fn not in audio_fns:
				header = "Cleaned onsets/offsets from: " + temp_fn
				out_seg_fn = os.path.split(temp_fn)[-1][:-4] + '.txt'
				out_seg_fn = os.path.join(out_seg_dir, out_seg_fn)
				np.savetxt(out_seg_fn, np.array([]), header=header)
	if verbose:
		msg = "Retained "+str(sum(1 for i in colors if i=='b'))
		msg += " out of " + str(len(colors)) + " segments."
		print(msg)


def _get_specs(audio_dirs, seg_dirs, p, max_num_specs=None, max_len=None, \
	return_segs=False):
	"""
	Make a bunch of spectrograms.

	Parameters
	----------
	audio_dirs : list of str
		Directories containing audio files
	seg_dirs : list of str
		Directories containing segmenting decisions
	p : dict
		Segementing parameters. TO DO: ADD REFERENCE!
	max_num_specs : {int, None}, optional
		Defaults to ``None``.
	max_len : {int, None}, optional
		Maximum number of spectrogram time bins.
	return_segs : bool, optional
		Defaults to ``False``.

	Returns
	-------
	specs : list of numpy.ndarray
		Spectrograms.
	max_len : int
		Maximum number of spectrogram time bins.
	all_fns : ...
		...
	segs : numpy.ndarray
		Onsets and offsets for each spectrogram. Returned if ``return_segs``.
	"""
	# Get the filenames.
	audio_fns, seg_fns = get_audio_seg_filenames(audio_dirs, seg_dirs)
	# Reproducibly shuffle.
	audio_fns, seg_fns = np.array(audio_fns), np.array(seg_fns)
	np.random.seed(42)
	perm = np.random.permutation(len(audio_fns))
	np.random.seed(None)
	audio_fns, seg_fns = audio_fns[perm], seg_fns[perm]
	# Collect spectrograms.
	specs, all_fns, segs = [], [], []
	for audio_fn, seg_fn in zip(audio_fns, seg_fns):
		onsets, offsets = _read_onsets_offsets(seg_fn)
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=WavFileWarning)
			fs, audio = wavfile.read(audio_fn)
		assert len(audio) >= p['nperseg'], "Short audio file: " + audio_fn + \
				", duration: " + str(len(audio)/fs)
		for onset, offset in zip(onsets, offsets):
			i1, i2 = int(onset * fs), int(offset * fs)
			if i2-i1 <= p['nperseg']:
				continue
			assert i1 >= 0, audio_fn + ", " + seg_fn
			spec, dt, _ = get_spec(audio[i1:i2], p)
			specs.append(spec)
			all_fns.append(os.path.split(seg_fn)[-1])
			segs.append(np.array([onset, 0.0])) # Offsets added below.
			if max_num_specs is not None and len(specs) >= max_num_specs:
				break
		if max_num_specs is not None and len(specs) >= max_num_specs:
			break
	# Zero-pad.
	assert len(specs) > 0, "Found no spectrograms!"
	n_freq_bins = specs[0].shape[0]
	if max_len is None:
		max_len = max(spec.shape[1] for spec in specs)
	for i in range(len(specs)):
		spec = np.zeros((n_freq_bins, max_len))
		spec[:,:specs[i].shape[1]] = specs[i][:,:max_len]
		specs[i] = spec
		segs[i][1] = segs[i][0] + dt * max_len
	if return_segs:
		segs = np.array(segs)
		return specs, max_len, all_fns, segs
	return specs, max_len, all_fns


def _plot_helper(embed, colors, title="", filename='temp.pdf', verbose=True):
	"""Helper function to plot a UMAP projection with grids."""
	plt.scatter(embed[:,0], embed[:,1], c=colors, s=0.9, alpha=0.7)
	delta = 1
	if np.max(embed) - np.min(embed) > 20:
		delta = 5
	min_xval = int(np.floor(np.min(embed[:,0])))
	if min_xval % delta != 0:
		min_xval -= min_xval % delta
	max_xval = int(np.ceil(np.max(embed[:,0])))
	if max_xval % delta != 0:
		max_xval -= (max_xval % delta) - delta
	min_yval = int(np.floor(np.min(embed[:,1])))
	if min_yval % delta != 0:
		min_yval -= min_yval % delta
	max_yval = int(np.ceil(np.max(embed[:,1])))
	if max_yval % delta != 0:
		max_yval -= (max_yval % delta) - delta
	for x_val in range(min_xval, max_xval+1):
		plt.axvline(x=x_val, lw=0.5, alpha=0.7)
	for y_val in range(min_yval, max_yval+1):
		plt.axhline(y=y_val, lw=0.5, alpha=0.7)
	plt.title(title)
	plt.savefig(filename)
	plt.close('all')
	if verbose:
		print("Grid plot saved to:", filename)


def _update_segs_helper(seg_dir, audio_dir, out_seg_dir, p, max_len,
	transform, bounds, verbose):
	"""
	Write updated segments.

	Parameters
	----------
	seg_dir : str
		Original segment directory.
	audio_dir : str
		Audio directory.
	out_seg_dir : str
		Output segment directory.
	p : dict
		Params. TO DO: add reference!
	max_len : int
		Maximum number of spectrogram time bins.
	transform : umap.umap_.UMAP
		UMAP object.
	bounds : dict
		Maps the keys ``'x1'``, ``'x2'``, ``'y1'``, and ``'y2'`` to values
		defining rectangular bounds.
	verbose : bool
		Verbosity.
	"""
	if verbose:
		print("Updating segments in:", seg_dir)
	if not os.path.exists(out_seg_dir):
		os.makedirs(out_seg_dir)
	specs, _, all_fns = \
			_get_specs([audio_dir], [seg_dir], p, max_len=max_len)
	specs = np.stack(specs)
	embed = transform.transform(specs.reshape(len(specs), -1))
	out_segs = []
	prev_fn, prev_segs = None, None
	for i in range(len(all_fns)):
		if all_fns[i] != prev_fn:
			if len(out_segs) > 0:
				audio_fn = os.path.join(audio_dir, prev_fn)
				out_seg_fn = os.path.join(out_seg_dir, prev_fn)
				_write_segs(out_segs, out_seg_fn, audio_fn)
				out_segs = []
			prev_fn = all_fns[i]
			prev_segs = np.loadtxt(os.path.join(seg_dir, prev_fn)).reshape(-1,2)
			index = 0 # within-file index
		if not _in_bounds(embed[i], bounds):
			out_segs.append(prev_segs[index])
		index += 1
	if len(out_segs) > 0:
		audio_fn = os.path.join(audio_dir, prev_fn)
		out_seg_fn = os.path.join(out_seg_dir, prev_fn)
		_write_segs(out_segs, out_seg_fn, audio_fn)


def _write_segs(segs, out_fn, header_fn):
	"""
	Write onstes/offsets to a text file.

	Parameters
	----------
	segs : list of lists
		Onsets and offsets for each segment.
	out_fn : str
		Output filename.
	header_fn : str
		Filename to write in header.
	"""
	segs = np.stack([np.array(seg) for seg in segs])
	header = "Cleaned onsets/offsets for " + header_fn
	np.savetxt(out_fn, segs, fmt='%.5f', header=header)


def _get_input(query_str):
	"""Get float-valued input."""
	while True:
		try:
			temp = float(input(query_str))
			return temp
		except:
			print("Unrecognized input!")
			pass


def _update_colors(colors, embed, bounds):
	"""Color red if embed is in the bounds, blue otherwise."""
	for i in range(len(colors)):
		if colors[i] == 'b' and _in_bounds(embed[i], bounds):
			colors[i] = 'r'
	return colors


def _in_bounds(point, bounds):
	"""Is the point in the given rectangular bounds?"""
	for i in range(len(bounds['x1'])):
		if point[0] > bounds['x1'][i] and point[0] < bounds['x2'][i] and \
				point[1] > bounds['y1'][i] and point[1] < bounds['y2'][i]:
			return True
	return False


def _is_audio_file(filename):
	return len(filename) > 4 and filename[-4:] == '.wav'


if __name__ == '__main__':
	pass


###
