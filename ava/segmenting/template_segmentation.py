"""
Segment song motifs by finding maxima in spectrogram cross correlations.

"""
__date__ = "April 2019 - November 2020"


from affinewarp import ShiftWarping
import h5py
from itertools import repeat
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
plt.switch_backend('agg')
try: # Numba >= 0.52
	from numba.core.errors import NumbaPerformanceWarning
except ModuleNotFoundError:
	try: # Numba <= 0.45
		from numba.errors import NumbaPerformanceWarning
	except (NameError, ModuleNotFoundError):
		pass
import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning
from scipy.signal import stft
from scipy.ndimage.filters import gaussian_filter
import os
import umap
import warnings

from ava.plotting.tooltip_plot import tooltip_plot


EPSILON = 1e-9



def get_template(feature_dir, p, smoothing_kernel=(0.5, 0.5), verbose=True):
	"""
	Create a linear feature template given exemplar spectrograms.

	Parameters
	----------
	feature_dir : str
		Directory containing multiple audio files to average together.
	p : dict
		Parameters. Must contain keys: ``'fs'``, ``'min_freq'``, ``'max_freq'``,
		``'nperseg'``, ``'noverlap'``, ``'spec_min_val'``, ``'spec_max_val'``.
	smoothing_kernel : tuple of floats, optional
		Each spectrogram is blurred using a gaussian kernel with the following
		bandwidths, in bins. Defaults to ``(0.5, 0.5)``.
	verbose : bool, optional
		Defaults to ``True``.

	Returns
	-------
	template : np.ndarray
		Spectrogram template.
	"""
	filenames = [os.path.join(feature_dir, i) for i in os.listdir(feature_dir) \
		if _is_wav_file(i)]
	specs = []
	for i, filename in enumerate(filenames):
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=WavFileWarning)
			fs, audio = wavfile.read(filename)
		assert fs == p['fs'], "Found samplerate="+str(fs)+\
			", expected "+str(p['fs'])
		spec, dt = _get_spec(fs, audio, p)
		spec = gaussian_filter(spec, smoothing_kernel)
		specs.append(spec)
	min_time_bins = min(spec.shape[1] for spec in specs)
	specs = np.array([i[:,:min_time_bins] for i in specs])
	# Average over all the templates.
	template = np.mean(specs, axis=0)
	# Normalize to unit norm.
	template -= np.mean(template)
	template /= np.sum(np.power(template, 2)) + EPSILON
	if verbose:
		duration = min_time_bins * dt
		print("Made template from", len(filenames), "files. Duration:", duration)
	return template


def segment_files(audio_dirs, segment_dirs, template, p, num_mad=2.0, \
	min_dt=0.05, n_jobs=1, verbose=True):
	"""
	Write segments to text files.

	Parameters
	----------
	audio_dirs : list of str
		Audio directories.
	segment_dirs : list of str
		Corresponding directories containing segmenting decisions.
	template : numpy.ndarray
		Spectrogram template.
	p : dict
		Parameters. Must contain keys: ``'fs'``, ``'min_freq'``, ``'max_freq'``,
		``'nperseg'``, ``'noverlap'``, ``'spec_min_val'``, ``'spec_max_val'``.
	num_mad : float, optional
		Number of median absolute deviations for cross-correlation threshold.
		Defaults to ``2.0``.
	min_dt : float, optional
		Minimum duration between cross correlation maxima. Defaults to ``0.05``.
	n_jobs : int, optional
		Number of jobs for parallelization. Defaults to ``1``.
	verbose : bool, optional
		Defaults to ``True``.

	Returns
	-------
	result : dict
		Maps audio filenames to segments (numpy.ndarrays).
	"""
	# Collect all the filenames we need to parallelize.
	all_audio_fns = []
	all_seg_dirs = []
	for audio_dir, segment_dir in zip(audio_dirs, segment_dirs):
		if not os.path.exists(segment_dir):
			os.makedirs(segment_dir)
		audio_fns = [os.path.join(audio_dir, i) for i in os.listdir(audio_dir) \
			if _is_wav_file(i)]
		all_audio_fns = all_audio_fns + audio_fns
		all_seg_dirs = all_seg_dirs + [segment_dir]*len(audio_fns)
	# Segment.
	if verbose:
		print("Segmenting files. n =",len(all_audio_fns))
	gen = zip(all_seg_dirs, all_audio_fns, repeat(template), repeat(p), \
			repeat(num_mad), repeat(min_dt))
	res = Parallel(n_jobs=n_jobs)(delayed(_segment_file)(*args) for args in gen)
	# Write results.
	result = {}
	num_segments = 0
	for segment_dir, audio_fn, segments in res:
		result[audio_fn] = segments
		segment_fn = os.path.split(audio_fn)[-1][:-4] + '.txt'
		segment_fn = os.path.join(segment_dir, segment_fn)
		np.savetxt(segment_fn, segments, fmt='%.5f')
		num_segments += len(segments)
	if verbose:
		print("\tFound", num_segments, "segments.")
		print("\tDone.")
	# Return a dictionary mapping audio filenames to segments.
	return result


def read_segment_decisions(audio_dirs, segment_dirs, verbose=True):
	"""
	Returns the same data as ``segment_files``.

	Parameters
	----------
	audio_dirs : list of str
		Audio directories.
	segment_dirs : list of str
		Segment directories.
	verbose : bool, optional
		Defaults to ``True``.

	Returns
	-------
	result : dict
		Maps audio filenames to segments.
	"""
	if verbose:
		print("Reading segments...")
	result = {}
	n_segs = 0
	for audio_dir, segment_dir in zip(audio_dirs, segment_dirs):
		audio_fns = [os.path.join(audio_dir, i) for i in os.listdir(audio_dir) \
			if _is_wav_file(i)]
		for audio_fn in audio_fns:
			segment_fn = os.path.split(audio_fn)[-1][:-4] + '.txt'
			segment_fn = os.path.join(segment_dir, segment_fn)
			segments = np.loadtxt(segment_fn).reshape(-1,2)
			result[audio_fn] = segments
			n_segs += len(segments)
	if verbose:
		print("\tFound", n_segs, "segments.")
		print("\tDone.")
	return result


def _segment_file(segment_dir, filename, template, p, num_mad=2.0, min_dt=0.05,\
	min_extra_time_bins=5):
	"""
	Match linear spetrogram features and extract times where features align.

	Parameters
	----------
	segment_dir : str
		Segment directory.
	filename : str
		Audio filename.
	template : numpy.ndarray
		Spectrogram template.
	p : dict
		Parameters. Must contain keys: ``'fs'``, ``'min_freq'``, ``'max_freq'``,
		``'nperseg'``, ``'noverlap'``, ``'spec_min_val'``, ``'spec_max_val'``.
	num_mad : float, optional
		Number of median absolute deviations for cross-correlation threshold.
		Defaults to ``2.0``.
	min_dt : float, optional
		...
	min_extra_time_bins : int, optional
		...

	Returns
	-------
	segment_dir : str
		Copied from input parameters.
	filename : str
		Copied from input parameters.
	segments : numpy.ndarray
		Onsets and offsets.
	"""
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", category=WavFileWarning)
		fs, audio = wavfile.read(filename)
	assert fs == p['fs'], "Found samplerate="+str(fs)+", expected "+str(p['fs'])
	if len(audio) < p['nperseg']:
		warnings.warn(
			"Found an audio file that is too short to make a spectrogram: "+\
			filename + "\nSamples: "+str(len(audio))+"\np[\'nperseg\']: "+\
			str(p['nperseg']),
			UserWarning
		)
		return segment_dir, filename, np.zeros((0, 2))
	big_spec, dt = _get_spec(fs, audio, p)
	spec_len = template.shape[1]
	template = template.flatten()
	if big_spec.shape[1] - spec_len < min_extra_time_bins:
		d1, d2 = dt*spec_len, dt*big_spec.shape[1]
		warnings.warn(
			"Found an audio file that is too short to extract segments from: "+\
			filename + "\nTemplate duration: "+str(d1)+"\nFile duration: "+\
			str(d2)+"\nConsider reducing the template duration.",
			UserWarning
		)
		return segment_dir, filename, np.zeros((0, 2))
	# Compute normalized cross-correlation.
	result = np.zeros(big_spec.shape[1] - spec_len)
	for i in range(len(result)):
		temp = big_spec[:,i:i+spec_len].flatten()
		temp -= np.mean(temp)
		temp /= np.sum(np.power(temp, 2)) + EPSILON
		result[i] = np.dot(template, temp)
	median = np.median(result)
	abs_devs = np.abs(result - median)
	mad = np.median(abs_devs) + EPSILON
	# Get maxima.
	times = dt * np.arange(len(result))
	indices = np.argwhere(result>median + num_mad*mad).flatten()[1:-1]
	max_indices = []
	for i in range(2,len(indices)-1):
		if max(result[indices[i]-1], result[indices[i]+1]) < result[indices[i]]:
			max_indices.append(indices[i])
	max_indices = np.array(max_indices, dtype='int')
	max_indices = _clean_max_indices(max_indices, times, result, min_dt=min_dt)
	# Define onsets/offsets.
	segments = np.zeros((len(max_indices), 2))
	segments[:,0] = dt * max_indices # onsets
	segments[:,1] = segments[:,0] + spec_len * dt
	return segment_dir, filename, segments


def clean_collected_data(result, audio_dirs, segment_dirs, p, \
	max_num_specs=10000, verbose=True, img_fn='temp.pdf', \
	tooltip_plot_dir='html'):
	"""Deprecated. See ``clean_collected_segments``."""
	warnings.warn(
		"ava.segmenting.template_segmentation.clean_collected_data has been" + \
		" renamed to clean_collected_segments in v0.3.0.",
		UserWarning
	)
	clean_collected_segments(result, audio_dirs, segment_dirs, p, \
		max_num_specs=max_num_specs, verbose=verbose, img_fn=img_fn, \
		tooltip_plot_dir=tooltip_plot_dir)


def clean_collected_segments(result, audio_dirs, segment_dirs, p, \
	max_num_specs=10000, verbose=True, img_fn='temp.pdf', \
	tooltip_plot_dir='html'):
	"""
	Take a look at the collected segments and discard false positives.

	Parameters
	----------
	result : dict
		Output of ``segment_files`` or `read_segment_decisions``.
	audio_dirs : list of str
		Directories containing audio.
	segment_dirs : list of str
		Directories containing segmenting decisions.
	p : dict
		Parameters. Must contain keys: ``'fs'``, ``'min_freq'``, ``'max_freq'``,
		``'nperseg'``, ``'noverlap'``, ``'spec_min_val'``, ``'spec_max_val'``.
	max_num_specs : int, optional
		Maximum number of spectrograms to feed to UMAP. Deafults to ``10000``.
	verbose : bool, optional
		Defaults to ``True``.
	img_fn : str, optional
		Image filename. Defaults to ``'temp.pdf'``.
	tooltip_plot_dir : str, optional
		Directory to save tooltip plot to. Defaults to ``'html'``.
	"""
	# Collect spectrograms.
	if verbose:
		print("Collecting spectrograms...")
	specs = []
	for filename in result.keys():
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=WavFileWarning)
			fs, audio = wavfile.read(filename)
		assert fs == p['fs'], "Found samplerate=" + str(fs) + \
				", expected " + str(p['fs'])
		for segment in result[filename]:
			i1 = int(round(segment[0] * fs))
			i2 = int(round(segment[1] * fs))
			spec, dt = _get_spec(fs, audio[i1:i2], p)
			specs.append(spec)
	if len(specs) == 0:
		warnings.warn(
			"Found no spectrograms in " + \
			"ava.segmenting.template_segmentation.clean_collected_segments.\n" \
			+ "Consider reducing the `num_mad` parameter in `segment_files`.",
			UserWarning
		)
		return
	max_t = max(spec.shape[1] for spec in specs)
	temp_specs = np.zeros((len(specs), specs[0].shape[0], max_t))
	for i, spec in enumerate(specs):
		temp_specs[i,:,:spec.shape[1]] = spec
	specs = temp_specs
	if len(specs) > max_num_specs:
		warnings.warn(
			"Found more spectrograms than `max_num_specs` (" + \
			str(max_num_specs) + "). Consider increasing `max_num_specs` or" + \
			" `num_mad`.",
			UserWarning
		)
	if verbose:
		print("\tCollected",len(specs),"spectrograms.")
		print("\tSpectrogram shape:", specs.shape[1:])
		if len(specs) > max_num_specs:
			print("\tRandomly sampling", max_num_specs, "spectrograms.")
		print("\tDone.")
	np.random.seed(42)
	specs = specs[np.random.permutation(len(specs))[:max_num_specs]]
	np.random.seed(None)
	# UMAP the spectrograms.
	if verbose:
		print("Running UMAP. n =", len(specs))
	transform = umap.UMAP(random_state=42, metric='correlation')
	# https://github.com/lmcinnes/umap/issues/252
	with warnings.catch_warnings():
		try:
			warnings.filterwarnings("ignore", \
					category=NumbaPerformanceWarning)
		except NameError:
			pass
		embedding = transform.fit_transform(specs.reshape(len(specs), -1))
	if verbose:
		print("\tDone.")
	# Plot and ask for user input.
	bounds = {
		'x1s':[],
		'x2s':[],
		'y1s':[],
		'y2s':[],
	}
	bounds_keys = ['x1s', 'x2s', 'y1s', 'y2s']
	queries = ['x1: ', 'x2: ', 'y1: ', 'y2: ']
	X, Y = embedding[:,0], embedding[:,1]
	i = 0

	while True:
		colors = ['b' if _in_region(embed, bounds) else 'r' for \
				embed in embedding]
		print("Selected", \
				len([c for c in colors if c=='b']), "out of", len(colors))
		plt.scatter(X, Y, c=colors, s=0.9, alpha=0.5)
		for x_tick in np.arange(np.floor(np.min(X)), np.ceil(np.max(X))):
			plt.axvline(x=x_tick, c='k', alpha=0.1, lw=0.5)
		for y_tick in np.arange(np.floor(np.min(Y)), np.ceil(np.max(Y))):
			plt.axhline(y=y_tick, c='k', alpha=0.1, lw=0.5)
		title = "Find relevant song"
		plt.title(title)
		plt.savefig(img_fn)
		plt.close('all')
		# Plot the tooltip plot.
		if i == 0:
			if verbose:
				print("Writing tooltip plot...")
			tooltip_plot(embedding, specs, output_dir=tooltip_plot_dir, \
					num_imgs=1000, title=title, grid=True)
			if verbose:
				print("\tDone.")
		# Get input from user.
		for key, query in zip(bounds_keys, queries):
			answer = 'initial input'
			while not _is_number(answer):
				answer = input(query)
			bounds[key].append(float(answer))
		# Continue?
		temp = input('[Enter] to select more regions, [c] to continue: ')
		if temp == 'c':
			break
		i += 1
	# Save only the good segments.
	if verbose:
		print("Saving segments...")
	num_deleted, num_total = 0, 0
	for audio_dir, seg_dir in zip(audio_dirs, segment_dirs):
		audio_fns = [os.path.join(audio_dir, i) for i in os.listdir(audio_dir) \
			if _is_wav_file(i)]
		for audio_fn in audio_fns:
			with warnings.catch_warnings():
				warnings.filterwarnings("ignore", category=WavFileWarning)
				fs, audio = wavfile.read(audio_fn)
			assert fs == p['fs'], "Found samplerate=" + str(fs) + \
					", expected " + str(p['fs'])
			segment_fn = os.path.split(audio_fn)[-1][:-4] + '.txt'
			segment_fn = os.path.join(seg_dir, segment_fn)
			segments = np.loadtxt(segment_fn).reshape(-1,2)
			if len(segments) == 0:
				continue
			new_segments = np.zeros(segments.shape)
			i = 0
			specs = []
			for segment in segments:
				i1 = int(round(segment[0] * fs))
				i2 = int(round(segment[1] * fs))
				spec, dt = _get_spec(fs, audio[i1:i2], p)
				temp_spec = np.zeros((spec.shape[0], max_t))
				temp_spec[:, :spec.shape[1]] = spec
				spec = temp_spec
				specs.append(spec)
			specs = np.stack(specs)
			embed = transform.transform(specs.reshape(specs.shape[0],-1))
			for j, segment in enumerate(segments):
				if _in_region(embed[j], bounds):
					new_segments[i] = segment[:]
					i += 1
					num_total += 1
				else:
					num_deleted += 1
			new_segments = new_segments[:i]
			np.savetxt(segment_fn, new_segments, fmt='%.5f')
	if verbose:
		print("\tdeleted:", num_deleted, "remaining:", num_total)
		print("\tDone.")


def segment_sylls_from_songs(audio_dirs, song_seg_dirs, syll_seg_dirs, p, \
	shoulder=0.05, img_fn='temp.pdf', verbose=True):
	"""
	Split song renditions into syllables, write segments.

	Enter quantiles to determine where to split the song motif. Entering the
	same quantile twice will remove it.

	Note
	----
	* All the song segments must be the same duration!

	Parameters
	----------
	audio_dirs : list of str
		Audio directories.
	song_seg_dirs : list of str
		Directories containing song segments.
	syll_seg_dirs : list of str
		Directories where syllable segments are written.
	p : dict
		Segmenting parameters.
	shoulder : float, optional
		Duration of padding on either side of song segments, in seconds.
	img_fn : str, optional
		Image filename. Defaults to ``'temp.pdf'``.
	verbose : bool, optional
		Defaults to `True`.
	"""
	# Read segments.
	song_segs = read_segment_decisions(audio_dirs, song_seg_dirs)
	# Collect spectrograms.
	empty_audio_files = []
	specs, fns, song_onsets = [], [], []
	for audio_fn in song_segs:
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=WavFileWarning)
			fs, audio = wavfile.read(audio_fn)
		for seg in song_segs[audio_fn].reshape(-1,2):
			# Make spectrogram.
			onset, offset = seg[0] - shoulder, seg[1] + shoulder
			i1, i2 = int(fs*onset), int(fs*offset)
			spec, dt = _get_spec(fs, audio[max(i1,0):i2], p)
			# Pad spectrogram if it's near the edge of the file.
			if i1 < 0 or i2 > len(audio):
				pre_bins = max(0, int(np.round(-i1/fs/dt)))
				post_bins = max(0, int(np.round((i2 - len(audio))/fs/dt)))
				new_spec = np.mean(spec) * \
						np.ones((spec.shape[0], spec.shape[1]+pre_bins+post_bins))
				if post_bins == 0:
					post_bins = -new_spec.shape[1]
				new_spec[pre_bins:-post_bins]
				spec = new_spec
			specs.append(spec)
			fns.append(audio_fn)
			song_onsets.append(onset)
		if len(song_segs[audio_fn]) == 0:
			empty_audio_files.append(audio_fn)
	assert len(specs) > 0, "Found no spectrograms!"
	# Calculate and smooth amplitude traces.
	amp_traces = []
	for spec in specs:
		amps = np.sum(spec, axis=0)
		amps -= np.mean(amps)
		amps /= np.std(amps) + EPSILON
		amp_traces.append(amps)
	# Truncate the amplitude traces if they aren't exactly the same length.
	min_time_bins = min(len(amp_trace) for amp_trace in amp_traces)
	max_time_bins = max(len(amp_trace) for amp_trace in amp_traces)
	if verbose and (min_time_bins != max_time_bins):
		print("Found different numbers of time bins in segments!")
		print("\tmin:" + str(min_time_bins) + ", max:", max_time_bins)
		print("\tTruncating to minimum number of time bins.")
	if min_time_bins != max_time_bins:
		amp_traces = [amp_trace[:min_time_bins] for amp_trace in amp_traces]
	amp_traces = np.array(amp_traces)
	# Warp the amplitude traces.
	max_t = amp_traces.shape[1]*dt*1e3
	num_time_bins = amp_traces.shape[1]
	model = ShiftWarping(maxlag=0.2, smoothness_reg_scale=10.0)
	model.fit(amp_traces[:,:,np.newaxis], iterations=50)
	aligned = model.predict().squeeze()
	max_raw_val = np.max(amp_traces)
	max_aligned_val = np.max(aligned)
	shifts = model.shifts
	quantiles = []
	break_flag = False
	while True:
		# Plot.
		_, axarr = plt.subplots(3,1, sharex=True)
		axarr[0].imshow(specs[np.random.randint(len(specs))], origin='lower', \
				aspect='auto', extent=[0,max_t,p['min_freq']/1e3, \
				p['max_freq']/1e3])
		temp = np.copy(amp_traces)
		for q in quantiles:
			for i in range(len(temp)):
				try:
					temp[i,int(round(q*num_time_bins))+shifts[i]] = max_raw_val
				except IndexError:
					pass
		axarr[1].imshow(temp, origin='lower', aspect='auto', \
				extent=[0,max_t,0,len(amp_traces)])
		temp = np.copy(aligned)
		for q in quantiles:
			for i in range(len(temp)):
				temp[i,int(round(q*num_time_bins))] = max_aligned_val
		axarr[2].imshow(temp, origin='lower', aspect='auto', \
				extent=[0,max_t,0,len(amp_traces)])
		axarr[0].set_ylabel("Frequency (kHz)")
		axarr[1].set_ylabel('Amplitude')
		axarr[2].set_ylabel('Shifted')
		axarr[0].set_title('Enter segmenting quantiles:')
		axarr[2].set_xlabel('Time (ms)')
		plt.savefig(img_fn)
		plt.close('all')
		# Ask for segmenting decisions.
		while True:
			temp = input("Add or delete quantile or [s]top: ")
			if temp == 's':
				break_flag = True
				break
			try:
				temp = float(temp)
				assert 0.0 < temp and temp < 1.0
				if temp in quantiles:
					quantiles.remove(temp)
				else:
					quantiles.append(temp)
				break
			except:
				print("Invalid input!")
				print("Must be \'s\' or a float between 0 and 1.")
				continue
		if break_flag:
			break
	# Write syllable segments.
	if verbose:
		print("Writing syllable segments...")
	duration = num_time_bins * dt
	quantiles = np.array(quantiles)
	quantiles.sort()
	files_encountered = {}
	for i, (fn, song_onset) in enumerate(zip(fns, song_onsets)):
		# Unshifted onsets and offsets.
		onsets = song_onset + duration * quantiles[:-1]
		offsets = song_onset + duration * quantiles[1:]
		# Apply shifts.
		onsets += shifts[i] * dt
		offsets += shifts[i] * dt
		# Save.
		index = audio_dirs.index(os.path.split(fn)[0])
		write_fn = os.path.join(syll_seg_dirs[index], os.path.split(fn)[-1])
		write_fn = write_fn[:-4] + '.txt'
		if not os.path.exists(os.path.split(write_fn)[0]):
			os.makedirs(os.path.split(write_fn)[0])
		segs = np.stack([onsets, offsets]).reshape(2,-1).T
		header, mode = "", 'ab'
		if fn not in files_encountered:
			files_encountered[fn] = 1
			mode = 'wb'
			header += "Syllables from song: " + fn + "\n"
		header += "Song onset: "+str(song_onset)
		with open(write_fn, mode) as f:
			np.savetxt(f, segs, fmt='%.5f', header=header)
	# Write empty files corresponding to audio files without song.
	for fn in empty_audio_files:
		index = audio_dirs.index(os.path.split(fn)[0])
		write_fn = os.path.join(syll_seg_dirs[index], os.path.split(fn)[-1])
		write_fn = write_fn[:-4] + '.txt'
		if not os.path.exists(os.path.split(write_fn)[0]):
			os.makedirs(os.path.split(write_fn)[0])
		header = "Syllables from song: " + fn
		np.savetxt(write_fn, np.array([]), header=header)


def segment_sylls_from_warped_songs(warped_window_dset, audio_dirs, spec_dirs, \
	time_bins=512, num_specs=3, img_fn='temp.pdf', verbose=True):
	"""
	Split time-warped song renditions into time-warped syllables, save specs.

	Enter quantiles to determine where to split the song motif. Entering the
	same quantile twice will remove it.

	Parameters
	----------
	warped_window_dset : ava.models.window_vae_dataset.WarpedWindowDataset
		Dataset defining a warping.
	audio_dirs : list of str
		Audio directories.
	spec_dirs : list of str
		Spectrogram directories.
	time_bins : int, optional
		Number of spectrogram time bins to plot.
	num_specs : int, optional
		Number of spectrograms to plot. Defaults to `1`.
	img_fn : str, optional
		Image filename. Defaults to ``'temp.pdf'``.
	verbose : bool, optional
		Defaults to `True`.
	"""
	# Check the input.
	audio_dir_to_spec_dir = dict(zip(audio_dirs,spec_dirs))
	for audio_fn in warped_window_dset.audio_filenames:
		assert os.path.split(audio_fn)[0] in audio_dirs, "Cannot find " + \
				os.path.split(audio_fn)[0] + " in audio_dirs!"
	# Collect segmenting quantiles.
	start_q, stop_q = warped_window_dset.start_q, warped_window_dset.stop_q
	error_msg = "Invalid input!\nMust be \'s\' or a float between " + \
			"{0:.2f}".format(start_q) + " and " + "{0:.2f}".format(stop_q) + "."
	p, fs = warped_window_dset.p, warped_window_dset.fs
	quantiles = []
	break_flag = False
	while True:
		# Plot.
		_, axarr = plt.subplots(nrows=num_specs, sharex=True)
		if num_specs == 1:
			axarr = [axarr]
		axarr[0].set_title('Enter segmenting quantiles:')
		for i in range(num_specs):
			plt.sca(axarr[i])
			index = np.random.randint(len(warped_window_dset.audio_filenames))
			audio_fn = warped_window_dset.audio_filenames[index]
			warped_spec = warped_window_dset.get_whole_warped_spectrogram( \
					audio_fn, time_bins=time_bins)
			plt.imshow(warped_spec, origin='lower', aspect='auto', \
					extent=[start_q,stop_q,p['min_freq']/1e3,p['max_freq']/1e3])
			for q in quantiles:
				plt.axvline(x=q, color='red')
			plt.ylabel("Frequency (kHz)")
		plt.xlabel('Warped Time Quantile')
		plt.savefig(img_fn)
		plt.close('all')
		# Ask for segmenting decisions.
		while True:
			temp = input("Add or delete quantile or [s]top: ")
			if temp == '':
				break
			elif temp == 's':
				break_flag = True
				break
			try:
				temp = float(temp)
				assert start_q < temp and temp < stop_q
				if temp in quantiles:
					quantiles.remove(temp)
				else:
					quantiles.append(temp)
				break
			except:
				print(error_msg)
				continue
		if break_flag:
			break
	assert len(quantiles) > 1, "Not enough quantiles to segment!"
	# Write syllable spectrograms.
	if verbose:
		print("Making and saving syllable spectrograms...")
	quantiles = sorted(quantiles)
	segs = [[q1,q2] for q1, q2 in zip(quantiles[:-1],quantiles[1:])]
	num_saved = 0
	template_dur = warped_window_dset.template_dur
	for audio_fn in warped_window_dset.audio_filenames:
		syll_data = {
			'specs':[],
			'onsets':[],
			'offsets':[],
			'audio_filenames':[],
		}
		for q1, q2 in segs:
			syll_data['onsets'].append(q1) # Quantiles are saved, not times.
			syll_data['offsets'].append(q2)
			syll_data['audio_filenames'].append(audio_fn)
			# Make spectrogram.
			index = warped_window_dset.audio_filenames.index(audio_fn)
			t_vals = np.linspace(q1, q2, p['num_time_bins'])
			# Inverse warp.
			target_ts = warped_window_dset._get_unwarped_times(t_vals, index)
			target_ts *= template_dur
			# Then make a spectrogram.
			spec, flag = p['get_spec'](0.0, template_dur, \
					warped_window_dset.audio[index], p, fs=fs, max_dur=None, \
					target_times=target_ts)
			assert flag
			syll_data['specs'].append(spec)
			num_saved += 1
		audio_dir = os.path.split(audio_fn)[0]
		write_fn = os.path.split(audio_fn)[-1][:-4] + '.hdf5'
		spec_dir = audio_dir_to_spec_dir[audio_dir]
		if not os.path.exists(spec_dir):
			os.makedirs(spec_dir)
		write_fn = os.path.join(spec_dir, write_fn)
		with h5py.File(write_fn, "w") as f:
			# Add all the fields.
			for key in ['onsets', 'offsets']:
				f.create_dataset(key, data=np.array(syll_data[key]))
			f.create_dataset('specs', data=np.stack(syll_data['specs']))
			f.create_dataset('audio_filenames', \
					data=np.array(syll_data['audio_filenames']).astype('S'))
	if verbose:
		print("\tSaved "+str(num_saved)+" spectrograms.")
		print("\tDone.")


def _get_spec(fs, audio, p):
	"""
	Get a spetrogram. Not many options here.

	Parameters
	----------
	fs : float
		Samplerate.
	audio : numpy.ndarray
		Raw audio.
	p : dict
		Parameters. Must contain keys: ``'nperseg'``, ``'noverlap'``,
		``'min_freq'``, ``'max_freq'``, ``'spec_min_val'``, and
		``'spec_max_val'``.

	Returns
	-------
	spec : numpy.ndarray
		Spectrogram.
	dt : float
		Timestep.
	"""
	f, t, spec = stft(audio, fs=fs, nperseg=p['nperseg'], \
			noverlap=p['noverlap'])
	f1 = np.searchsorted(f, p['min_freq'])
	f2 = np.searchsorted(f, p['max_freq'])
	spec = spec[f1:f2]
	spec = np.log(np.abs(spec) + EPSILON)
	spec -= p['spec_min_val']
	spec /= p['spec_max_val'] - p['spec_min_val']
	spec[spec<0.0] = 0.0
	spec[spec>1.0] = 1.0
	return spec, t[1]-t[0]


def _clean_max_indices(old_indices, old_times, values, min_dt=0.05):
	"""Remove maxima that are too close together."""
	if len(old_indices) <= 1:
		return old_indices
	old_indices = old_indices[np.argsort(values[old_indices])]
	indices = [old_indices[0]]
	times = [old_times[old_indices[0]]]
	i = 1
	while i < len(old_indices):
		time = old_times[old_indices[i]]
		flag = True
		for j in range(len(indices)):
			if abs(old_times[indices[j]] - time) < min_dt:
				flag = False
				break
		if flag:
			indices.append(old_indices[i])
			times.append(old_times[old_indices[i]])
		i += 1
	indices = np.array(indices)
	indices.sort()
	return indices


def _in_region(point, bounds):
	"""Is the point in the union of the given rectangles?"""
	for i in range(len(bounds['x1s'])):
		x_min = min(bounds['x1s'][i], bounds['x2s'][i])
		x_max = max(bounds['x1s'][i], bounds['x2s'][i])
		y_min = min(bounds['y1s'][i], bounds['y2s'][i])
		y_max = max(bounds['y1s'][i], bounds['y2s'][i])
		if point[0] > x_min and point[0] < x_max and point[1] > y_min and \
				point[1] < y_max:
			return True
	return False


def _is_wav_file(filename):
	return len(filename) > 4 and filename[-4:] == '.wav'


def _is_number(answer):
	try:
		answer = float(answer)
		return True
	except:
		if answer != 'initial input':
			print("Invalid input!")
		return False



if __name__ == '__main__':
	pass


###
