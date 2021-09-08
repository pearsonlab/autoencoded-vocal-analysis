"""
Useful functions for segmenting.

"""
__date__ = "August 2019 - March 2021"


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os
from scipy.signal import stft
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning
import warnings


EPSILON = 1e-9



def get_spec(audio, p):
	"""
	Get a spectrogram.

	Much simpler than ``ava.preprocessing.utils.get_spec``.

	Raises
	------
	- ``AssertionError`` if ``len(audio) < p['nperseg']``.

	Parameters
	----------
	audio : numpy array of floats
		Audio
	p : dict
		Spectrogram parameters. Should the following keys: `'fs'`, `'nperseg'`,
		`'noverlap'`, `'min_freq'`, `'max_freq'`, `'spec_min_val'`,
		`'spec_max_val'`

	Returns
	-------
	spec : numpy array of floats
		Spectrogram of shape [freq_bins x time_bins]
	dt : float
		Time step between time bins.
	f : numpy.ndarray
		Array of frequencies.
	"""
	assert len(audio) >= p['nperseg'], \
			"len(audio): " + str(len(audio)) + ", nperseg: " + str(p['nperseg'])
	f, t, spec = stft(audio, fs=p['fs'], nperseg=p['nperseg'], \
			noverlap=p['noverlap'])
	i1 = np.searchsorted(f, p['min_freq'])
	i2 = np.searchsorted(f, p['max_freq'])
	f, spec = f[i1:i2], spec[i1:i2]
	spec = np.log(np.abs(spec) + EPSILON)
	spec -= p['spec_min_val']
	spec /= p['spec_max_val'] - p['spec_min_val']
	spec = np.clip(spec, 0.0, 1.0)
	return spec, t[1]-t[0], f


def clean_segments_by_hand(audio_dirs, orig_seg_dirs, new_seg_dirs, p, \
	nrows=4, ncols=4, shoulder=0.1, select_to_reject=True, \
	img_filename='temp.pdf'):
	"""
	Plot spectrograms and ask for accept/reject input.

	The accepted segments are taken from `orig_seg_dirs` and copied to
	`new_seg_dirs`.

	Notes
	-----
	* Enter indices of false positive spectrograms (or if `select_to_reject` is
	  `False`, true positive spectrograms) separated by spaces.
	* This will not overwrite existing segmentation files and will raise an
	  `AssertionError` if any of the files already exist.

	Parameters
	----------
	audio_dirs : list of str
		Audio directories.
	orig_seg_dirs : list of str
		Original segment directories.
	new_seg_dirs : list of str
		New segment directories.
	p : dict
		Parameters. Should the following keys: `'fs'`, `'nperseg'`,
		`'noverlap'`, `'min_freq'`, `'max_freq'`, `'spec_min_val'`,
		`'spec_max_val'`
	nrows : int, optional
		Number of rows of spectrograms to plot. Defaults to ``4``.
	ncols : int, optional
		Number of columns of spectrograms to plot. Defaults to ``4``.
	shoulder : float, optional
		Duration of audio to plot on either side of segment. Defaults to `0.1`.
	select_to_reject : bool, optional
		If ``True``, the user is asked to identify false positives. Else, the
		user is asked to identify true positives. Defaults to ``True``.
	img_filename : str, optional
		Where to write images. Defaults to ``'temp.pdf'``.
	"""
	# Make new directories, if needed.
	for new_seg_dir in new_seg_dirs:
		if not os.path.exists(new_seg_dir):
			os.makedirs(new_seg_dir)
	# Collect all the filenames.
	audio_fns, orig_seg_fns = get_audio_seg_filenames(audio_dirs, orig_seg_dirs)
	temp_dict = dict(zip(orig_seg_dirs, new_seg_dirs))
	new_seg_fns = []
	for orig_seg_fn in orig_seg_fns:
		a,b = os.path.split(orig_seg_fn)
		new_seg_fns.append(os.path.join(temp_dict[a], b))
	for new_seg_fn in new_seg_fns:
		assert not os.path.isfile(new_seg_fn), "File already exists: " + \
				new_seg_fn
	# Collect all of the segments.
	all_onsets, all_offsets = [], []
	all_audio_fns, all_orig_seg_fns, all_new_seg_fns = [], [], []
	gen = zip(audio_fns, orig_seg_fns, new_seg_fns)
	for audio_fn, orig_seg_fn, new_seg_fn in gen:
		segs = np.loadtxt(orig_seg_fn).reshape(-1,2)
		header = "Onsets/offsets cleaned by hand from " + orig_seg_fn
		np.savetxt(new_seg_fn, np.array([]), header=header)
		onsets, offsets = segs[:,0], segs[:,1]
		all_onsets += onsets.tolist()
		all_offsets += offsets.tolist()
		all_audio_fns += [audio_fn]*len(segs)
		all_orig_seg_fns += [orig_seg_fn]*len(segs)
		all_new_seg_fns += [new_seg_fn]*len(segs)
	# Loop through the segments, asking for accept/reject descisions.
	index = 0
	while index < len(all_onsets):
		print("orig_seg_fn:", all_orig_seg_fns[index])
		print(str(index)+"/"+str(len(all_onsets))+" reviewed")
		num_specs = min(len(all_onsets) - index, nrows*ncols)
		_, axarr = plt.subplots(nrows=nrows, ncols=ncols)
		axarr = axarr.flatten()
		# Plot spectrograms.
		for i in range(num_specs):
			if i == 0 or all_audio_fns[index+i] != all_audio_fns[index+i-1]:
				audio_fn = all_audio_fns[index+i]
				orig_seg_fn = all_orig_seg_fns[index+i]
				new_seg_fn = all_new_seg_fns[index+i]
				# Get spectrogram.
				with warnings.catch_warnings():
					warnings.filterwarnings("ignore", category=WavFileWarning)
					fs, audio = wavfile.read(audio_fn)
				assert fs == p['fs'], "Found fs="+str(fs)+", expected fs="+\
						str(p['fs'])
				spec, dt, f = get_spec(audio, p)
			onset, offset = all_onsets[index+i], all_offsets[index+i]
			i1 = max(0, int((onset - shoulder) / dt))
			i2 = min(spec.shape[1], int((offset + shoulder) / dt))
			t1 = max(0, onset-shoulder)
			t2 = min(len(audio)/fs, offset+shoulder)
			plt.sca(axarr[i])
			plt.imshow(spec[:,i1:i2], origin='lower', aspect='auto', \
					extent=[t1, t2, f[0]/1e3, f[-1]/1e3])
			plt.title(str(i))
			plt.axis('off')
			plt.axvline(x=onset, c='r')
			plt.axvline(x=offset, c='r')
		plt.tight_layout()
		plt.savefig(img_filename)
		plt.close('all')
		# Collect user input.
		response = 'invalid response'
		while not _is_valid_response(response, num_specs):
			if select_to_reject:
				response = input("List bad spectrograms: ")
			else:
				response = input("List good spectrograms: ")
		if response == '':
			if select_to_reject:
				good_indices = [index+i for i in range(num_specs)]
			else:
				good_indices = []
		else:
			responses = [int(i) for i in response.split(' ')]
			good_indices = []
			for i in range(num_specs):
				if select_to_reject and i not in responses:
					good_indices.append(index+i)
				elif not select_to_reject and i in responses:
					good_indices.append(index+i)
			good_indices = np.unique( \
					np.array(good_indices, dtype='int')).tolist()
		# Copy the good segments.
		for i in range(num_specs):
			if index + i in good_indices:
				with open(all_new_seg_fns[index+i], 'ab') as f:
					seg = np.array([all_onsets[index+i], all_offsets[index+i]])
					np.savetxt(f, seg.reshape(1,2), fmt='%.5f')
		index += num_specs


def copy_segments_to_standard_format(orig_seg_dirs, new_seg_dirs, seg_ext, \
	delimiter, usecols, skiprows, max_duration=None):
	"""
	Copy onsets/offsets from SAP, MUPET, or Deepsqueak into a standard format.

	Note
	----
	- `delimiter`, `usecols`, and `skiprows` are all passed to `numpy.loadtxt`.

	Parameters
	----------
	orig_seg_dirs : list of str
		Directories containing original segments.
	new_seg_dirs : list of str
		Corresponding directories for new segments.
	seg_ext : str
		Input filename extension.
	delimiter : str
		Input filename delimiter. For a CSV file, for example, this would be a
		comma: `','`
	usecols : tuple
		Input file onset and offset columns, zero-indexed.
	skiprows : int
		Number of rows to skip. For example, if there is a single-line header
		set `skiprows=1`.
	max_duration : {None, float}, optional
		Maximum segment duration. If None, no max is set. Defaults to `None`.
	"""
	assert len(seg_ext) == 4
	assert len(orig_seg_dirs) == len(new_seg_dirs), \
			f"{len(orig_seg_dirs)} != {len(new_seg_dirs)}"
	assert len(usecols) == 2, "Expected two columns (for onsets and offsets)!"
	for orig_seg_dir, new_seg_dir in zip(orig_seg_dirs, new_seg_dirs):
		if not os.path.exists(new_seg_dir):
			os.makedirs(new_seg_dir)
		seg_fns = [os.path.join(orig_seg_dir,i) for i in \
				os.listdir(orig_seg_dir) if len(i) > 4 and i[-4:] == seg_ext]
		for seg_fn in seg_fns:
			segs = np.loadtxt(seg_fn, delimiter=delimiter, skiprows=skiprows, \
					usecols=usecols).reshape(-1,2)
			if max_duration is not None:
				new_segs = []
				for seg in segs:
					if seg[1]-seg[0] < max_duration:
						new_segs.append(seg)
				if len(new_segs) > 0:
					segs = np.stack(new_segs)
				else:
					segs = np.array([])
			new_seg_fn = os.path.join(new_seg_dir, os.path.split(seg_fn)[-1])
			new_seg_fn = new_seg_fn[:-4] + '.txt'
			header = "Onsets/offsets copied from "+seg_fn
			np.savetxt(new_seg_fn, segs, fmt='%.5f', header=header)


def write_segments_to_audio(in_audio_dirs, out_audio_dirs, seg_dirs, n_zfill=3,\
	verbose=True):
	"""
	Write each segment as its own audio file.

	Parameters
	----------
	in_audio_dirs : list of str
		Where to read audio.
	out_audio_dirs : list of str
		Where to write audio.
	seg_dirs : list of str
		Where to read segments.
	n_zfill : int, optional
		For filename formatting. Defaults to ``3``.
	verbose : bool, optional
		Deafults to ``True``.
	"""
	assert len(in_audio_dirs) == len(out_audio_dirs), \
			f"{len(in_audio_dirs)} != {len(out_audio_dirs)}"
	assert len(in_audio_dirs) == len(seg_dirs), \
			f"{len(in_audio_dirs)} != {len(seg_dirs)}"
	if verbose:
		print("Writing segments to audio,", len(in_audio_dirs), "directories")
	for in_dir, out_dir, seg_dir in zip(in_audio_dirs, out_audio_dirs, seg_dirs):
		seg_fns = [j for j in sorted(os.listdir(seg_dir)) if _is_txt_file(j)]
		audio_fns = [os.path.join(in_dir, j[:-4]+'.wav') for j in seg_fns]
		seg_fns = [os.path.join(seg_dir, j) for j in seg_fns]
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
		for seg_fn, audio_fn in zip(seg_fns, audio_fns):
			segs = np.loadtxt(seg_fn).reshape(-1,2)
			if len(segs) == 0:
				continue
			with warnings.catch_warnings():
				warnings.filterwarnings("ignore", category=WavFileWarning)
				fs, audio = wavfile.read(audio_fn)
			for j in range(segs.shape[0]):
				num_samples = int(round(fs * (segs[j,1]-segs[j,0])))
				i1 = int(round(fs * segs[j,0]))
				out_audio = audio[i1:i1+num_samples]
				out_audio_fn = os.path.split(audio_fn)[-1][:-4]
				out_audio_fn += '_' + str(j).zfill(n_zfill) + '.wav'
				out_audio_fn = os.path.join(out_dir, out_audio_fn)
				wavfile.write(out_audio_fn, fs, out_audio)
	if verbose:
		print("\tDone.")


def merge_segments(orig_seg_dirs, new_seg_dirs, merge_threshold, \
	left_shoulder=0.0, right_shoulder=0.0, min_duration=0.0, verbose=True):
	"""
	Merge nearby segments into larger segments.

	Parameters
	----------
	orig_seg_dirs : list of str
		Directories containing original segments.
	new_seg_dirs : list of str
		Corresponding directories for new segments.
	merge_threshold : float
		All segments closer than this duration are merged.
	left_shoulder : float, optional
		Extra time to add before merged segments. Defaults to `0.0`
	right_shoulder : float, optional
		Extra time to add after merged segments. Defaults to `0.0`.
	min_duration : float, optional
		Minumum duration of a merged segment. Defaults to `0.0`.
	"""
	assert len(orig_seg_dirs) == len(new_seg_dirs), \
			f"{len(orig_seg_dirs)} != {len(new_seg_dirs)}"
	if verbose:
		print("Merging segments:")
	# Make new directories, if needed.
	for new_seg_dir in new_seg_dirs:
		if not os.path.exists(new_seg_dir):
			os.makedirs(new_seg_dir)
	# Merge segments.
	for orig_seg_dir, new_seg_dir in zip(orig_seg_dirs, new_seg_dirs):
		orig_seg_fns = [os.path.join(orig_seg_dir,i) for i in \
				os.listdir(orig_seg_dir) if _is_txt_file(i)]
		new_seg_fns = [os.path.join(new_seg_dir,os.path.split(i)[1]) for i in \
				orig_seg_fns]
		for orig_seg_fn, new_seg_fn in zip(orig_seg_fns, new_seg_fns):
			header = "Merged segments from " + orig_seg_fn
			segs = np.loadtxt(orig_seg_fn).reshape(-1,2)
			if len(segs) == 0:
				np.savetxt(new_seg_fn, np.array([]), header=header)
				continue
			# Collect merged onsets/offsets.
			merged_segs = []
			current_onset, current_offset = segs[0,0], segs[0,1]
			for i in range(1,len(segs)):
				if segs[i,0] - current_offset < merge_threshold: # merge
					current_offset = segs[i,1]
				else: # or don't merge
					# Apply shoulders.
					current_onset = max(0.0, current_onset - left_shoulder)
					current_offset = current_offset + right_shoulder
					# Add to segments list.
					merged_segs.append([current_onset, current_offset])
					# Set up next segment.
					current_onset, current_offset = segs[i,0], segs[i,1]
			merged_segs.append([current_onset, current_offset])
			# Delete segments that are too short.
			if min_duration > 0.0:
				for i in reversed(range(len(merged_segs))):
					if merged_segs[i][1] - merged_segs[i][0] < min_duration:
						del merged_segs[i]
			# Save segments.
			merged_segs = np.array(merged_segs).reshape(-1,2)
			np.savetxt(new_seg_fn, merged_segs, fmt='%.5f', header=header)
	if verbose:
		print("\tDone.")


def get_audio_seg_filenames(audio_dirs, seg_dirs):
	"""
	Return lists of audio filenames and corresponding segment filenames.

	Parameters
	----------
	audio_dirs : list of str
		Audio directories
	seg_dirs : list of str
		Corresponding segmenting directories

	Returns
	-------
	audio_fns : list of str
		Audio filenames
	seg_fns : list of str
		Corresponding segment filenames
	"""
	assert len(audio_dirs) == len(seg_dirs), \
			f"{len(audio_dirs)} != {len(seg_dirs)}"
	audio_fns, seg_fns = [], []
	for audio_dir, seg_dir in zip(audio_dirs, seg_dirs):
		temp_fns = [i for i in sorted(os.listdir(audio_dir)) if \
				_is_audio_file(i)]
		audio_fns += [os.path.join(audio_dir, i) for i in temp_fns]
		temp_fns = [i[:-4] + '.txt' for i in temp_fns]
		seg_fns += [os.path.join(seg_dir, i) for i in temp_fns]
	return audio_fns, seg_fns


def softmax(arr, t=0.5):
	"""Softmax along first array dimension. Not numerically stable."""
	temp = np.exp(arr/t)
	temp /= np.sum(temp, axis=0) + EPSILON
	return np.sum(np.multiply(arr, temp), axis=0)


def _read_onsets_offsets(filename):
	"""
	A wrapper around numpy.loadtxt for reading onsets & offsets.

	Parameters
	----------
	filename : str
		Filename of a text file containing one header line and two columns.

	Returns
	-------
	onsets : numpy.ndarray
		Onset times.
	offsets : numpy.ndarray
		Offset times.
	"""
	arr = np.loadtxt(filename, skiprows=1)
	if len(arr) == 0:
		return [], []
	if len(arr.shape) == 1:
		arr = arr.reshape(1,2)
	assert arr.shape[1] == 2, "Found invalid shape: "+str(arr.shape)
	return arr[:,0], arr[:,1]


def _is_audio_file(filename):
	return filename.endswith('.wav')


def _is_txt_file(filename):
	return filename.endswith('.txt')


def _is_valid_response(response, num_specs):
	if response == '':
		return True
	try:
		responses = [int(i) for i in response.split(' ')]
		return min(responses) >= 0 and max(responses) < num_specs
	except:
		return False



if __name__ == '__main__':
	pass


###
