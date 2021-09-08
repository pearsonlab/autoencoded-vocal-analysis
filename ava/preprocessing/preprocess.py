"""
Make and save syllable spectrograms.

"""
__date__ = "December 2018 - July 2020"


import h5py
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning
import warnings

from ava.preprocessing.utils import _mel, _inv_mel

EPSILON = 1e-12



def process_sylls(audio_dir, segment_dir, save_dir, p, shuffle=True, \
	verbose=True):
	"""
	Extract syllables from `audio_dir` and save to `save_dir`.

	Parameters
	----------
	audio_dir : str
		Directory containing audio files.
	segment_dir : str
		Directory containing segmenting decisions.
	save_dir : str
		Directory to save processed syllables in.
	p : dict
		Preprocessing parameters. TO DO: add reference.
	shuffle : bool, optional
		Shuffle by filename. Defaults to ``True``.
	verbose : bool, optional
		Defaults to ``True``.
	"""
	if verbose:
		print("Processing audio files in", audio_dir)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	audio_filenames, seg_filenames = \
			get_audio_seg_filenames(audio_dir, segment_dir, p)
	if shuffle:
		np.random.seed(42)
		perm = np.random.permutation(len(audio_filenames))
		np.random.seed(None)
		audio_filenames = np.array(audio_filenames)[perm]
		seg_filenames = np.array(seg_filenames)[perm]
	write_file_num = 0
	syll_data = {
		'specs':[],
		'onsets':[],
		'offsets':[],
		'audio_filenames':[],
	}
	sylls_per_file = p['sylls_per_file']
	# For each pair of files...
	for audio_filename, seg_filename in zip(audio_filenames, seg_filenames):
		# Get onsets and offsets.
		onsets, offsets = read_onsets_offsets_from_file(seg_filename, p)
		# Retrieve a spectrogram for each detected syllable.
		specs, good_sylls = get_syll_specs(onsets, offsets, audio_filename, p)
		onsets = [onsets[i] for i in good_sylls]
		offsets = [offsets[i] for i in good_sylls]
		# Add the syllables to <syll_data>.
		syll_data['specs'] += specs
		syll_data['onsets'] += onsets
		syll_data['offsets'] += offsets
		syll_data['audio_filenames'] += \
				len(onsets)*[os.path.split(audio_filename)[-1]]
		# Write files until we don't have enough syllables.
		while len(syll_data['onsets']) >= sylls_per_file:
			save_filename = \
					"syllables_" + str(write_file_num).zfill(4) + '.hdf5'
			save_filename = os.path.join(save_dir, save_filename)
			with h5py.File(save_filename, "w") as f:
				# Add all the fields.
				for key in ['onsets', 'offsets']:
					f.create_dataset(key, \
							data=np.array(syll_data[key][:sylls_per_file]))
				f.create_dataset('specs', \
						data=np.stack(syll_data['specs'][:sylls_per_file]))
				temp = [os.path.join(audio_dir, i) for i in \
						syll_data['audio_filenames'][:sylls_per_file]]
				f.create_dataset('audio_filenames', \
						data=np.array(temp).astype('S'))
			write_file_num += 1
			# Remove the written data from temporary storage.
			for key in syll_data:
				syll_data[key] = syll_data[key][sylls_per_file:]
			# Stop if we've written `max_num_syllables`.
			if p['max_num_syllables'] is not None and \
					write_file_num*sylls_per_file >= p['max_num_syllables']:
				if verbose:
					print("\tSaved max_num_syllables (" + \
							str(p['max_num_syllables'])+"). Returning.")
				return
	if verbose:
		print("\tDone.")


def get_syll_specs(onsets, offsets, audio_filename, p):
	"""
	Return the spectrograms corresponding to `onsets` and `offsets`.

	Parameters
	----------
	onsets : list of floats
		Syllable onsets.
	offsets : list of floats
		Syllable offsets.
	audio_filename : str
		Audio filename.
	p : dict
		A dictionary mapping preprocessing parameters to their values. NOTE: ADD
		REFERENCE HERE!

	Returns
	-------
	specs : list of {numpy.ndarray, None}
		Spectrograms.
	valid_syllables : list of int
		Indices of `specs` containing valid syllables.
	"""
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", category=WavFileWarning)
		fs, audio = wavfile.read(audio_filename)
	assert p['nperseg'] % 2 == 0 and p['nperseg'] > 2
	if p['mel']:
		target_freqs = np.linspace( \
				_mel(p['min_freq']), _mel(p['max_freq']), p['num_freq_bins'])
		target_freqs = _inv_mel(target_freqs)
	else:
		target_freqs = np.linspace( \
				p['min_freq'], p['max_freq'], p['num_freq_bins'])
	specs, valid_syllables = [], []
	# For each syllable...
	for i, t1, t2 in zip(range(len(onsets)), onsets, offsets):
		spec, valid = p['get_spec'](t1, t2, audio, p, fs, \
				target_freqs=target_freqs)
		if valid:
			valid_syllables.append(i)
			specs.append(spec)
	return specs, valid_syllables


def tune_syll_preprocessing_params(audio_dirs, seg_dirs, p, img_fn='temp.pdf'):
	"""
	Flip through spectrograms and tune preprocessing parameters.

	Parameters
	----------
	audio_dirs : list of str
		Audio directories
	seg_dirs : list of str
		Segment directories
	p : dict
		Preprocessing parameters: Add a reference!

	Returns
	-------
	p : dict
		Adjusted preprocessing parameters.
	"""
	print("Tune preprocessing parameters:")

	# Collect all the relevant filenames.
	audio_filenames, seg_filenames = [], []
	for audio_dir, seg_dir in zip(audio_dirs, seg_dirs):
		temp_audio, temp_seg = get_audio_seg_filenames(audio_dir, seg_dir, p)
		audio_filenames += temp_audio
		seg_filenames += temp_seg
	audio_filenames = np.array(audio_filenames)
	seg_filenames = np.array(seg_filenames)
	assert len(audio_filenames) > 0, "Didn't find any audio files!"

	# Main loop: keep tuning parameters ...
	while True:

		# Tune parameters.
		p = _tune_input_helper(p)

		# Keep plotting example spectrograms.
		temp = 'not (s or r)'
		while temp != 's' and temp != 'r':

			# Grab a random file.
			file_index = np.random.randint(len(audio_filenames))
			audio_filename = audio_filenames[file_index]
			seg_filename = seg_filenames[file_index]

			# Grab a random syllable from within the file.
			onsets, offsets = read_onsets_offsets_from_file(seg_filename, p)
			if len(onsets) == 0:
				continue
			syll_index = np.random.randint(len(onsets))
			onsets, offsets = [onsets[syll_index]], [offsets[syll_index]]

			# Get the preprocessed spectrogram.
			specs, good_sylls = get_syll_specs(onsets, offsets, \
					audio_filename, p)
			specs = [specs[i] for i in good_sylls]
			if len(specs) == 0:
				continue
			spec = specs[np.random.randint(len(specs))]

			# Plot.
			plt.imshow(spec, aspect='equal', origin='lower', vmin=0, vmax=1)
			plt.axis('off')
			plt.savefig(img_fn)
			plt.close('all')
			temp = input('Continue? [y] or [s]top tuning or [r]etune params: ')
			if temp == 's':
				return p


def tune_window_preprocessing_params(audio_dirs, p, img_fn='temp.pdf'):
	"""
	Flip through spectrograms and tune preprocessing parameters.

	Parameters
	----------
	audio_dirs : list of str
		Audio directories
	p : dict
		Preprocessing parameters ADD REFERENCE
	img_fn : str, optional
		Where to save images. Defaults to ``'temp.pdf'``.

	Returns
	-------
	p : dict
		Adjusted preprocessing parameters.
	"""
	print("Tune preprocessing parameters:")

	# Collect all the relevant filenames.
	audio_filenames = []
	for audio_dir in audio_dirs:
		audio_filenames += get_audio_filenames(audio_dir)
	audio_filenames = np.array(audio_filenames)

	# Main loop: keep tuning parameters ...
	while True:

		# Tune parameters.
		p = _tune_input_helper(p)

		# Keep plotting example spectrograms.
		temp = 'not (s or r)'
		while temp != 's' and temp != 'r':

			# Grab a random file.
			file_index = np.random.randint(len(audio_filenames))
			audio_filename = audio_filenames[file_index]
			with warnings.catch_warnings():
				warnings.filterwarnings("ignore", category=WavFileWarning)
				fs, audio = wavfile.read(audio_filename)
			assert fs == p['fs'], "Found fs="+str(fs)+", expected "+str(p['fs'])

			# Get a random onset & offset.
			duration = len(audio) / fs
			assert duration > p['window_length']
			onset = np.random.rand() * (duration - p['window_length'])
			offset = onset + p['window_length']
			target_times = np.linspace(onset, offset, p['num_time_bins'])

			# Get the preprocessed spectrogram.
			spec, flag = p['get_spec'](0.0, duration, audio, p, fs=p['fs'], \
					max_dur=None, target_times=target_times)
			assert flag

			# Plot.
			plt.imshow(spec, aspect='equal', origin='lower', vmin=0, vmax=1)
			plt.axis('off')
			plt.savefig(img_fn)
			plt.close('all')
			temp = input('Continue? [y] or [s]top tuning or [r]etune params: ')
			if temp == 's':
				return p


def _tune_input_helper(p):
	"""Get parameter adjustments from the user."""
	for key in p['real_preprocess_params']:
		temp = 'not (number or empty)'
		while not _is_number_or_empty(temp):
			temp = input('Set value for '+key+': ['+str(p[key])+ '] ')
		if temp != '':
			p[key] = float(temp)
	for key in p['int_preprocess_params']:
		temp = 'not (number or empty)'
		while not _is_number_or_empty(temp):
			temp = input('Set value for '+key+': ['+str(p[key])+ '] ')
		if temp != '':
			p[key] = int(temp)
	for key in p['binary_preprocess_params']:
		temp = 'not (t or f)'
		while temp not in ['t', 'T', 'f', 'F', '']:
			current_value = 'T' if p[key] else 'F'
			temp = input('Set value for '+key+': ['+current_value+'] ')
		if temp != '':
			p[key] = temp in ['t', 'T']
	return p


def get_audio_seg_filenames(audio_dir, segment_dir, p):
	"""Return lists of sorted filenames."""
	# Collect all the audio filenames.
	temp_filenames = [i for i in sorted(os.listdir(audio_dir)) if \
			is_audio_file(i)]
	audio_filenames = [os.path.join(audio_dir, i) for i in temp_filenames]
	temp_filenames = [i[:-4] + '.txt' for i in temp_filenames]
	seg_filenames = [os.path.join(segment_dir, i) for i in temp_filenames]
	# Remove filenames with segments that don't exist.
	for i in range(len(seg_filenames)-1,-1,-1):
		if not os.path.exists(seg_filenames[i]):
			del seg_filenames[i]
			del audio_filenames[i]
	return audio_filenames, seg_filenames


def get_audio_filenames(audio_dir):
	"""Return a list of sorted audio files."""
	fns = [os.path.join(audio_dir, i) for i in sorted(os.listdir(audio_dir)) \
			if is_audio_file(i)]
	return fns


def read_onsets_offsets_from_file(txt_filename, p):
	"""
	Read a text file to collect onsets and offsets.

	Note
	----
	* The text file must have two coulumns separated by whitespace and ``#``
	  prepended to header and footer lines.
	"""
	segs = np.loadtxt(txt_filename)
	assert segs.size % 2 == 0, "Incorrect formatting: " + txt_filename
	segs = segs.reshape(-1,2)
	return segs[:,0], segs[:,1]


def _is_number_or_empty(s):
	if s == '':
		return True
	try:
		float(s)
		return True
	except:
		return False


def _is_number(s):
	return type(s) == type(4) or type(s) == type(4.0)


def is_audio_file(fn):
	"""Return whether the given filename is an audio filename."""
	return len(fn) >= 4 and fn[-4:] == '.wav'



if __name__ == '__main__':
	pass


###
