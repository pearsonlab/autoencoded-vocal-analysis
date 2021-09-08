"""
DataContainer class for linking directories containing different sorts of data.

This is meant to make plotting and analysis easier.

TO DO
-----
- request random subsets.
- make sure input directories are iterable
- add features to existing files.
"""
__date__ = "July 2019 - November 2020"


import h5py
try: # Numba >= 0.52
	from numba.core.errors import NumbaPerformanceWarning
except ModuleNotFoundError:
	try: # Numba <= 0.45
		from numba.errors import NumbaPerformanceWarning
	except (NameError, ModuleNotFoundError):
		pass
import numpy as np
import os
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning
from sklearn.decomposition import PCA
from time import strptime, mktime
import torch
import umap
import warnings

from ava.models.vae import VAE
from ava.models.vae_dataset import get_syllable_partition, \
		get_syllable_data_loaders
from ava.models.utils import get_hdf5s_from_dir


AUDIO_FIELDS = ['audio']
FILENAME_FIELDS = ['sap_time']
SEGMENT_FIELDS = ['segments', 'segment_audio']
PROJECTION_FIELDS = ['latent_means', 'latent_mean_pca', 'latent_mean_umap']
SPEC_FIELDS = ['specs', 'onsets', 'offsets', 'audio_filenames']
MUPET_FIELDS = ['syllable_number', 'syllable_start_time', 'syllable_end_time',
	'inter-syllable_interval', 'syllable_duration', 'starting_frequency',
	'final_frequency', 'minimum_frequency', 'maximum_frequency',
	'mean_frequency', 'frequency_bandwidth', 'total_syllable_energy',
	'peak_syllable_amplitude', 'cluster']
DEEPSQUEAK_FIELDS = ['id', 'label', 'accepted', 'score', 'begin_time',
	'end_time', 'call_length', 'principal_frequency', 'low_freq', 'high_freq',
	'delta_freq', 'frequency_standard_deviation', 'slope', 'sinuosity',
	'mean_power', 'tonality']
SAP_FIELDS = ['syllable_duration_sap', 'syllable_start', 'mean_amplitude',
	'mean_pitch', 'mean_FM', 'mean_AM2', 'mean_entropy', 'mean_pitch_goodness',
	'mean_mean_freq', 'pitch_variance', 'FM_variance', 'entropy_variance',
	'pitch_goodness_variance', 'mean_freq_variance', 'AM_variance']
ALL_FIELDS = AUDIO_FIELDS + FILENAME_FIELDS + SEGMENT_FIELDS + \
	PROJECTION_FIELDS + SPEC_FIELDS + MUPET_FIELDS + DEEPSQUEAK_FIELDS + \
	SAP_FIELDS
"""All fields that can be requested by a DataContainer object."""

MUPET_ONSET_COL = MUPET_FIELDS.index('syllable_start_time')
DEEPSQUEAK_ONSET_COL = DEEPSQUEAK_FIELDS.index('begin_time')
SAP_ONSET_COL = SAP_FIELDS.index('syllable_start')
PRETTY_NAMES = {
	'audio': 'Audio',
	'segments': 'Segments',
	'segment_audio': 'Segment Audio',
	'latent_means': 'Latent Means',
	'latent_mean_pca': 'Latent Mean PCA Projection',
	'latent_mean_umap': 'Latent Mean UMAP Projection',
	'specs': 'Spectrograms',
	'onsets': 'Onsets (s)',
	'offsets': 'Offsets (s)',
	'aduio_filenames': 'Filenames',
	'syllable_number': 'Syllable Number',
	'syllable_start_time': 'Onsets (s)',
	'syllable_duration': 'Duration (ms)',
	'starting_frequency': 'Starting Freq. (kHz)',
	'final_frequency': 'Final Freq. (kHz)',
	'minimum_frequency': 'Min Freq. (kHz)',
	'maximum_frequency': 'Max Freq. (kHz)',
	'mean_frequency': 'Mean Freq. (kHz)',
	'frequency_bandwidth': 'Freq. Bandwidth (kHz)',
	'total_syllable_energy': 'Total Energy (dB)',
	'peak_syllable_amplitude': 'Peak Amplitude (dB)',
	'cluster': 'Cluster',
	'id': 'Syllabler Number',
	'label': 'Label',
	'accepted': 'Accepted',
	'score': 'DeepSqueak Detection Score',
	'begin_time': 'Onsets (s)',
	'end_time': 'Offsets (s)',
	'call_length': 'Duration (ms)',
	'principal_frequency': 'Principal Freq. (kHz)',
	'low_freq': 'Minimum Freq. (kHz)',
	'high_freq': 'Max Freq. (kHz)',
	'delta_freq': 'Freq. Bandwidth (kHz)',
	'frequency_standard_deviation': 'Freq Std. Dev. (kHz)',
	'slope': 'Freq. Mod. (kHz/s)',
	'sinuosity': 'Sinuosity',
	'mean_power': 'Power (dB/Hz)',
	'tonality': 'Tonality',
	'syllable_duration_sap': 'Duration (s)',
	'syllable_start': 'Onset (s)',
	'mean_amplitude': 'Amplitude',
	'mean_pitch': 'Pitch',
	'mean_FM': 'Freq. Mod.',
	'mean_AM2': 'Amp. Mod.',
	'mean_entropy': 'Entropy',
	'mean_pitch_goodness': 'Goodness of Pitch',
	'mean_mean_freq': 'Mean Frequency',
	'pitch_variance': 'Pitch Variance',
	'FM_variance': 'Freq. Mod. Var.',
	'entropy_variance': 'Entropy Var.',
	'pitch_goodness_variance': 'Goodness of Pitch Var.',
	'mean_freq_variance': 'Freq. Var.',
	'AM_variance': 'Amp. Mod. Var.',
}
PRETTY_NAMES_NO_UNITS = {}
for k in PRETTY_NAMES:
	PRETTY_NAMES_NO_UNITS[k] = ' '.join(PRETTY_NAMES[k].split('(')[0].split(' '))



class DataContainer():
	"""
	Link directories containing different data sources for easy plotting.

	The idea here is for plotting and analysis tools to accept a DataContainer,
	from which they can request different types of data. Those requests can then
	be handled here in a central location, which can cut down on redundant code
	and processing steps.

	Attributes
	----------
	audio_dirs : {list of str, None}, optional
		Directories containing audio. Defaults to None.
	segment_dirs : {list of str, None}, optional
		Directories containing segmenting decisions.
	spec_dirs : list of {str, None}, optional
		Directories containing hdf5 files of spectrograms. These should be
		files output by ava.preprocessing.preprocessing. Defaults to None.
	model_filename : {str, None}, optional
		The VAE checkpoint to load. Written by models.vae.save_state.
		Defaults to None.
	projection_dirs : list of {str, None}, optional
		Directory containing different projections. This is where things
		like latent means, their projections, and handcrafted features
		found in feature_dirs are saved. Defaults to None.
	plots_dir : str, optional
		Directory to save plots. Defaults to '' (current working directory).
	feature_dirs : list of {str, None}, optional
		Directory containing text files with different syllable features.
		For exmaple, this could contain exported MUPET, DeepSqueak or SAP
		syllable tables. Defaults to None.
	template_dir : {str, None}, optional
		Directory continaing audio files of song templates. Defaults to
		None.

	Methods
	-------
	request(field)
		Request some type of data.

	Notes
	-----

	Supported directory structure:

	::

		├── animal_1
		│   ├── audio                     (raw audio)
		│   │   ├── foo.wav
		│   │   ├── bar.wav
		│   │   └── baz.wav
		│   ├── features                 (output of MUPET, DeepSqueak, SAP, ...)
		│   │   ├── foo.csv
		│   │   ├── bar.csv
		│   │   └── baz.csv
		│   ├── spectrograms             (used to train models, written by
		│   │   ├── syllables_000.hdf5   preprocessing.preprocess.process_sylls)
		│   │   └── syllables_001.hdf5
		│   └── projections              (latent means, UMAP, PCA, tSNE
		│      ├── syllables_000.hdf5    projections, copies of features in
		│      └── syllables_001.hdf5    experiment_1/features. These are
		│                                written by a DataContainer object.)
		├── animal_2
		│   ├── audio
		│   │   ├── 1.wav
		│   │   └── 2.wav
		│   ├── features
		│   │   ├── 1.csv
		│   │   └── 2.csv
		│   ├── spectrograms
		│   │   ├── syllables_000.hdf5
		│   │   └── syllables_001.hdf5
		│   └── projections
		│       ├── syllables_000.hdf5
		│       └── syllables_001.hdf5
		.
		.
		.


	There should be a 1-to-1 correspondence between, for example, the syllables
	in `animal_1/audio/baz.wav` and the features described in
	`animal_1/features/baz.csv`. Analogously, the fifth entry in
	`animal_2/spectrograms/syllables_000.hdf5` should describe the same syllable
	as the fifth entry in `animal_2/projections/syllables_000.hdf5`. There is no
	strict relationship, however, between individual files in `animal_1/audio`
	and `animal_1/spectrograms`. The hdf5 files in the spectrograms and
	projections directories should contain a subset of the syllables in the
	audio and features directories.

	Then a DataContainer object can be initialized as:

	>>> from ava.data.data_container import DataContainer
	>>> audio_dirs = ['animal_1/audio', 'animal_2/audio']
	>>> spec_dirs = ['animal_1/spectrograms', 'animal_2/spectrograms']
	>>> model_filename = 'checkpoint.tar'
	>>> dc = DataContainer(audio_dirs=audio_dirs, spec_dirs=spec_dirs, \
	model_filename=model_filename)
	>>> latent_means = dc.request('latent_means')

	It's fine to leave some of the initialization parameters unspecified. If the
	DataContainer object is asked to do something it can't, it will hopefully
	complain politely. Or at least informatively.
	"""

	def __init__(self, audio_dirs=None, segment_dirs=None, spec_dirs=None, \
		feature_dirs=None, projection_dirs=None, plots_dir='', \
		model_filename=None, template_dir=None, verbose=True):
		self.audio_dirs = audio_dirs
		self.segment_dirs = segment_dirs
		self.spec_dirs = spec_dirs
		self.feature_dirs = feature_dirs
		self.projection_dirs = projection_dirs
		self.plots_dir = plots_dir
		self.model_filename = model_filename
		self.template_dir = template_dir
		self.verbose = verbose
		self.sylls_per_file = None # syllables in each hdf5 file in spec_dirs
		self.fields = self._check_for_fields()
		if self.plots_dir not in [None, ''] and \
					not os.path.exists(self.plots_dir):
			os.makedirs(self.plots_dir)


	def request(self, field):
		"""
		Request some type of data.

		Parameters
		----------
		field : str
			The type of data being requested. Should come from ...

		Raises
		------
		`NotImplementedError`
			when `field` is not recognized.

		Note
		----
		Besides `__init__` and `clear_projections`, this should be the only
		external-facing method.
		"""
		if field not in ALL_FIELDS:
			print(str(field) + " is not a valid field!")
			raise NotImplementedError
		# If it's not here, make it and return it.
		if field not in self.fields:
			if self.verbose:
				print("Making field:", field)
			data = self._make_field(field)
		# Otherwise, read it and return it.
		else:
			if self.verbose:
				print("Reading field:", field)
			data = self._read_field(field)
		if self.verbose:
			print("\tDone with:", field)
		return data


	def clear_projections(self):
		"""
		Remove all projections.

		This deletes all the ``.hdf5`` files in ``self.projection_dirs``.
		"""
		for proj_dir in self.projection_dirs:
			if not os.path.exists(proj_dir):
				continue
			fns = [os.path.join(proj_dir, i) for i in os.listdir(proj_dir)]
			fns = [i for i in fns if len(i) > 5 and i[-5:] == '.hdf5']
			for fn in fns:
				os.remove(fn)
		self.fields = self._check_for_fields()


	def _make_field(self, field):
		"""Make a field."""
		if field == 'latent_means':
			data = self._make_latent_means()
		elif field == 'latent_mean_pca':
			data = self._make_latent_mean_pca_projection()
		elif field == 'latent_mean_umap':
			data = self._make_latent_mean_umap_projection()
		elif field in MUPET_FIELDS:
			data = self._make_feature_field(field, kind='mupet')
		elif field in DEEPSQUEAK_FIELDS:
			data = self._make_feature_field(field, kind='deepsqueak')
		elif field in SAP_FIELDS:
			data = self._make_feature_field(field, kind='sap')
		elif field in FILENAME_FIELDS:
			data = self._read_filename_field(field)
		elif field == 'specs':
			raise NotImplementedError
		else:
			raise NotImplementedError
		# Add this field to the collection of fields that have been computed.
		self.fields[field] = 1
		if self.verbose:
			print("Making field:", field)
		return data


	def _read_field(self, field):
		"""
		Read a field from memory.

		Parameters
		----------
		field : str
			Field name to read from file. See ``ALL_FIELDS`` for possible
			fields.
		"""
		if field in AUDIO_FIELDS:
			raise NotImplementedError
		elif field == 'segments':
			return self._read_segments()
		elif field == 'segment_audio':
			return self._read_segment_audio()
		elif field in PROJECTION_FIELDS:
			load_dirs = self.projection_dirs
		elif field in SPEC_FIELDS:
			load_dirs = self.spec_dirs
		elif field in MUPET_FIELDS:
			load_dirs = self.projection_dirs
		elif field in DEEPSQUEAK_FIELDS:
			load_dirs = self.projection_dirs
		elif field in SAP_FIELDS:
			load_dirs = self.projection_dirs
		else:
			raise Exception("Can\'t read field: "+field+"\n This should have \
				been caught in self.request!")
		to_return = []
		for i in range(len(self.spec_dirs)):
			spec_dir, load_dir = self.spec_dirs[i], load_dirs[i]
			hdf5s = get_hdf5s_from_dir(spec_dir)
			for j, hdf5 in enumerate(hdf5s):
				filename = os.path.join(load_dir, os.path.split(hdf5)[-1])
				with h5py.File(filename, 'r') as f:
					assert field in f, "Can\'t find field \'"+field+"\' in"+ \
						" file \'"+filename+"\'!"
					if field == 'audio_filenames':
						data = np.array([k.decode('UTF-8') for k in f[field]])
						to_return.append(data)
					else:
						to_return.append(np.array(f[field]))
		return np.concatenate(to_return)


	def _read_segment_audio(self):
		"""
		Read all the segmented audio and return it.

		result[audio_dir][audio_filename] = [audio_1, audio_2, ..., audio_n]
		"""
		self._check_for_dirs(['audio_dirs'], 'audio')
		segments = self.request('segments')
		result = {}
		for audio_dir in self.audio_dirs:
			dir_result = {}
			audio_fns = [i for i in os.listdir(audio_dir) if _is_wav_file(i) \
				and i in segments[audio_dir]]
			for audio_fn in audio_fns:
				with warnings.catch_warnings():
					warnings.filterwarnings("ignore", category=WavFileWarning)
					fs, audio = wavfile.read(os.path.join(audio_dir, audio_fn))
				fn_result = []
				for seg in segments[audio_dir][audio_fn]:
					i1 = int(round(seg[0]*fs))
					i2 = int(round(seg[1]*fs))
					fn_result.append(audio[i1:i2])
				dir_result[audio_fn] = fn_result
			result[audio_dir] = dir_result
		return result


	def _read_segments(self):
		"""
		Return all the segmenting decisions.

		Return a dictionary mapping audio directories to audio filenames to
		numpy arrays of shape [num_segments,2] containing onset and offset
		times.

		TO DO: add support for other delimiters, file extstensions, etc.

		Returns
		-------
		segments : dict
			Maps audio directories to audio filenames to numpy arrays.
		"""
		self._check_for_dirs(['audio_dirs', 'segment_dirs'], 'segments')
		result = {}
		for audio_dir, seg_dir in zip(self.audio_dirs, self.segment_dirs):
			dir_result = {}
			seg_fns = [os.path.join(seg_dir, i) for i in os.listdir(seg_dir) \
				if _is_seg_file(i)]
			audio_fns = [os.path.split(i)[1][:-4]+'.wav' for i in seg_fns]
			for audio_fn, seg_fn in zip(audio_fns, seg_fns):
				segs = _read_columns(seg_fn, delimiter='\t', unpack=False, \
					skiprows=0)
				if len(segs) > 0:
					dir_result[audio_fn] = segs
			result[audio_dir] = dir_result
		return result


	def _make_latent_means(self):
		"""
		Write latent means for the syllables in self.spec_dirs.

		Returns
		-------
		latent_means : numpy.ndarray
			Latent means of shape (max_num_syllables, z_dim)

		Note
		----
		* Duplicated code with ``_write_projection``?
		"""
		self._check_for_dirs(['projection_dirs', 'spec_dirs', 'model_filename'],\
			'latent_means')
		# First, see how many syllables are in each file.
		temp = get_hdf5s_from_dir(self.spec_dirs[0])
		assert len(temp) > 0, "Found no specs in" + self.spec_dirs[0]
		hdf5_file = temp[0]
		with h5py.File(hdf5_file, 'r') as f:
			self.sylls_per_file = len(f['specs'])
		spf = self.sylls_per_file
		# Load the model, making sure to get z_dim correct.
		map_loc = 'cuda' if torch.cuda.is_available() else 'cpu'
		z_dim = torch.load(self.model_filename, map_location=map_loc)['z_dim']
		model = VAE(z_dim=z_dim)
		model.load_state(self.model_filename)
		# For each directory...
		all_latent = []
		for i in range(len(self.spec_dirs)):
			spec_dir, proj_dir = self.spec_dirs[i], self.projection_dirs[i]
			# Make the projection directory if it doesn't exist.
			if proj_dir != '' and not os.path.exists(proj_dir):
				os.makedirs(proj_dir)
			# Make a DataLoader for the syllables.
			partition = get_syllable_partition([spec_dir], 1, shuffle=False)
			try:
				loader = get_syllable_data_loaders(partition, \
					shuffle=(False,False))['train']
				# Get the latent means from the model.
				latent_means = model.get_latent(loader)
				all_latent.append(latent_means)
				# Write them to the corresponding projection directory.
				hdf5s = get_hdf5s_from_dir(spec_dir)
				assert len(latent_means) // len(hdf5s) == spf
				for j in range(len(hdf5s)):
					filename = os.path.join(proj_dir, os.path.split(hdf5s[j])[-1])
					data = latent_means[j*spf:(j+1)*spf]
					with h5py.File(filename, 'a') as f:
						f.create_dataset('latent_means', data=data)
			except AssertionError: # No specs in this directory
				pass
		return np.concatenate(all_latent)


	def _read_filename_field(self, field):
		if field == 'sap_time':
			data = self._make_sap_time()
		else:
			raise NotImplementedError
		return data


	def _make_sap_time(self):
		"""Return time in seconds, following SAP conventions."""
		onsets = self.request('syllable_start')
		fns = self.request('audio_filenames')
		result = np.zeros(lemn(onsets))
		for i, onset, fn in zip(range(len(onsets)), onsets, fns):
			# December 29, 1899, 7pm is the SAP anchor time.
			anchor = mktime(strptime("1899 12 29 19", "%Y %m %d %H"))
			temp = os.path.split(fn)[-1].split('_')[1].split('.')
			day = float(temp[0])
			millisecond = float(temp[1])
			time = anchor + 24*60*60*day + 1e-3*millisecond
			result[i] = time + onset
		return result


	def _make_latent_mean_umap_projection(self):
		"""Project latent means to two dimensions with UMAP."""
		# Get latent means.
		latent_means = self.request('latent_means')
		# UMAP them.
		transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, \
				metric='euclidean', random_state=42)
		if self.verbose:
			print("Running UMAP... (n="+str(len(latent_means))+")")
		# https://github.com/lmcinnes/umap/issues/252
		with warnings.catch_warnings():
			try:
				warnings.filterwarnings("ignore", \
						category=NumbaPerformanceWarning)
			except NameError:
				pass
			embedding = transform.fit_transform(latent_means)
		if self.verbose:
			print("\tDone.")
		# Write to files.
		self._write_projection("latent_mean_umap", embedding)
		return embedding


	def _make_latent_mean_pca_projection(self):
		"""Project latent means to two dimensions with PCA."""
		# Get latent means.
		latent_means = self.request('latent_means')
		# UMAP them.
		transform = PCA(n_components=2, copy=False, random_state=42)
		if self.verbose:
			print("Running PCA...")
		embedding = transform.fit_transform(latent_means)
		if self.verbose:
			print("\tDone.")
		# Write to files.
		self._write_projection("latent_mean_pca", embedding)
		return embedding


	def _make_feature_field(self, field, kind):
		"""
		Read a feature from a text file and put it in an hdf5 file.

		Read from self.feature_dirs and write to self.projection_dirs. This
		could be a bit tricky because we need to match up the syllables in the
		text file with the ones in the hdf5 file.

		Parameters
		----------
		field : str
			Name of data being requested. See ``ALL_FIELDS`` for a complete
			list.
		kind : str, 'mupet' or 'deepsqueak'
			Is this a MUPET or a DeepSqueak field?

		Returns
		-------
		data : numpy.ndarray
			Requested data.
		"""
		self._check_for_dirs( \
			['spec_dirs', 'feature_dirs', 'projection_dirs'], field)
		# Find which column the field is stored in.
		if kind == 'mupet':
			file_fields = MUPET_FIELDS
			onset_col = MUPET_ONSET_COL
		elif kind == 'deepsqueak':
			file_fields = DEEPSQUEAK_FIELDS
			onset_col = DEEPSQUEAK_ONSET_COL
		elif kind == 'sap':
			file_fields = SAP_FIELDS
			onset_col = SAP_ONSET_COL
		else:
			assert NotImplementedError
		field_col = file_fields.index(field)
		to_return = []
		# Run through each directory.
		for i in range(len(self.spec_dirs)):
			spec_dir = self.spec_dirs[i]
			feature_dir = self.feature_dirs[i]
			proj_dir = self.projection_dirs[i]
			hdf5s = get_hdf5s_from_dir(spec_dir)
			current_fn, k = None, None
			for hdf5 in hdf5s:
				# Get the filenames and onsets from self.spec_dirs.
				with h5py.File(hdf5, 'r') as f:
					audio_filenames = np.array(f['audio_filenames'])
					spec_onsets = np.array(f['onsets'])
					# if kind == 'sap': # SAP writes onsets in milliseconds.
					# 	spec_onsets /= 1e3
				feature_arr = np.zeros(len(spec_onsets))
				# Loop through each syllable.
				for j in range(len(spec_onsets)):
					audio_fn, spec_onset = audio_filenames[j], spec_onsets[j]
					audio_fn = audio_fn.decode('UTF-8')
					# Update the feature file, if needed.
					if audio_fn != current_fn:
						current_fn = audio_fn
						feature_fn = os.path.split(audio_fn)[-1][:-4]
						if kind == 'deepsqueak':   # DeepSqueak appends '_Stats'
							feature_fn += '_Stats' # when exporting features.
						feature_fn += '.csv'
						feature_fn = os.path.join(feature_dir, feature_fn)
						# Read the onsets and features.
						feature_onsets, features = \
							_read_columns(feature_fn, [onset_col, field_col])
						if kind == 'sap': # SAP writes onsets in milliseconds.
							feature_onsets /= 1e3
						k = 0
					# Look for the corresponding onset in the feature file.
					while spec_onset > feature_onsets[k] + 0.01:
						k += 1
						assert k < len(feature_onsets)
					if abs(spec_onset - feature_onsets[k]) > 0.01:
						print("Mismatch between spec_dirs and feature_dirs!")
						print("hdf5 file:", hdf5)
						print("\tindex:", j)
						print("audio filename:", audio_fn)
						print("feature filename:", feature_fn)
						print("Didn't find spec_onset", spec_onset)
						print("in feature onsets of min:", \
								np.min(feature_onsets), "max:", \
								np.max(feature_onsets))
						print("field:", field)
						print("kind:", kind)
						quit()
					# And add it to the feature array.
					feature_arr[j] = features[k]
				# Write the fields to self.projection_dirs.
				write_fn = os.path.join(proj_dir, os.path.split(hdf5)[-1])
				with h5py.File(write_fn, 'a') as f:
					f.create_dataset(field, data=feature_arr)
				to_return.append(feature_arr)
		self.fields[field] = 1
		return np.concatenate(to_return)


	def _write_projection(self, key, data):
		"""Write the given projection to self.projection_dirs."""
		sylls_per_file = self.sylls_per_file
		# For each directory...
		k = 0
		for i in range(len(self.projection_dirs)):
			spec_dir, proj_dir = self.spec_dirs[i], self.projection_dirs[i]
			hdf5s = get_hdf5s_from_dir(spec_dir)
			for j in range(len(hdf5s)):
				filename = os.path.join(proj_dir, os.path.split(hdf5s[j])[-1])
				to_write = data[k:k+sylls_per_file]
				with h5py.File(filename, 'a') as f:
					f.create_dataset(key, data=to_write)
				k += sylls_per_file


	def _check_for_fields(self):
		"""Check to see which fields are saved."""
		fields = {}
		# If self.spec_dirs is registered, assume everything is there.
		if self.spec_dirs is not None:
			for field in SPEC_FIELDS:
				fields[field] = 1
		# Same for self.audio_dirs.
		if self.audio_dirs is not None:
			fields['audio'] = 1
		# Same for self.segment_dirs.
		if self.segment_dirs is not None:
			fields['segments'] = 1
			fields['segment_audio'] = 1
		# If self.projection_dirs is registered, see what we have.
		# If it's in one file, assume it's in all of them.
		if self.projection_dirs is not None:
			if os.path.exists(self.projection_dirs[0]):
				hdf5s = get_hdf5s_from_dir(self.projection_dirs[0])
				if len(hdf5s) > 0:
					hdf5 = hdf5s[0]
					if os.path.exists(hdf5):
						with h5py.File(hdf5, 'r') as f:
							for key in f.keys():
								if key in ALL_FIELDS:
									fields[key] = 1
									self.sylls_per_file = len(f[key])
		return fields


	def _check_for_dirs(self, dir_names, field):
		"""Check that the given directories exist."""
		for dir_name in dir_names:
			if dir_name == 'audio_dirs':
				temp = self.audio_dirs
			elif dir_name == 'segment_dirs':
				temp = self.segment_dirs
			elif dir_name == 'spec_dirs':
				temp = self.spec_dirs
			elif dir_name == 'feature_dirs':
				temp = self.feature_dirs
			elif dir_name == 'projection_dirs':
				temp = self.projection_dirs
			elif dir_name == 'model_filename':
				temp = self.model_filename
			else:
				raise NotImplementedError
			assert temp is not None, dir_name + " must be specified before " + \
				field + " is made!"



def _read_columns(filename, columns=(0,1), delimiter=',', skiprows=1, \
	unpack=True):
	"""
	A wrapper around numpy.loadtxt to handle empty files.

	TO DO: Add categorical variables.
	"""
	data = np.loadtxt(filename, delimiter=delimiter, usecols=columns, \
		skiprows=skiprows).reshape(-1,len(columns))
	if unpack:
		return tuple(data[:,i] for i in range(data.shape[1]))
	return data


def _is_seg_file(filename):
	"""Is this a segmenting file?"""
	return len(filename) > 4 and filename[-4:] == '.txt'


def _is_wav_file(filename):
	"""Is this a wav file?"""
	return len(filename) > 4 and filename[-4:] == '.wav'



if __name__ == '__main__':
	pass


###
