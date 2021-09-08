Preprocessing
=============

The preprocessing step determines how spectrograms will look when they're fed
into the VAE. First, we want to define and tune our preprocessing parameters:

.. code:: Python3

	from ava.preprocessing.utils import get_spec # makes spectrograms
	from ava.models.vae import X_SHAPE # spectrogram dimensions

	preprocess_params = {
	    'get_spec': get_spec, # spectrogram maker
	    'max_dur': 0.2, # maximum syllable duration
	    'min_freq': 30e3, # minimum frequency
	    'max_freq': 110e3, # maximum frequency
	    'num_freq_bins': X_SHAPE[0], # hard-coded
	    'num_time_bins': X_SHAPE[1], # hard-coded
	    'nperseg': 1024, # FFT
	    'noverlap': 512, # FFT
	    'spec_min_val': 2.0, # minimum log-spectrogram value
	    'spec_max_val': 6.0, # maximum log-spectrogram value
	    'fs': 250000, # audio samplerate
	    'mel': False, # frequency spacing, mel or linear
	    'time_stretch': True, # stretch short syllables?
	    'within_syll_normalize': False, # normalize spectrogram values on a
	                                    # spectrogram-by-spectrogram basis
	    'max_num_syllables': None, # maximum number of syllables per directory
	    'sylls_per_file': 20, # syllable per file
	    'real_preprocess_params': ('min_freq', 'max_freq', 'spec_min_val', \
	            'spec_max_val', 'max_dur'), # tunable parameters
	    'int_preprocess_params': ('nperseg','noverlap'), # tunable parameters
	    'binary_preprocess_params': ('time_stretch', 'mel', \
	            'within_syll_normalize'), # tunable parameters
	}

	from ava.preprocessing.preprocess import tune_syll_preprocessing_params
	audio_dirs = [...] # directories containing audio
	seg_dirs = [...] # directories containing onset/offset decisions
	preprocess_params = tune_syll_preprocessing_params(audio_dirs, seg_dirs, \
			preprocess_params)



This will start an interactive tuning process, where parameters can be adjusted
and the resulting spectrograms will be saved as an image, by default
:code:`temp.pdf`. We can keep tuning these parameter values until we're happy
with the spectrograms they produce.

If we're doing a shotgun VAE analysis, then we use these parameters to make the
Dataloaders (next section). Otherwise, if we're doing a syllable-level analysis,
we make a spectrogram for each detected syllable. These spectrograms should be
saved in their own directories with a 1-to-1 correspondence between audio
directories and segment directories:


.. code:: Python3

	# Define directories.
	audio_dirs = ['path/to/animal1/audio/', 'path/to/animal2/audio/']
	seg_dirs = ['path/to/animal1/segments/', 'path/to/animal2/segments/']
	spec_dirs = ['path/to/animal1/specs/', 'path/to/animal2/specs/']

	from ava.preprocessing.preprocess import process_sylls
	from joblib import Parallel, delayed
	from itertools import repeat

	gen = zip(audio_dirs, seg_dirs, spec_dirs, repeat(preprocess_params))
	Parallel(n_jobs=4)(delayed(process_sylls)(*args) for args in gen)



This will write a bunch of :code:`.hdf5` files in :code:`spec_dirs`, each with
the following attributes:

* :code:`'onsets'`: syllable onsets within file, in seconds
* :code:`'offsets'`: syllable offsets within file, in seconds
* :code:`'specs'`: syllable spectrograms
* :code:`'audio_filenames'`: names of audio files the syllables were extracted from
