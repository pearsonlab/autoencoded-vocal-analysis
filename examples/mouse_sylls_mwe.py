"""
Minimal working example for mouse syllables.

0) Define directories and parameters.
1) Tune segmenting parameters.
2) Segment.
	2.5) Clean segmenting decisions. (optional)
3) Tune preprocessing parameters.
4) Preprocess.
5) Train a generative model on these syllables.
6) Plot.
7) The world is your oyster.

"""

from itertools import repeat
from joblib import Parallel, delayed
import numpy as np
import os

from ava.data.data_container import DataContainer
from ava.models.vae import X_SHAPE, VAE
from ava.models.vae_dataset import get_syllable_partition, \
	get_syllable_data_loaders
from ava.preprocessing.preprocess import process_sylls, \
	tune_syll_preprocessing_params
from ava.preprocessing.utils import get_spec
from ava.segmenting.refine_segments import refine_segments_pre_vae
from ava.segmenting.segment import tune_segmenting_params, segment
from ava.segmenting.amplitude_segmentation import get_onsets_offsets



#########################################
# 0) Define directories and parameters. #
#########################################
params = {
	'segment': {
		'min_freq': 30e3, # minimum frequency
		'max_freq': 110e3, # maximum frequency
		'nperseg': 1024, # FFT
		'noverlap': 512, # FFT
		'spec_min_val': 2.0, # minimum log-spectrogram value
		'spec_max_val': 6.0, # maximum log-spectrogram value
		'fs': 250000, # audio samplerate
		'th_1':0.1, # segmenting threshold 1
		'th_2':0.2, # segmenting threshold 2
		'th_3':0.3, # segmenting threshold 2
		'max_dur': 0.2, # maximum syllable duration
		'min_dur':0.03, # minimum syllable duration
		'smoothing_timescale': 0.007, # timescale for smoothing amplitude trace
		'softmax': True, # puts amplitude values in [0,1]
		'temperature': 0.5, # temperature parameter for softmax
		'algorithm': get_onsets_offsets, # segmentation algorithm
	},
	'preprocess': {
		'get_spec': get_spec,
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
	},
}

root = '/path/to/directory/'
audio_dirs = [os.path.join(root, 'audio')]
seg_dirs = [os.path.join(root, 'segs')]
proj_dirs = [os.path.join(root, 'projections')]
spec_dirs = [os.path.join(root, 'specs')]
model_filename = os.path.join(root, 'checkpoint_150.tar')
plots_dir = root
dc = DataContainer(projection_dirs=proj_dirs, spec_dirs=spec_dirs, \
		plots_dir=plots_dir, model_filename=model_filename)


##################################
# 1) Tune segmenting parameters. #
##################################
params['segment'] = tune_segmenting_params(audio_dirs, params['segment'])


###############
# 2) Segment. #
###############
n_jobs = min(len(audio_dirs), os.cpu_count()-1)
gen = zip(audio_dirs, seg_dirs, repeat(params['segment']))
Parallel(n_jobs=n_jobs)(delayed(segment)(*args) for args in gen)


# ###############################################
# # 2.5) (optional) Clean segmenting decisions. #
# ###############################################
# refine_segments_pre_vae(seg_dirs, audio_dirs, new_seg_dirs, params['segment'])


#####################################
# 3) Tune preprocessing parameters. #
#####################################
preprocess_params = tune_syll_preprocessing_params(audio_dirs, seg_dirs, \
		params['preprocess'])
params['preprocess'] = preprocess_params


##################
# 4) Preprocess. #
##################
n_jobs = os.cpu_count()-1
gen = zip(audio_dirs, seg_dirs, spec_dirs, repeat(params['preprocess']))
Parallel(n_jobs=n_jobs)(delayed(process_sylls)(*args) for args in gen)


###################################################
# 5) Train a generative model on these syllables. #
###################################################
model = VAE(save_dir=root)
# model.load_state(root+'checkpoint_150.tar')
partition = get_syllable_partition(spec_dirs, split=1, max_num_files=2500)
num_workers = os.cpu_count()-1
loaders = get_syllable_data_loaders(partition, num_workers=num_workers)
loaders['test'] = loaders['train']
model.train_loop(loaders, epochs=151, test_freq=None)


############
# 6) Plot. #
############
from ava.plotting.tooltip_plot import tooltip_plot_DC
from ava.plotting.latent_projection import latent_projection_plot_DC
latent_projection_plot_DC(dc)
tooltip_plot_DC(dc, num_imgs=2000)


################################
# 7) The world is your oyster. #
################################
pass



if __name__ == '__main__':
	pass


###
