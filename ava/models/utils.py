"""
Useful functions related to the `ava.models` subpackage.

"""
__date__ = "July - November 2020"


from affinewarp import PiecewiseWarping
from affinewarp.crossval import paramsearch
import h5py
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning
from scipy.signal import stft
import torch
import warnings


DEFAULT_SEARCH_PARAMS = {
	'samples_per_knot': 10,
	'n_valid_samples': 5,
	'n_train_folds': 3,
	'n_valid_folds': 1,
	'n_test_folds': 1,
	'knot_range': (-1, 2),
	'smoothness_range': (1e-1, 1e2),
	'warpreg_range': (1e-1, 1e2),
	'iter_range': (50, 51),
	'warp_iter_range': (50, 101),
	'outfile': None,
}
"""Default parameters sent to `affinewarp.crossval.paramsearch`"""

PARAM_NAMES = [
	'n_knots',
	'warp_reg_scale',
	'smoothness_reg_scale',
]

EPSILON = 1e-9



def cross_validation_warp_parameter_search(audio_dirs, spec_params, \
	search_params={}, warp_type='spectrogram', verbose=True, make_plot=True,
	img_fn='temp.pdf'):
	"""
	Perform a parameter search over timewarping parameters.

	This is a wrapper around `affinewarp.crossval.paramsearch`.

	Note
	----
	* All `.wav` files should be the same duration!

	Parameters
	----------
	audio_dirs : list of str
		Audio directories.
	spec_params : dict
		Preprocessing parameters. Must contain keys: ``'window_length'``,
		``'nperseg'``, ``'noverlap'``, ``'min_freq'``, ``'max_freq'``,
		``'spec_min_val'``, and ``'spec_max_val'``.
	search_params : dict, optional
		Parameters sent to `affinewarp.crossval.paramsearch`. Defaults to
		`DEFAULT_SEARCH_PARAMS`.
	warp_type : {``'amplitude'``, ``'spectrogram'``}, optional
		Whether to time-warp using ampltidue traces or full spectrograms.
		Defaults to ``'spectrogram'``.
	verbose : bool, optional
		Defaults to `True`.
	make_plot : bool, optional
		Defaults to ``True``.
	img_fn : str, optional
		Defaults to ``temp.pdf``.

	Returns
	-------
	res : dict
		Complete `affinewarp.crossval.paramsearch` result. See
		github.com/ahwillia/affinewarp/blob/master/affinewarp/crossval.py
	"""
	assert type(spec_params) == type({})
	assert warp_type in ['amplitude', 'spectrogram']
	search_params = {**DEFAULT_SEARCH_PARAMS, **search_params}
	# Collect audio filenames.
	if verbose:
		print("Collecting spectrograms...")
	audio_fns = []
	for audio_dir in audio_dirs:
		audio_fns += _get_wavs_from_dir(audio_dir)
	# Make spectrograms.
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", category=WavFileWarning)
		all_audio = [wavfile.read(audio_fn)[1] for audio_fn in audio_fns]
		fs = wavfile.read(audio_fns[0])[0]
	specs, amps, _  = _get_specs_and_amplitude_traces(all_audio, fs, \
			spec_params)
	if verbose:
		print("\tDone.")
		print("Running parameter search...")
	# Run the parameter search and return.
	if warp_type == 'amplitude':
		to_warp = amps
	else:
		to_warp = specs
	res = paramsearch(to_warp, **search_params)
	if verbose:
		print("\tDone.")
	# Plot.
	# Stolen from: github.com/ahwillia/affinewarp/blob/master/examples/piecewise_warping.ipynb
	if make_plot:
		train_rsq = np.median(res['train_rsq'], axis=1)
		valid_rsq = np.median(res['valid_rsq'], axis=1)
		test_rsq = res['test_rsq']
		knots = res['knots']
		plt.scatter(knots-0.1, train_rsq, c='k', label='train', alpha=0.5)
		plt.scatter(knots, valid_rsq, c='b', label='validation', alpha=0.7)
		plt.scatter(knots+0.1, test_rsq, c='r', label='test', alpha=0.7)
		plt.ylabel("$R^2$")
		plt.xlabel("n_knots")
		plt.legend(loc='best')
		ax = plt.gca()
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		plt.savefig(img_fn)
		plt.close('all')
	return res


def anchor_point_warp_parameter_search(audio_dirs, anchor_dir, spec_params, \
	search_params, num_iter=20, gridpoints=6, warp_type='amplitude', \
	aw_iterations=25, aw_warp_iterations=100, verbose=True, make_plot=True, \
	img_fn='temp.pdf'):
	"""
	Evaluate time-warping parameters on aligning hand-labeled anchor points.

	Randomly samples different values of `n_knots`, `warp_reg_scale`, and
	`smoothness_reg_scale`. `n_knots` is sampled uniformly in
	[`search_params['knot_range'][0]`, `search_params['knot_range'][1]`).
	`warp_reg_scale` and `smoothness_reg_scale` are sampled log-uniformly on
	grids with `gridpoints` points. Those ranges are given by
	`search_params['smoothness_range']` and `search_params['warpreg_range']`.

	Note
	----
	* All `.wav` files should be the same duration!

	Parameters
	----------
	audio_dirs : list of str
		Audio directories.
	anchor_dir : str
		Directory containing audio files and corresponding anchor point
		annotation text files. The text files should have the same names as the
		audio files they correspond to and should be formatted like syllable
		segments. Each text file should contain the same number of segments,
		where the i^th anchor point is given by the i^th onset time. Offsets are
		ignored.
	spec_params : dict
		Preprocessing parameters. Must contain keys: ``'window_length'``,
		``'nperseg'``, ``'noverlap'``, ``'min_freq'``, ``'max_freq'``,
		``'spec_min_val'``, and ``'spec_max_val'``.
	search_params : dict, optional
		Must contain keys `'knot_range'`, `'smoothness_range'`, and
		`'warpreg_range'`.
	num_iter : int, optional
		Number of warping runs. Defualts to `50`.
	gridpoints : int, optional
		How finely to sample `warp_reg_scale` and `smoothness_reg_scale`
	warp_type : {``'amplitude'``, ``'spectrogram'``}, optional
		Whether to time-warp using ampltidue traces or full spectrograms.
		Defaults to ``'spectrogram'``.
	aw_iterations : int, optional
		Affinewarp `iterations` parameter. Defaults to `25`.
	aw_warp_iterations : int, optional
		Affinewarp `warp_iterations` parameter. Defaults to `100`.
	verbose : bool, optional
		Defaults to `True`.
	make_plot : bool, optional
		Defaults to ``True``.
	img_fn : str, optional
		Defaults to ``temp.pdf``.

	Returns
	-------
	param_history : numpy.ndarray
		Sampled parameter values. The three columns denote `n_knots`,
		`warp_reg_scale`, and `smoothness_reg_scale`. Elements index the
		corresponding entries of `support`.
		Shape: `(num_iter,3)`
	loss_history : numpy.ndarray
		Mean absolute errors. Shape: `(num_iter,)`
	support : list of numpy.ndarray
		The support for `n_knots`, `warp_reg_scale`, and `smoothness_reg_scale`,
		respectively.
	"""
	assert type(spec_params) == type({})
	assert warp_type in ['amplitude', 'spectrogram']
	# Get anchor times.
	anchor_fns = _get_txts_from_dir(anchor_dir)
	anchor_times = [np.loadtxt(fn).reshape(-1,2)[:,0] for fn in anchor_fns]
	anchor_times = np.array(anchor_times)
	mean_times = np.mean(anchor_times, axis=0, keepdims=True)
	null_warp_mae = 1e3 * np.mean(np.abs(mean_times - anchor_times))
	if verbose:
		print("Null warp MAE:", '{0:.3f}'.format(null_warp_mae), 'ms')
	for i in range(1,len(anchor_times)):
		assert len(anchor_times[0]) == len(anchor_times[i]), 'Unequal numbers'+\
				' of anchor times!'
	# Collect audio filenames.
	if verbose:
		print("Collecting spectrograms...")
	audio_fns = []
	for audio_dir in audio_dirs + [anchor_dir]: # annotated audio is at the end
		audio_fns += _get_wavs_from_dir(audio_dir)
	# Make spectrograms.
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", category=WavFileWarning)
		all_audio = [wavfile.read(audio_fn)[1] for audio_fn in audio_fns]
		fs = wavfile.read(audio_fns[0])[0]
	specs, amps, template_dur  = _get_specs_and_amplitude_traces(all_audio, fs,\
			spec_params)
	if warp_type == 'amplitude':
		to_warp = amps
	else:
		to_warp = specs
	if verbose:
		print("\tDone.")
		print("Evaluating parameters...")
	# Set up search.
	search_params = {**DEFAULT_SEARCH_PARAMS, **search_params}
	knot_range = search_params['knot_range']
	support = [
		np.arange(*knot_range),
		np.geomspace(*search_params['warpreg_range'], num=gridpoints),
		np.geomspace(*search_params['smoothness_range'], num=gridpoints),
	]
	param_ranges = [
		np.arange(knot_range[1]-knot_range[0]),
		np.arange(gridpoints),
		np.arange(gridpoints),
	]
	param_history =  np.zeros((num_iter,len(PARAM_NAMES)), dtype='int')
	loss_history = np.zeros(num_iter)
	eval_func = _get_eval_func(support, anchor_times, to_warp, aw_iterations, \
			aw_warp_iterations, template_dur)
	# Repeatedly sample parameters and evaluate.
	for i in range(num_iter): # num_iter
		for j in range(len(PARAM_NAMES)):
			param_history[i,j] = np.random.choice(param_ranges[j])
		loss = eval_func(param_history[i])
		if verbose:
			print('\t'+str(param_history[i]), '{0:.3f}'.format(loss), 'ms')
		loss_history[i] = loss
	# Plot objective vs. parameter marginals.
	if make_plot:
		_, axarr = plt.subplots(nrows=3)
		for i, (ax, key) in enumerate(zip(axarr, PARAM_NAMES)):
			x_vals = param_history[:,i] - 0.1 + 0.2 * np.random.rand(num_iter)
			ax.axhline(y=null_warp_mae, c='k', ls='--', alpha=0.5, lw=0.8)
			ax.scatter(x_vals, loss_history, c='k', alpha=0.5)
			ax.set_xlabel(key)
			ax.set_ylabel('MAE (ms)')
			ax.spines['top'].set_visible(False)
			ax.spines['right'].set_visible(False)
			plt.sca(ax)
			plt.xticks(param_ranges[i], ['{0:.5f}'.format(j) for j in support[i]])
		plt.tight_layout()
		plt.savefig(img_fn)
		plt.close('all')
	return param_history, loss_history, support


def _get_eval_func(support, anchor_times, to_warp, aw_iterations, \
	aw_warp_iterations, template_dur):
	"""Return an objective function."""

	def eval_func(params):
		"""Run affinewarp and return anchor time mean absolute errors."""
		# Set up parameters.
		warp_params = {}
		for i, key in enumerate(PARAM_NAMES):
			warp_params[key] = support[i][params[i]]
		# Fit.
		model = PiecewiseWarping(**warp_params)
		model.fit(to_warp, iterations=aw_iterations, \
				warp_iterations=aw_warp_iterations, verbose=False)
		# Evaluate.
		warped_anchor_times = np.zeros_like(anchor_times)
		for i in range(len(anchor_times)):
			x_knots, y_knots = model.x_knots[i], model.y_knots[i]
			# Convert empirical times to template times.
			interp = interp1d(x_knots, y_knots, bounds_error=False, \
					fill_value='extrapolate', assume_sorted=True)
			warped_anchor_times[i] = interp(anchor_times[i]/template_dur)
		warped_anchor_times *= template_dur
		mean_times = np.mean(warped_anchor_times, axis=0, keepdims=True)
		mae = np.mean(np.abs(mean_times - warped_anchor_times))
		# Correct for changes in timescale, convert to milliseconds.
		mae *= 1e3 * np.std(anchor_times) / np.std(warped_anchor_times)
		return mae

	return eval_func


def _get_sylls_per_file(partition):
	"""
	Open an hdf5 file and see how many syllables it has.

	Assumes all hdf5 file referenced by `partition` have the same number of
	syllables.

	Parameters
	----------
	partition : dict
		Contains two keys, ``'test'`` and ``'train'``, that map to lists of hdf5
		files. Defines the random test/train split.

	Returns
	-------
	sylls_per_file : int
		How many syllables are in each file.
	"""
	key = 'train' if len(partition['train']) > 0 else 'test'
	assert len(partition[key]) > 0
	filename = partition[key][0] # Just grab the first file.
	with h5py.File(filename, 'r') as f:
		sylls_per_file = len(f['specs'])
	return sylls_per_file


def _get_spec(audio, fs, p):
	"""
	Make a basic spectrogram.

	Parameters
	----------
	audio : numpy.ndarray
		Audio
	fs : int
		Samplerate
	p : dict
		Contains keys `'nperseg'`, `'noverlap'`, `'min_freq'`, `'max_freq'`,
		`'spec_min_val'`, and `'spec_max_val'`.

	Returns
	-------
	spec : numpy.ndarray
		Spectrogram, freq_bins x time_bins
	dt : float
		Spectrogram time step
	"""
	f, t, spec = stft(audio, fs=fs, nperseg=p['nperseg'], \
			noverlap=p['noverlap'])
	i1 = np.searchsorted(f, p['min_freq'])
	i2 = np.searchsorted(f, p['max_freq'])
	spec = spec[i1:i2]
	f = f[i1:i2]
	spec = np.log(np.abs(spec) + EPSILON)
	spec -= p['spec_min_val']
	spec /= p['spec_max_val'] - p['spec_min_val'] + EPSILON
	spec = np.clip(spec, 0.0, 1.0)
	return spec, t[1]-t[0]


def _get_specs_and_amplitude_traces(all_audio, fs, spec_params):
	"""
	Return spectrograms and amplitude traces given a list of audio.

	Parameters
	----------
	all_audio : list of numpy.ndarray
		List of audio.
	fs : int
		Audio samplerate
	spec_params : dict
		Contains keys `'nperseg'`, `'noverlap'`, `'min_freq'`, `'max_freq'`,
		`'spec_min_val'`, and `'spec_max_val'`.

	Returns
	-------
	specs : numpy.ndarray
		Spectrograms
	amps : numpy.ndarray
		Amplitude traces
	template_dur : float
		Template duration
	"""
	# Make spectrograms.
	specs = []
	for i in range(len(all_audio)):
		spec, dt = _get_spec(all_audio[i], fs, spec_params)
		specs.append(spec.T)
	# Check to make sure everything's the same shape.
	assert len(specs) > 0
	min_time_bins = min(spec.shape[0] for spec in specs)
	specs = [spec[:min_time_bins] for spec in specs]
	min_freq_bins = min(spec.shape[1] for spec in specs)
	specs = [spec[:,:min_freq_bins] for spec in specs]
	num_time_bins = specs[0].shape[0]
	assert num_time_bins == min_time_bins
	template_dur = num_time_bins * dt
	# Compute amplitude traces.
	amps = []
	for i in range(len(all_audio)):
		amp_trace = np.sum(specs[i], axis=-1, keepdims=True)
		amp_trace -= np.min(amp_trace)
		amp_trace /= np.max(amp_trace) + EPSILON
		amps.append(amp_trace)
	# Stack and return.
	amps = np.stack(amps)
	specs = np.stack(specs)
	return specs, amps, template_dur


def get_hdf5s_from_dir(dir):
	"""
	Return a sorted list of all hdf5s in a directory.

	Note
	----
	``ava.data.data_container`` relies on this.
	"""
	return [os.path.join(dir, f) for f in sorted(os.listdir(dir)) if \
			_is_hdf5_file(f)]


def _get_wavs_from_dir(dir):
	"""Return a sorted list of wave files from a directory."""
	return [os.path.join(dir, f) for f in sorted(os.listdir(dir)) if \
			_is_wav_file(f)]


def _get_txts_from_dir(dir):
	"""Return a sorted list of text files from a directory."""
	return [os.path.join(dir, f) for f in sorted(os.listdir(dir)) if \
			_is_txt_file(f)]

def numpy_to_tensor(x):
	"""Transform a numpy array into a torch.FloatTensor."""
	return torch.from_numpy(x).type(torch.FloatTensor)


def _is_hdf5_file(filename):
	"""Is the given filename an hdf5 file?"""
	return len(filename) > 5 and filename[-5:] == '.hdf5'


def _is_wav_file(filename):
	"""Is the given filename a wave file?"""
	return len(filename) > 4 and filename[-4:] == '.wav'


def _is_txt_file(filename):
	"""Is the given filename a text file?"""
	return len(filename) > 4 and filename[-4:] == '.txt'



if __name__ == '__main__':
	pass


###
