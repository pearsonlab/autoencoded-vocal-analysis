"""
Useful functions for preprocessing.

"""
__date__ = "August 2019 - October 2020"


import numpy as np
import warnings
from scipy.signal import stft
from scipy.interpolate import interp2d


EPSILON = 1e-12



def get_spec(t1, t2, audio, p, fs=32000, target_freqs=None, target_times=None, \
	fill_value=-1/EPSILON, max_dur=None, remove_dc_offset=True):
	"""
	Norm, scale, threshold, stretch, and resize a Short Time Fourier Transform.

	Notes
	-----
	* ``fill_value`` necessary?
	* Look at all references and see what can be simplified.
	* Why is a flag returned?

	Parameters
	----------
	t1 : float
		Onset time.
	t2 : float
		Offset time.
	audio : numpy.ndarray
		Raw audio.
	p : dict
		Parameters. Must include keys: ...
	fs : float
		Samplerate.
	target_freqs : numpy.ndarray or ``None``, optional
		Interpolated frequencies.
	target_times : numpy.ndarray or ``None``, optional
		Intepolated times.
	fill_value : float, optional
		Defaults to ``-1/EPSILON``.
	max_dur : float, optional
		Maximum duration. Defaults to ``None``.
	remove_dc_offset : bool, optional
		Whether to remove any DC offset from the audio. Defaults to ``True``.

	Returns
	-------
	spec : numpy.ndarray
		Spectrogram.
	flag : bool
		``True``
	"""
	if max_dur is None:
		max_dur = p['max_dur']
	if t2 - t1 > max_dur + 1e-4:
		message = "Found segment longer than max_dur: " + str(t2-t1) + \
				"s, max_dur = " + str(max_dur) + "s"
		warnings.warn(message)
	s1, s2 = int(round(t1*fs)), int(round(t2*fs))
	assert s1 < s2, "s1: " + str(s1) + " s2: " + str(s2) + " t1: " + str(t1) + \
			" t2: " + str(t2)
	# Get a spectrogram and define the interpolation object.
	temp = min(len(audio),s2) - max(0,s1)
	if temp < p['nperseg'] or s2 <= 0 or s1 >= len(audio):
		return np.zeros((p['num_freq_bins'], p['num_time_bins'])), True
	else:
		temp_audio = audio[max(0,s1):min(len(audio),s2)]
		if remove_dc_offset:
			temp_audio = temp_audio - np.mean(temp_audio)
		f, t, spec = stft(temp_audio, fs=fs, nperseg=p['nperseg'], \
				noverlap=p['noverlap'])
	t += max(0,t1)
	spec = np.log(np.abs(spec) + EPSILON)
	interp = interp2d(t, f, spec, copy=False, bounds_error=False, \
		fill_value=fill_value)
	# Define target frequencies.
	if target_freqs is None:
		if p['mel']:
			target_freqs = np.linspace(_mel(p['min_freq']), \
					_mel(p['max_freq']), p['num_freq_bins'])
			target_freqs = _inv_mel(target_freqs)
		else:
			target_freqs = np.linspace(p['min_freq'], p['max_freq'], \
					p['num_freq_bins'])
	# Define target times.
	if target_times is None:
		duration = t2 - t1
		if p['time_stretch']:
			duration = np.sqrt(duration * max_dur) # stretched duration
		shoulder = 0.5 * (max_dur - duration)
		target_times = np.linspace(t1-shoulder, t2+shoulder, p['num_time_bins'])
	# Then interpolate.
	interp_spec = interp(target_times, target_freqs, assume_sorted=True)
	spec = interp_spec
	# Normalize.
	spec -= p['spec_min_val']
	spec /= (p['spec_max_val'] - p['spec_min_val'])
	spec = np.clip(spec, 0.0, 1.0)
	# Within-syllable normalize.
	if p['within_syll_normalize']:
		spec -= np.quantile(spec, p['normalize_quantile'])
		spec[spec<0.0] = 0.0
		spec /= np.max(spec) + EPSILON
	return spec, True


def _mel(a):
	"""https://en.wikipedia.org/wiki/Mel-frequency_cepstrum"""
	return 1127 * np.log(1 + a / 700)


def _inv_mel(a):
	"""https://en.wikipedia.org/wiki/Mel-frequency_cepstrum"""
	return 700 * (np.exp(a / 1127) - 1)



if __name__ == '__main__':
	pass


###
