"""
Amplitude-based syllable segmentation.

"""
__date__ = "December 2018 - October 2019"


import numpy as np
from scipy.io import wavfile
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d

from ava.segmenting.utils import get_spec, softmax


EPSILON = 1e-9



def get_onsets_offsets(audio, p, return_traces=False):
	"""
	Segment the spectrogram using thresholds on its amplitude.

	A syllable is detected if the amplitude trace exceeds `p['th_3']`. An offset
	is then detected if there is a subsequent local minimum in the amplitude
	trace with amplitude less than `p['th_2']`, or when the amplitude drops
	below `p['th_1']`, whichever comes first. Syllable onset is determined
	analogously.

	Note
	----
	`p['th_1'] <= p['th_2'] <= p['th_3']`

	Parameters
	----------
	audio : numpy.ndarray
		Raw audio samples.
	p : dict
		Parameters.
	return_traces : bool, optional
		Whether to return traces. Defaults to `False`.

	Returns
	-------
	onsets : numpy array
		Onset times, in seconds
	offsets : numpy array
		Offset times, in seconds
	traces : list of a single numpy array
		The amplitude trace used in segmenting decisions. Returned if
		`return_traces` is `True`.
	"""
	if len(audio) < p['nperseg']:
		if return_traces:
			return [], [], None
		return [], []
	spec, dt, _ = get_spec(audio, p)
	min_syll_len = int(np.floor(p['min_dur'] / dt))
	max_syll_len = int(np.ceil(p['max_dur'] / dt))
	th_1, th_2, th_3 = p['th_1'], p['th_2'], p['th_3'] # tresholds
	onsets, offsets = [], []
	too_short, too_long = 0, 0

	# Calculate amplitude and smooth.
	if p['softmax']:
		amps = softmax(spec, t=p['temperature'])
	else:
		amps = np.sum(spec, axis=0)
	amps = gaussian_filter(amps, p['smoothing_timescale']/dt)

	# Find local maxima greater than th_3.
	local_maxima = []
	for i in range(1,len(amps)-1,1):
		if amps[i] > th_3 and amps[i] == np.max(amps[i-1:i+2]):
			local_maxima.append(i)

	# Then search to the left and right for onsets and offsets.
	for local_max in local_maxima:
		if len(offsets) > 0 and local_max < offsets[-1]:
			continue
		i = local_max - 1
		while i > 0:
			if amps[i] < th_1:
				onsets.append(i)
				break
			elif amps[i] < th_2 and amps[i] == np.min(amps[i-1:i+2]):
				onsets.append(i)
				break
			i -= 1
		if len(onsets) != len(offsets) + 1:
			onsets = onsets[:len(offsets)]
			continue
		i = local_max + 1
		while i < len(amps):
			if amps[i] < th_1:
				offsets.append(i)
				break
			elif amps[i] < th_2 and amps[i] == np.min(amps[i-1:i+2]):
				offsets.append(i)
				break
			i += 1
		if len(onsets) != len(offsets):
			onsets = onsets[:len(offsets)]
			continue

	# Throw away syllables that are too long or too short.
	new_onsets = []
	new_offsets = []
	for i in range(len(offsets)):
		t1, t2 = onsets[i], offsets[i]
		if t2 - t1 + 1 <= max_syll_len and t2 - t1 + 1 >= min_syll_len:
			new_onsets.append(t1 * dt)
			new_offsets.append(t2 * dt)
		elif t2 - t1 + 1 > max_syll_len:
			too_long += 1
		else:
			too_short += 1

	# Return decisions.
	if return_traces:
		return new_onsets, new_offsets, [amps]
	return new_onsets, new_offsets



if __name__ == '__main__':
	pass


###
