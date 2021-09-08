"""
Simple shift-only and linear time-warping functions.

This is an alternative to `affinewarp` time warping.

Warning
-------
* `ava.preprocessing.warping` is experimental and may change in a future version
  of AVA!
"""
__date__ = "September 2020 - November 2020"


import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import warnings


WARNING_MSG = "ava.preprocessing.warping is experimental and may change in " + \
		"a future version of AVA!"



def apply_warp(specs, warp_params):
	"""
	Take real spectrograms, apply linear warps, making warped spectrograms.

	Parameters
	----------
	specs : numpy.ndarray
		Spectrograms with shape `[n_specs, num_freq_bins, num_time_bins]``
	warp_params : dict
		Returned by `align_specs`. Maps keys `'shifts'` and `'slopes'` to Numpy
		arrays with shape `[n_specs]`.

	Returns
	-------
	warped_specs : numpy.ndarray
		Time-warped spectrograms. Same shape as `spec`.
	"""
	shifts, slopes = warp_params['shifts'], warp_params['slopes']
	warped_specs = []
	for i in range(len(specs)):
		spec = specs[i]
		f = interp1d(np.arange(spec.shape[1]), spec, assume_sorted=True, \
				bounds_error=False, fill_value=(spec[:,0],spec[:,-1]))
		warped_spec = f(shifts[i] + slopes[i]*np.arange(spec.shape[1]))
		warped_specs.append(warped_spec)
	return np.array(warped_specs)


def align_specs(specs, shift_λs, slope_λs, verbose=True):
	"""
	Align the spectrograms, return warping parameters and warped specs.

	Minimizes the following regularized L2 loss:


	.. math::

		\| \\textrm{warped_spec} - \\textrm{target_spec} \|_2^2 +
		\\textrm{shift_λ} \cdot \\textrm{shift}^2 +
		\\textrm{slope_λ} \cdot (\log \\textrm{slope})^2


	where `target_spec` is the average warped spectrogram, updated after every
	optimization iteration. It's a good idea to start with large values of
	``shift_λ`` and ``slope_λ`` that gradually decrease to zero if you want to
	end up with a maximum likelihood estimate. In particular, I've found it's
	helpful to do a shift-only warp the first few iterations by setting
	``slope_λ`` to ``np.inf``.

	Notes
	-----
	* If the optimization fails, the failure message is printed and
	  ``(None, None)`` is returned.
	* This works better when the spectrograms are summed over the frequency axis
	  like this: ``np.sum(specs, axis=1, keepdims=True)``
	* Because the objective changes every iteration, it's not necessarily bad if
	  the loss isn't monotonically decreasing.

	Raises
	------
	`UserWarning`
		Tells you this is experimental and may change in future versions of AVA.

	Parameters
	----------
	specs : numpy.ndarray
		Spectrograms, shape: [n_specs, freq_bins, time_bins]
	shift_λs : sequence of float
	slope_λs : sequence of float
	verbose : bool, optional

	Returns
	-------
	warped_specs : numpy.ndarray
		Warped spectrograms. Same shape as input spectrograms.
	warp_params : dictionary
		Maps `'shifts'` and `'slopes'` to their inferred values. Values are in
		units of time bins.
	"""
	warnings.warn(WARNING_MSG)
	# Set up some things.
	specs = np.copy(specs)
	warped_specs = np.copy(specs)
	x0 = np.zeros((len(specs),2))
	interps = []
	for i in range(len(specs)):
		spec = specs[i]
		f = interp1d(np.arange(spec.shape[1]), spec, assume_sorted=True, \
				bounds_error=False, fill_value=(spec[:,0],spec[:,-1]))
		interps.append(f)
	total_iterations = min(len(shift_λs), len(slope_λs))
	# Warp to average spectrogram, recalculate average, repeat.
	for warp_iter in range(total_iterations):
		mean_spec = np.mean(warped_specs, axis=0)
		squared_errors = np.zeros(len(specs))
		shift_λ, slope_λ = shift_λs[warp_iter], slope_λs[warp_iter]
		for i in range(len(specs)):
			# Get an objective.
			if slope_λs[warp_iter] == np.inf:
				objective = _get_shift_objective(specs[i], mean_spec, \
						interps[i], shift_λ)
			else:
				objective = _get_linear_objective(specs[i], mean_spec, \
						interps[i], shift_λ, slope_λ)
			# Optimize.
			res = minimize(objective, x0[i], method='Powell')
			x0[i] = res.x
			if slope_λ == np.inf:
				x0[i,1] = 0.0 # slope = 1, log slope = 0
			if not res.success:
				print("Optimization failed:", res.message)
				return None, None
			loss = res.fun
			# Update warped specs.
			warped_specs[i] = interps[i](x0[i,0] + \
					np.exp(x0[i,1])*np.arange(specs[i].shape[1]))
		if verbose:
			temp_loss = round(np.mean(loss),3)
			print("Iteration {}, loss={}".format(warp_iter, temp_loss))
	warp_params = {'shifts': x0[:,0], 'slopes': np.exp(x0[:,1])}
	return warped_specs, warp_params


def _get_linear_objective(spec, target_spec, f, shift_λ, slope_λ):
	""" """
	def objective(x):
		pred_spec = f(x[0] + np.exp(x[1])*np.arange(spec.shape[1]))
		loss = np.sum(np.power(pred_spec - target_spec, 2))
		return loss + shift_λ * np.power(x[0],2) + slope_λ * np.power(x[1], 2)
	return objective


def _get_shift_objective(spec, target_spec, f, shift_λ):
	""" """
	def objective(x):
		pred_spec = f(x[0] + np.arange(spec.shape[1]))
		loss = np.sum(np.power(pred_spec - target_spec, 2))
		return loss + shift_λ * np.power(x[0],2)
	return objective



if __name__ == '__main__':
	pass



###
