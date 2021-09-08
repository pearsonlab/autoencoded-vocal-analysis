"""
Plot a latent mean projection.

"""
__date__ = "July 2019 - December 2020"


try: # Numba >= 0.52
	from numba.core.errors import NumbaPerformanceWarning
except ModuleNotFoundError:
	try: # Numba <= 0.45
		from numba.errors import NumbaPerformanceWarning
	except (NameError, ModuleNotFoundError):
		pass
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import numpy as np
import joblib
import umap
import warnings

from ava.data.data_container import PRETTY_NAMES



def latent_projection_plot_DC(dc, embedding_type='latent_mean_umap', \
	color_by=None, title=None, filename='latent.pdf', colorbar=False, \
	colormap='viridis', alpha=0.5, s=0.9, ax=None, cax=None, shuffle=True, \
	save_and_close=True, show_axis=False, default_color='b', \
	condition_func=None):
	"""
	Make a scatterplot of latent means projected to two dimensions.

	Parameters
	----------
	dc : ava.data.data_container.DataContainer
		See ava.data.data_container.
	embedding_type : str, optional
		Defaults to ``'latent_mean_umap'``.
	color_by : {str, None}, optional
		If ``None``, all points are colored ``default_color``. Otherwise,
		``color_by`` is requested from the DataContainer and passed to the
		matplotlib.pyplot.scatter as the color parameter. The one exception is
		if ``color_by == 'filename_lambda'``, in which case, scatter color is
		some function of the audio filename, passed to ``condition_func``.
		Defaults to ``None``.
	title : {str, None}, optional
		Plot title. Defaults to ``None``.
	filename : str, optional
		Where to save the image, relative to ``dc.plots_dir``. Defaults to
		``'latent.pdf'``.
	colorbar : bool, optional
		Whether to include a colorbar. Defaults to ``False``.
	colormap : str, optional
		The pyplot colormap to use if ``color_by`` returns numerical values.
		Defaults to ``'viridis'``.
	alpha : float, optional
		Alpha value of scatterpoints. Defaults to ``0.5``.
	s : float, optional
		Size of scatterpoints. Defaults to ``0.9``.
	ax : {matplotlib.axes._subplots.AxesSubplot, None}, optional
	 	Scatter axis. If ``None``, ``matplotlib.pyplot.gca()`` is used. Defaults
		to ``None``.
	cax : {matplotlib.axes._subplots.AxesSubplot, None}, optional
		Colorbar axis. If ``None``, an axis is made. Defaults to ``None``.
	shuffle : bool, optional
		Whether to shuffle the zorder of points. Defaults to ``True``.
	save_and_close : bool, optional
		Defaults to ``True``.
	show_axis : bool, optional
		Defaults to ``False``.
	default_color : str, optional
		Defaults to ``'b'``.
	condition_func : {function, None}, optional
		Only used when ``color_by == 'filename_lambda'``, in which case
		``condition_func`` maps audio filenames to pyplot colors. Defaults to
		``None``.
	"""
	embedding = dc.request(embedding_type)
	if color_by is None:
		color = default_color
	elif color_by == 'filename_lambda':
		assert condition_func is not None
		fns = dc.request('audio_filenames')
		color = [condition_func(fn) for fn in fns]
		alpha = None # Let condition_func handle alpha values.
	else:
		color = dc.request(color_by)
	if title is None and color_by not in [None, 'filename_lambda']:
		title = PRETTY_NAMES[color_by]
	if dc.plots_dir is not None:
		filename = os.path.join(dc.plots_dir, filename)
	projection_plot(embedding, color=color, title=title, \
		save_filename=filename, colorbar=colorbar, colormap=colormap, \
		shuffle=shuffle, alpha=alpha, s=s, ax=ax, cax=cax, \
		save_and_close=save_and_close, show_axis=show_axis)


def latent_projection_plot_with_noise_DC(dc, noise_box,
	embedding_type='latent_mean_umap', color_by=None, title=None, \
	filename='latent.pdf', colorbar=False, colormap='viridis', alpha=0.5, \
	s=0.9, ax=None, cax=None, shuffle=True, save_and_close=True, \
	show_axis=False, default_color='b', condition_func=None):
	"""
	Same as `latent_projection_plot_DC`, but with noise to exclude.

	Parameters
	----------
	dc : ava.data.data_container.DataContainer
		DataContainer object.
	noise_box : list of float
		Must contain four elements: ``[x1, x2, y1, y2]``, which are interpreted
		as a region of the latent mean embedding containing noise. The points
		within this rectangle are excluded.

	Note
	----
	For more parameters, see latent_projection_plot_DC.
	"""
	embedding = dc.request(embedding_type)
	indices = []
	x1, x2, y1, y2 = noise_box
	for i in range(len(embedding)):
		if embedding[i,0] < x1 or embedding[i,0] > x2 or \
				embedding[i,1] < y1 or embedding[i,1] > y2:
			indices.append(i)
	indices = np.array(indices, dtype='int')
	try:
		default_color = np.array(default_color)[indices]
	except:
		pass
	latent = dc.request('latent_means')[indices]
	transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, \
		metric='euclidean', random_state=42)
	with warnings.catch_warnings():
		try:
			warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
		except NameError:
			pass
		embedding = transform.fit_transform(latent)
	if color_by is None:
		color = default_color
	elif color_by == 'filename_lambda':
		assert condition_func is not None
		fns = dc.request('audio_filenames')[indices]
		color = [condition_func(fn) for fn in fns]
	else:
		color = dc.request(color_by)[indices]
	if title is None and color_by not in [None, 'filename_lambda']:
		title = PRETTY_NAMES[color_by]
	if dc.plots_dir is not None:
		filename = os.path.join(dc.plots_dir, filename)
	projection_plot(embedding, color=color, title=title, \
		save_filename=filename, colorbar=colorbar, colormap=colormap, shuffle=shuffle, \
		alpha=alpha, s=s, ax=ax, cax=cax, save_and_close=save_and_close, \
		show_axis=show_axis)


def projection_plot(embedding, color='b', title=None,
	save_filename='latent.pdf', colorbar=False, shuffle=True, \
	colormap='viridis', alpha=0.6, s=0.9, ax=None, cax=None, \
	save_and_close=True, show_axis=False):
	"""
	Plot a projection of the data.

	Parameters
	----------
	embedding : numpy.ndarray
		Data embedding.
	color : {str, numpy.ndarray}, optional
		Defaults to ``'b'``.
	title : {str, None}, optional
		Defaults to ``None``.
	save_filename : str, optional
		Defaults to ``'temp.pdf'``.
	colorbar : bool, optional
		Whether to include a colorbar. Defaults to ``False``.
	shuffle : bool, optional
		Whether to shuffle the zorder of points. Defaults to ``True``.
	colormap : str, optional
		The pyplot colormap to use if ``color_by`` returns numerical values.
		Defaults to ``'viridis'``.
	alpha : float, optional
		Alpha value of scatterpoints. Defaults to ``0.5``.
	s : float, optional
		Size of scatterpoints. Defaults to ``0.9``.
	ax : {matplotlib.axes._subplots.AxesSubplot, None}, optional
	 	Scatter axis. If ``None``, ``matplotlib.pyplot.gca()`` is used. Defaults
		to ``None``.
	cax : {matplotlib.axes._subplots.AxesSubplot, None}, optional
		Colorbar axis. If ``None``, an axis is made. Defaults to ``None``.
	save_and_close : bool, optional
		Defaults to ``True``.
	show_axis : bool, optional
		Defaults to ``False``.
	"""
	X, Y = embedding[:,0], embedding[:,1]
	if shuffle:
		np.random.seed(42)
		perm = np.random.permutation(len(X))
		np.random.seed(None)
		X, Y = X[perm], Y[perm]
		try:
			color = np.array(color)[perm]
		except IndexError:
			pass
	if ax is None:
		ax = plt.gca()
	im = ax.scatter(X, Y, c=color, alpha=alpha, s=s, cmap=colormap)
	ax.set_aspect('equal')
	if title is not None and len(title) > 0:
		ax.set_xlabel(title, fontdict={'fontsize':8}) # Really a title
	if not show_axis:
		ax.get_xaxis().set_ticks([])
		ax.get_yaxis().set_ticks([])
		ax.spines['right'].set_visible(False)
		ax.spines['left'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.spines['bottom'].set_visible(False)
	else:
		ax.grid(True)
	if colorbar:
		min_val, max_val = np.min(color), np.max(color)
		ticks = [int(round(i)) for i in [0.8*min_val+0.2*max_val, \
			0.5*(min_val+max_val), 0.8*max_val+0.2*min_val]]
		fig = plt.gcf()
		if cax is None:
			cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
		cbar = fig.colorbar(im, cax=cax, fraction=0.046, \
			orientation="horizontal", ticks=ticks)
		cbar.solids.set_edgecolor("face")
		cbar.solids.set_rasterized(True)
		cbar.ax.set_xticklabels([str(int(round(t))) for t in ticks])
	save_dir = os.path.split(save_filename)[0]
	if save_dir != '' and not os.path.exists(save_dir):
		os.makedirs(save_dir)
	if save_and_close:
		plt.tight_layout()
		plt.savefig(save_filename)
		plt.close('all')




if __name__ == '__main__':
	pass


###
