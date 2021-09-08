"""
Plot a grid of spectrograms.

"""
__date__ = "July-August 2019"


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os



def indexed_grid_plot_DC(dc, indices, ax=None, save_and_close=True, gap=3, \
	side_len=128, filename='grid.pdf'):
	"""
	Plot a grid of spectrograms.

	Parameters
	----------
	dc : ava.data.data_container.DataContainer
		Data container object.
	indices : list of lists of ints
		Spectrogram indices.
	ax : matplotlib.axes._subplots.AxesSubplot or None
		Plotting axis. Defaults to `None`.
	save_and_close : bool
		Save and close the plot. Defaults to `True`.
	gap : int
		Number of pixels between spectrograms. Defaults to `3`.
	side_len : int
		Spectrogram height and width, in pixels. Defaults to `128`.
	filename : str
		Save the image here. Defaults to `'grid.pdf'`.
	"""
	specs = dc.request('specs')
	a, b, c, d = len(indices), len(indices[0]), side_len, side_len
	result = np.zeros((a,b,c,d))
	for i, row in enumerate(indices):
		for j, col in enumerate(row):
			result[i,j] = specs[col]
	filename = os.path.join(dc.plots_dir, filename)
	grid_plot(result, gap=gap, ax=ax, \
			save_and_close=save_and_close, filename=filename)


def grid_plot(specs, gap=3, vmin=0.0, vmax=1.0, ax=None, save_and_close=True, \
	filename='temp.pdf'):
	"""
	Parameters
	----------
	specs : numpy.ndarray
		Spectrograms
	gap : int or tuple of two ints, optional
		The vertical and horizontal gap between images, in pixels. Defaults to
		`3`.
	vmin : float, optional
		Passed to matplotlib.pyplot.imshow. Defaults to `0.0`.
	vmax : float, optional
		Passed to matplotlib.pyplot.imshow. Defaults to `1.0`.
	ax : matplotlib.pyplot.axis, optional
		Axis to plot figure. Defaults to matplotlib.pyplot.gca().
	save_and_close : bool, optional
		Whether to save and close after plotting. Defaults to True.
	filename : str, optional
		Save the image here.
	"""
	if type(gap) == type(4):
		gap = (gap,gap)
	try:
		a, b, c, d = specs.shape
	except:
		print("Invalid shape:", specs.shape, "Should have 4 dimensions.")
		quit()
	dx, dy = d+gap[1], c+gap[0]
	height = a*c + (a-1)*gap[0]
	width = b*d + (b-1)*gap[1]
	img = np.zeros((height, width))
	for j in range(a):
		for i in range(b):
			img[j*dy:j*dy+c,i*dx:i*dx+d] = specs[-j-1,i]
	for i in range(1,b):
		img[:,i*dx-gap[1]:i*dx] = np.nan
	for j in range(1,a):
		img[j*dy-gap[0]:j*dy,:] = np.nan
	if ax is None:
		ax = plt.gca()
	ax.imshow(img, aspect='equal', origin='lower', interpolation='none',
		vmin=vmin, vmax=vmax)
	ax.axis('off')
	if save_and_close:
		plt.tight_layout()
		plt.savefig(filename)
		plt.close('all')



if __name__ == '__main__':
	pass


###
