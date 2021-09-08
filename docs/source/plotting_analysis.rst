Plotting and Analysis
=====================

The :code:`ava.plotting` subpackage contains useful functions for plotting and
analysis. These are very easy to use if you've already made a DataContainer
object (see previous section):

.. code:: Python3

	from ava.plotting.grid_plot import indexed_grid_plot_DC

	dc = ... # define DataContainer
	indices = [[0,1,2], [3,4,5]]
	indexed_grid_plot_DC(dc, indices)


This plots a 2-by-3 grid of spectrograms with indices determined by
:code:`indices` and saves the image to
:code:`os.path.join(dc.plots_dir, 'grid.pdf')`.
Try :code:`help(ava.plotting.grid_plot)` or read the docs for more options.


Another useful plot:

.. code:: Python3

	from ava.plotting.latent_projection import latent_projection_plot_DC

	dc = ... # define DataContainer
	latent_projection_plot_DC(dc, embedding_type='latent_mean_umap')


This plots a UMAP projection of the latent means and saves the result to
:code:`os.path.join(dc.plots_dir, 'latent.pdf')`.

See the :code:`ava.plotting` documentation for more plotting and analysis tools.


Shotgun VAE
###########

In order to use some of these plotting functions with the shotgun VAE, we need
to first save some spectrograms. The shotgun VAE dataset has a method to do
this: ``write_hdf5_files``

.. code:: Python3

	spec_dir = 'where/to/save/specs'
	loaders['test'].dataset.write_hdf5_files(spec_dir, num_files=1000)


Warped Shotgun VAE
##################

TO DO
