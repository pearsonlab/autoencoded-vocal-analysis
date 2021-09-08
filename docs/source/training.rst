Training
========

This section describes how to train the VAE.

Training on Syllables
#####################

By this point, we should have a list of directories containing preprocessed
syllable spectrograms. Our first step is to define PyTorch Dataloader objects
that are responsible for shipping data to and from the GPU. We want one for a
training set, and one for a test set.

.. code:: Python3

	spec_dirs = [...] # directories containing saved spectrograms (hdf5s)
	split = 0.8 # 80/20 train/test split

	# Construct a random train/test partition.
	from ava.models.vae_dataset import get_syllable_partition
	partition = get_syllable_partition(spec_dirs, split)

	# Make Dataloaders.
	from ava.models.vae_dataset import get_syllable_data_loaders
	loaders = get_syllable_data_loaders(partition)



Now we're ready to train the VAE.


.. code:: Python3

	# Construct network.
	from ava.models.vae import VAE
	save_dir = 'model/parameters/should/be/saved/here/'
	model = VAE(save_dir=save_dir)

	# Train.
	model.train_loop(loaders, epochs=101)



This should periodically save the model, print train and test loss, and write
a file in :code:`save_dir` called :code:`reconstruction.pdf` which displays
several spectrograms and their reconstructions.

You may also want to continue training a previously saved model:


.. code:: Python3

	# Make an untrained model.
	model = VAE(save_dir=save_dir)

	# Load saved state.
	model_checkpoint = 'path/to/checkpoint_100.tar'
	model.load_state(model_checkpoint)

	# Train another few epochs.
	model.train_loop(loaders, epochs=51)



Another VAE parameter worth knowing about is :code:`model_precision`, which
controls the reconstruction/regularization tradeoff of the model. Very large
positive values will make the VAE behave more like a deterministic autoencoder,
encouraging better reconstructions. Small positive values produce
better-behaved latents, but with poorer reconstructions.
The default value is 10.  


Shotgun VAE Training
####################

Training the shotgun VAE is pretty similar to training the syllable VAE, the
main difference being the Dataloader we feed it: instead of making spectrograms
beforehand, we make them during training.

.. code:: Python3

	# Define parameters and directories.
	params = {...}
	audio_dirs = [...]
	roi_dirs = [...] # same format as syllable segments

	# Make a Dataloader.
	from ava.models.window_vae_dataset import get_window_partition, \
			get_fixed_window_data_loaders
	split = 0.8 # 80/20 train/test split
	partition = get_window_partition(audio_dirs, roi_dirs, split)
	loaders = get_fixed_window_data_loaders(partition, params)


Then training is the same as before:

.. code:: Python3

	# Train.
	from ava.models.vae import VAE
	save_dir = 'model/parameters/should/be/saved/here/'
	model = VAE(save_dir=save_dir)
	model.train_loop(loaders, epochs=101)


Note that we define segments for the shotgun VAE in :code:`roi_dirs`. These
should have the same format as syllable segments, but should cover longer
periods of vocalization.


Warped Shotgun VAE Training
###########################

TO DO

Mode Collapse
#############

One possible issue during training is known as mode collapse or posterior
collapse. This happens when
the VAE's tendency to regularize overwhelms its ability to reconstruct
spectrograms, and is the tendency of the VAE to ignore its input so that each
reconstruction is simply the mean spectrogram. There are two ways to deal with
this in AVA. First, we can increase the contrast of the spectrograms by
decreasing the range between :code:`'spec_min_val'` and :code:`'spec_max_val'`
in the preprocessing step. Second, we can increase the model precision in the
training step to strike a different regularization/reconstruction tradeoff:

.. code:: Python3

	model = VAE(model_precision=20.0) # default is 10.0
