Data Management
===============

We have a lot of different kinds of data floating around: audio files, text
files, hdf5s, saved models, etc. To simplify plotting and analysis, we have a
:code:`DataContainer` object that ties together the different strands of data.
So imagine this is the file tree of your project:

::

	├── animal_1
	│   ├── audio                    (raw audio)
	│   │   ├── foo.wav
	│   │   ├── bar.wav
	│   │   └── baz.wav
	│   ├── features                 (output of MUPET, DeepSqueak, SAP, ...)
	│   │   ├── foo.csv
	│   │   ├── bar.csv
	│   │   └── baz.csv
	│   ├── specs                    (used to train models, written by
	│   │   ├── syllables_000.hdf5   preprocessing.preprocess.process_sylls)
	│   │   └── syllables_001.hdf5
	│   └── projs                    (latent means, UMAP, PCA, tSNE
	│      ├── syllables_000.hdf5    projections, copies of features in
	│      └── syllables_001.hdf5    experiment_1/features. These are
	│                                written by a DataContainer object.)
	├── animal_2
	│   ├── audio
	│   │   ├── 1.wav
	│   │   └── 2.wav
	│   ├── features
	│   │   ├── 1.csv
	│   │   └── 2.csv
	│   ├── specs
	│   │   ├── syllables_000.hdf5
	│   │   └── syllables_001.hdf5
	│   └── projs
	│       ├── syllables_000.hdf5
	│       └── syllables_001.hdf5
	│
	├── checkpoint_150.tar            (saved model)
	.
	.
	.

We can use the DataContainer object to tie this all together:

.. code:: Python3

	import os
	from ava.data.data_container import DataContainer

	root = 'path/to/project/'
	animals = ['animal_1', 'animal_2']
	audio_dirs = [os.path.join(root, animal, 'audio') for animal in animals]
	feature_dirs = [os.path.join(root, animal, 'features') for animal in animals]
	spec_dirs = [os.path.join(root, animal, 'specs') for animal in animals]
	projection_dirs = [os.path.join(root, animal, 'projs') for animal in animals]
	model_fn = os.path.join(root, 'checkpoint_150.tar')

	dc = DataContainer(audio_dirs=audio_dirs, feature_dirs=feature_dirs, \
			spec_dirs=spec_dirs, projection_dirs=projection_dirs, \
			model_filename=model_fn)


The DataContainer has one main publicly-facing method: :code:`request`

.. code:: Python3

	latent = dc.request('latent_means')

	print(type(latent)) # <class 'numpy.ndarray'>
	print(latent.shape) # (18020, 32) == (num_syllables, latent_dimension)


Behind the scenes the DataContainer constructs a VAE, loads your model state,
feeds all the preprocessed spectrograms through the encoder, and collects the
latent means.

You can also request fields such as :code:`'segment_audio'`,
:code:`'latent_mean_umap'`, and :code:`'specs'`. For a complete list of fields
you can request, see the docs for :code:`ava.data.data_container`.
