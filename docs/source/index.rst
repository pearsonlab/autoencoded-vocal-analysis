AVA: autoencoded vocal analysis
===============================

Welcome to AVA, a python package for inferring latent descriptions of animal
vocalizations using variational autoencoders. See our article for details:

	Goffinet, J., Brudner, S., Mooney, R., & Pearson, J. (2021). Low-dimensional
	learned feature spaces quantify individual and group differences in vocal
	repertoires. *eLife*, 10:e67855.
	`https://doi.org/10.7554/eLife.67855 <https://doi.org/10.7554/eLife.67855>`_

You can find the code
`on github <https://github.com/jackgoffinet/autoencoded-vocal-analysis>`_.


**Quick Install**

.. code-block:: bash

   $ git clone https://github.com/jackgoffinet/autoencoded-vocal-analysis.git
   $ cd autoencoded-vocal-analysis
   $ pip install .


**Examples**

See the `examples/` subdirectory
`on github <https://github.com/jackgoffinet/autoencoded-vocal-analysis>`_.


.. toctree::
   :maxdepth: 1
   :caption: User Guide:

   install
   overview
   segment
   preprocess
   training
   data_management
   plotting_analysis


.. toctree::
   :maxdepth: 1
   :caption: Docs:

   ava.segmenting
   ava.preprocessing
   ava.models
   ava.plotting


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
