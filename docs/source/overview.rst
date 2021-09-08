Package Overview
================

AVA is composed of five subpackages: :code:`segmenting`, :code:`preprocessing`,
:code:`models`, :code:`data`, and :code:`plotting`. There are three sorts of
analyses AVA can do: a syllable-level analysis, a shotgun VAE analysis, and a
warped-time shotgun VAE analysis, and each of these make use of the five
subpackages in this order:

* The segmenting subpackage identifies syllables for syllable-level
  analysis and song renditions for the warped-time shotgun VAE analysis.
* The preprocessing subpackage creates spectrograms to feed into the VAE.
* The models subpackage defines the VAE as well as PyTorch DataLoader objects
  that feed data to the VAE.
* The data subpackage defines the DataContainer object, which is useful for
  organizing various files for subsequent plotting and analysis.
* The plotting subpackage contains various functions for plotting and analysis.


The remaining sections go through each subpackage in more detail.

If you prefer to learn from examples, see the :code:`examples/` subdirectory
in the `code <https://github.com/jackgoffinet/autoencoded-vocal-analysis/>`__,
which contains python scripts implementing these three sorts of analysis.
