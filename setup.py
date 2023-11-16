import setuptools

import ava

with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	name="AVA: Autoencoded Vocal Analysis",
	version=ava.__version__,
	author="Jack Goffinet",
	author_email="jack.goffinet@duke.edu",
	description="Generative modeling of animal vocalizations",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/pearsonlab/autoencoded-vocal-analysis",
	packages=setuptools.find_packages(),
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
	install_requires = [
		"torch>=1.1,!=1.12.0",
		"numpy",
		"matplotlib",
		"joblib",
		"umap-learn",
		"numba",
		"scikit-learn",
		"scipy",
		"bokeh",
		"h5py",
		"pytest",
		"tqdm",
		"affinewarp @ git+https://github.com/ahwillia/affinewarp.git"
	]
)
