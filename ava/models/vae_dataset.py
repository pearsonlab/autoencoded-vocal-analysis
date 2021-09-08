"""
Methods for feeding syllable data to the VAE.

Meant to be used with `ava.models.vae.VAE`.
"""
__date__ = "November 2018 - July 2020"


import h5py
import numpy as np
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader

from ava.models.utils import _get_sylls_per_file, numpy_to_tensor, \
		get_hdf5s_from_dir


EPSILON = 1e-9


def get_syllable_partition(dirs, split, shuffle=True, max_num_files=None):
	"""
	Partition the filenames into a random test/train split.

	Parameters
	----------
	dirs : list of strings
		List of directories containing saved syllable hdf5 files.
	split : float
		Portion of the hdf5 files to use for training,
		:math:`0 < \mathtt{split} \leq 1.0`
	shuffle : bool, optional
		Whether to shuffle the hdf5 files. Defaults to `True`.
	max_num_files : {int, None}, optional
		The number of files in the train and test partitions <= `max_num_files`.
		If ``None``, all files are used. Defaults to ``None``.

	Returns
	-------
	partition : dict
		Contains two keys, ``'test'`` and ``'train'``, that map to lists of hdf5
		files. Defines the random test/train split.
	"""
	assert(split > 0.0 and split <= 1.0)
	# Collect filenames.
	filenames = []
	for dir in dirs:
		filenames += get_hdf5s_from_dir(dir)
	# Reproducibly shuffle.
	filenames = sorted(filenames)
	if shuffle:
		np.random.seed(42)
		np.random.shuffle(filenames)
		np.random.seed(None)
	if max_num_files is not None:
		filenames = filenames[:max_num_files]
	# Split.
	index = int(round(split * len(filenames)))
	return {'train': filenames[:index], 'test': filenames[index:]}


def get_syllable_data_loaders(partition, batch_size=64, shuffle=(True, False), \
	num_workers=4):
	"""
	Return a pair of DataLoaders given a test/train split.

	Parameters
	----------
	partition : dictionary
		Test train split: a dictionary that maps the keys 'test' and 'train'
		to disjoint lists of .hdf5 filenames containing syllables.
	batch_size : int, optional
		Batch size of the returned Dataloaders. Defaults to 32.
	shuffle : tuple of bools, optional
		Whether to shuffle data for the train and test Dataloaders,
		respectively. Defaults to (True, False).
	num_workers : int, optional
		How many subprocesses to use for data loading. Defaults to 3.

	Returns
	-------
	dataloaders : dictionary
		Dictionary mapping two keys, ``'test'`` and ``'train'``, to respective
		torch.utils.data.Dataloader objects.
	"""
	sylls_per_file = _get_sylls_per_file(partition)
	train_dataset = SyllableDataset(filenames=partition['train'], \
		transform=numpy_to_tensor, sylls_per_file=sylls_per_file)
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, \
		shuffle=shuffle[0], num_workers=num_workers)
	if not partition['test']:
		return {'train':train_dataloader, 'test':None}
	test_dataset = SyllableDataset(filenames=partition['test'], \
		transform=numpy_to_tensor, sylls_per_file=sylls_per_file)
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size, \
		shuffle=shuffle[1], num_workers=num_workers)
	return {'train':train_dataloader, 'test':test_dataloader}



class SyllableDataset(Dataset):
	"""torch.utils.data.Dataset for animal vocalization syllables"""

	def __init__(self, filenames, sylls_per_file, transform=None):
		"""
		Create a torch.utils.data.Dataset for animal vocalization syllables.

		Parameters
		----------
		filenames : list of strings
			List of hdf5 files containing syllable spectrograms.
		sylls_per_file : int
			Number of syllables in each hdf5 file.
		transform : None or function, optional
			Transformation to apply to each item. Defaults to None (no
			transformation)
		"""
		self.filenames = filenames
		self.sylls_per_file = sylls_per_file
		self.transform = transform

	def __len__(self):
		return len(self.filenames) * self.sylls_per_file

	def __getitem__(self, index):
		result = []
		single_index = False
		try:
			iterator = iter(index)
		except TypeError:
			index = [index]
			single_index = True
		for i in index:
			# First find the file.
			load_filename = self.filenames[i // self.sylls_per_file]
			file_index = i % self.sylls_per_file
			# Then collect fields from the file.
			with h5py.File(load_filename, 'r') as f:
				spec = f['specs'][file_index]
			if self.transform:
				spec = self.transform(spec)
			result.append(spec)
		if single_index:
			return result[0]
		return result



if __name__ == '__main__':
	pass


###
