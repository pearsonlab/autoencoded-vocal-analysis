"""
Make a movie out of a shotgun VAE projection and an audio file.

Reduce speed by 50%:

::

ffmpeg -i out.mp4 -filter_complex "[0:v]setpts=PTS/0.5[v];[0:a]atempo=0.5[a]" -map "[v]" -map "[a]" -strict -2 out.mp4

TO DO
-----
* Check whether ffmpeg is installed.

"""
__date__ = "November 2019 - November 2020"


import joblib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning
from scipy.spatial.distance import euclidean, correlation
from sklearn.neighbors import NearestNeighbors
import subprocess
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

from ava.models.vae import VAE



def shotgun_movie_DC(dc, audio_file, p, method='spectrogram_correlation', \
	output_dir='temp', fps=30, shoulder=0.01, c='b', alpha=0.2, s=0.9, \
	marker_c='r', marker_s=50.0, marker_marker='*', transform_fn=None,
	load_transform=False, save_transform=False, mp4_fn='out.mp4'):
	"""
	Make a shotgun VAE projection movie with the given audio file.

	Parameters
	----------
	dc : ava.data.data_container.DataContainer
		See ava.data.data_container.
	audio_file : str
		Path to audio file.
	p : dict
		Preprocessing parameters. Must contain keys: ``'fs'``, ``'get_spec'``,
		``'num_freq_bins'``, ``'num_time_bins'``, ``'nperseg'``, ``'noverlap'``,
		``'window_length'``, ``'min_freq'``, ``'max_freq'``, ``'spec_min_val'``,
		``'spec_max_val'``, ``'mel'``, ...
	method : str, optional
		How to map spectrograms to points in the UMAP embedding. `'latent_nn'`
		assigns embeddings based on nearest neighbors in latent space.
		`'re_umap'` uses a pretrained UMAP object to map the spectrogram's
		latent features directly. `'spectrogram_correlation'` finds the
		spectrogram with the highest correlation. Defaults to
		`'spectrogram_correlation'`.
	output_dir : str, optional
		Directory where output images are written. Defaults to ``'temp'``.
	fps : int, optional
		Frames per second. Defaults to ``20``.
	shoulder : float, optional
		The movie will start this far into the audio file and stop this far from
		the end. This removes weird edge effect of making spectrograms. Defaults
		to ``0.05``.
	c : str, optional
		Passed to ``matplotlib.pyplot.scatter`` for background points. Defaults
		to ``'b'``.
	alpha : float, optional
		Passed to ``matplotlib.pyplot.scatter`` for background points. Defaults
		to ``0.2``.
	s : float, optional
		Passed to ``matplotlib.pyplot.scatter`` for background points. Defaults
		to ``0.9``.
	marker_c : str, optional
		Passed to ``matplotlib.pyplot.scatter`` for the marker. Defaults to
		``'r'``.
	marker_s : float, optional
		Passed to ``matplotlib.pyplot.scatter`` for the marker. Defaults to
		``40.0``.
	marker_marker : str, optional
		Passed to ``matplotlib.pyplot.scatter`` for the marker. Defaults to
		``'r'``.
	"""
	assert dc.model_filename is not None
	assert method in ['latent_nn', 're_umap', 'spectrogram_correlation']
	if os.path.exists(output_dir):
		for fn in os.listdir(output_dir):
			if len(fn) > 4 and fn[-4:] == '.jpg':
				os.remove(os.path.join(output_dir, fn))
	# Read the audio file.
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", category=WavFileWarning)
		fs, audio = wavfile.read(audio_file)
	assert fs == p['fs'], "found fs="+str(fs)+", expected "+str(p['fs'])
	# Make spectrograms.
	specs = []
	dt = 1/fps
	onset = shoulder
	while onset + p['window_length'] < len(audio)/fs - shoulder:
		offset = onset + p['window_length']
		target_times = np.linspace(onset, offset, p['num_time_bins'])
		# Then make a spectrogram.
		spec, flag = p['get_spec'](onset-shoulder, offset+shoulder, audio, p, \
				fs=fs, target_times=target_times)
		assert flag
		specs.append(spec)
		onset += dt
	assert len(specs) > 0
	specs = np.stack(specs)
	if method in ['latent_nn', 're_umap']:
		# Make a DataLoader out of these spectrograms.
		loader = DataLoader(SimpleDataset(specs))
		# Get latent means.
		model = VAE()
		model.load_state(dc.model_filename)
		latent = model.get_latent(loader)
	if method == 'latent_nn':
		# Get original latent and embeddings.
		original_embed = dc.request('latent_mean_umap')
		original_latent = dc.request('latent_means')
		# Find nearest neighbors in latent space to determine embeddings.
		new_embed = np.zeros((len(latent),2))
		for i in range(len(latent)):
			index = np.argmin([euclidean(latent[i], j) for j in original_latent])
			new_embed[i] = original_embed[index]
	elif method == 're_umap':
		# Get transform.
		if load_transform:
			transform = joblib.load(transform_fn)
			original_embed = dc.request('latent_mean_umap')
		else:
			latent_means = dc.request('latent_means')
			transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1,\
				metric='euclidean', random_state=42)
			original_embed = transform.fit_transform(latent_means)
			if save_transform:
				joblib.dump(transform, transform_fn)
		# Make projections.
		new_embed = transform.transform(latent)
	elif method == 'spectrogram_correlation':
		dim = specs.shape[1] * specs.shape[2]
		specs = specs.reshape(-1, dim)
		original_specs = dc.request('specs')
		original_specs = original_specs.reshape(-1, dim)
		print("Finding nearest neighbors:")
		nbrs = NearestNeighbors(n_neighbors=1, metric='correlation')
		nbrs.fit(original_specs)
		indices = nbrs.kneighbors(specs, return_distance=False).flatten()
		print("\tDone.")
		original_embed = dc.request('latent_mean_umap')
		new_embed = original_embed[indices]
	# Calculate x and y limits.
	xmin = np.min(original_embed[:,0])
	ymin = np.min(original_embed[:,1])
	xmax = np.max(original_embed[:,0])
	ymax = np.max(original_embed[:,1])
	x_pad = 0.05 * (xmax - xmin)
	y_pad = 0.05 * (ymax - ymin)
	xmin, xmax = xmin - x_pad, xmax + x_pad
	ymin, ymax = ymin - y_pad, ymax + y_pad
	# Save images.
	print("Saving images:")
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	for i in range(len(new_embed)):
		plt.scatter(original_embed[:,0], original_embed[:,1], \
				c=[c]*len(original_embed), alpha=alpha, s=s)
		plt.scatter([new_embed[i,0]], [new_embed[i,1]], s=marker_s, \
				marker=marker_marker, c=marker_c)
		plt.xlim(xmin, xmax)
		plt.ylim(ymin, ymax)
		plt.gca().set_aspect('equal')
		plt.axis('off')
		plt.savefig(os.path.join(output_dir, f"viz-{i:05d}.jpg"))
		plt.close('all')
	print("\tDone.")
	# Make video.
	img_fns = os.path.join(output_dir, 'viz-%05d.jpg')
	video_fn = os.path.join(output_dir, mp4_fn)
	bashCommand = 'ffmpeg -y -r {} -i {} {}'.format(fps, img_fns, 'temp.mp4')
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()
	bashCommand = 'ffmpeg -y -r {} -i {} -i {} -c:a aac -strict ' + \
			'-2 {}'.format(fps,'temp.mp4',audio_file,video_fn)
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()



class SimpleDataset(Dataset):
	def __init__(self, specs):
		self.specs = specs

	def __len__(self):
		return self.specs.shape[0]

	def __getitem__(self, index):
		return torch.from_numpy(self.specs[index]).type(torch.FloatTensor)



if __name__ == '__main__':
	pass



###
