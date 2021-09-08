Segmenting
==========


Importing syllable segments
###########################

AVA reads syllable segments from :code:`.txt` files with two tab-separated
columns containing onsets and offsets for each syllable. Lines beginning with
:code:`#` are ignored.

.. note:: This is the default format for Numpy's :code:`loadtxt`
	function. It also happens to be Audacity's label format.

AVA provides a function for copying onsets and offsets from other formats to the
standard format:

.. code:: Python3

	from ava.segmenting.utils import copy_segments_to_standard_format
	help(copy_segments_to_standard_format)

This has been tested with MUPET and Deepsqueak file formats. Also check out
`vak <https://github.com/NickleDave/vak>`_ and
`TweetyNet <https://github.com/yardencsGitHub/tweetynet>`_ for syllable
segmentation.

Syllable segmenting in AVA
##########################

You can also use AVA's built-in segmenting functions. Here, we'll go through the
amplitude segmentation method. First, import the
segmenting function and set a bunch of segmenting parameters:

.. code:: Python3

	from ava.segmenting.amplitude_segmentation import get_onsets_offsets

	seg_params = {
	    'min_freq': 30e3, # minimum frequency
	    'max_freq': 110e3, # maximum frequency
	    'nperseg': 1024, # FFT
	    'noverlap': 512, # FFT
	    'spec_min_val': 2.0, # minimum STFT log-modulus
	    'spec_max_val': 6.0, # maximum STFT log-modulus
	    'fs': 250000, # audio samplerate
	    'th_1':1.5, # segmenting threshold 1
	    'th_2':2.0, # segmenting threshold 2
	    'th_3':2.5, # segmenting threshold 3
	    'min_dur':0.03, # minimum syllable duration
	    'max_dur': 0.2, # maximum syllable duration
	    'smoothing_timescale': 0.007, # amplitude
	    'softmax': False, # apply softmax to the frequency bins to calculate
	                      # amplitude
	    'temperature':0.5, # softmax temperature parameter
	    'algorithm': get_onsets_offsets, # (defined above)
	}

.. note:: AVA only reads audio files in :code:`.wav` format!

Then we can tune these parameter values by visualizing segmenting decisions:

.. code:: Python3

	from ava.segmenting.segment import tune_segmenting_params
	audio_directories = [...] # list of audio directories
	seg_params = tune_segmenting_params(audio_directories, seg_params)



This will start an interactive tuning process, where parameters can be adjusted
and the resulting segmenting decisions will be displayed in a saved image, by
default :code:`temp.pdf`. The three thresholds will be displayed with an
amplitude trace, detected onsets and offsets, and a spectrogram.

From :code:`ava.segmenting.amplitude_segmentation.get_onsets_offsets`:

	A syllable is detected if the amplitude trace exceeds ``p['th_3']``. An offset
	is then detected if there is a subsequent local minimum in the amplitude
	trace with amplitude less than ``p['th_2']``, or when the amplitude drops
	below ``p['th_1']``, whichever comes first. Syllable onset is determined
	analogously.

Once we're happy with a particular set of parameters, we can go through a whole
collection of audio files and write segmenting decisions in corresponding
directories. It's useful to have a 1-to-1 correspondence between audio
directories and segmenting directories.

.. code:: Python3

	from ava.segmenting.segment import segment
	audio_dirs = ['path/to/animal1/audio/', 'path/to/animal2/audio/']
	segment_dirs = ['path/to/animal1/segments/', 'path/to/animal2/segments/']
	for audio_dir, segment_dir in zip(audio_dirs, segment_dirs):
		segment(audio_dir, segment_dir, seg_params)



Or, parallelized this time:

.. code:: Python3

	from joblib import Parallel, delayed
	from itertools import repeat
	gen = zip(audio_dirs, segment_dirs, repeat(seg_params))
	Parallel(n_jobs=4)(delayed(segment)(*args) for args in gen)



Song segmenting in AVA
######################

For stereotyped vocalizations that are longer than about 100ms, such as adult
zebra finch song motifs, it's easy to extract segments by finding peaks in
cross correlation between a template spectrogram and the spectrograms of a
large collection of audio files. To do this with AVA, we need to first collect
several short audio files of the song motif and save them in the same directory,
say :code:`path/to/template_audio/`. These should should all be roughly the same
duration.

.. code:: Python3

	params = {
		'min_freq': 400, # minimum frequency
		'max_freq': 10e3, # maximum frequency
		'nperseg': 512, # FFT
		'noverlap': 256, # FFT
		'spec_min_val': 2.0, # minimum spectrogram value
		'spec_max_val': 6.5, # maximum spectrogram value
		'fs': 44100, # audio samplerate
	}

	from ava.segmenting.template_segmentation import get_template
	template_dir = 'path/to/template_audio/'
	template = get_template(template_dir, params)


Next we can collect peaks in the cross correlation above a semi-automatically
determined threshold: the median cross-correlation plus some number of median
absolute deviations, :code:`num_mad`:

.. code:: Python3

	from ava.segmenting.template_segmentation import segment_files
	audio_dirs = [...] # list of audio directories
	song_seg_dirs = [...] # list of segment directories
	result = segment_files(audio_dirs, song_seg_dirs, template, params, \
			num_mad=8.0, n_jobs=8)


.. note:: Since this step can take a while if you have many files to segment,
	the function
	:code:`ava.segmenting.template_segmentation.read_segment_decisions` reads
	and returns the same result as :code:`segment_files`.

If we've chosen a good threshold, :code:`num_mad`, then we should have collected
all of the song motifs as well as some false positives. Next we can clean out
the false positives by running UMAP on the putative song segments and selecting
the cluster that corresponds to song motifs.

.. code:: Python3

	from ava.segmenting.template_segmentation import clean_collected_segments
	clean_collected_segments(result, audio_dirs, song_seg_dirs, params)


This will ask you for x and y coordinates of rectangles that surround the song
motif. A tooltip plot is made to match scatterpoints to spectrograms, by default
saved in :code:`'html/'`. You can enter multiple rectangles to cover more
complicated regions. Selected songs will be shown in blue in an image, by
default saved as :code:`'temp.pdf'`.

When we press :code:`c` to continue, the function will keep running, going
through each segment directory and cleaning out the segments that don't get
projected into one of the boxes we just defined.


Syllable segmenting from song segments in AVA
#############################################

Another way to segment syllables of adult zebra finch song is to first segment
out full song motifs and then align all of the full motifs and then extract
the constituent syllables. This is what the function
:code:`ava.segmenting.template_segmentation.segment_sylls_from_songs` is
intended to do.

.. code:: Python3

	params = {...} # same as previous section

	from ava.segmenting.template_segmentation import segment_sylls_from_songs
	audio_dirs = [...] # audio directories
	song_seg_dirs = [...] # directories containing song segments
	syll_seg_dirs = [...] # where we'll write syllable segments
	segment_sylls_from_songs(audio_dirs, song_seg_dirs, syll_seg_dirs, params)


This will collect all the songs, align them, and save an image of the result to,
by default, :code:`temp.pdf`. Then the function will ask for quantiles (between
0 and 1), and the image will be rewritten to reflect the new quantile. A
quantile can be deleted by entering it again. When we press :code:`s` to stop,
the syllables between each consecutive quantile are written to
:code:`syll_seg_dirs`.


You can also extract time-warped syllables from time-warped song segments. See
:code:`ava.segmenting.template_segmentation.segment_sylls_from_warped_songs`.
This works similarly to
:code:`ava.segmenting.template_segmentation.segment_sylls_from_songs`, except it
directly writes warped spectrograms to :code:`.hdf5` files instead of writing
syllable segment to :code:`.txt` files.
