# coding: utf-8
"""
===================================
Enhanced chroma and chroma variants
===================================

This notebook demonstrates a variety of techniques for enhancing chroma features and
also, introduces chroma variants implemented in librosa.
"""

###############################################################################################
#
# Enhanced chroma
# ^^^^^^^^^^^^^^^
# Beyond the default parameter settings of librosa's chroma functions, we apply the following
# enhancements:
#
#    1. Over-sampling the frequency axis to reduce sensitivity to tuning deviations
#    2. Harmonic-percussive-residual source separation to eliminate transients.
#    3. Nearest-neighbor smoothing to eliminate passing tones and sparse noise.  This is inspired by the
#       recurrence-based smoothing technique of
#       `Cho and Bello, 2011 <http://ismir2011.ismir.net/papers/OS8-4.pdf>`_.
#    4. Local median filtering to suppress remaining discontinuities.

# Code source: Brian McFee
# License: ISC
# sphinx_gallery_thumbnail_number = 6

from __future__ import print_function

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy


class Chroma:

    def __init__(self, filename, duration=None) -> None:
        super().__init__()
        self.y, self.sr = librosa.load(filename, duration=duration)
        self.y = librosa.effects.harmonic(y=self.y, margin=8)
        self.chroma_smooth = None

    def analyse(self) -> np.array:
        if self.chroma_smooth is not None:
            return self.chroma_smooth

        ########################################################
        # That cleaned up some rough edges, but we can do better
        # by isolating the harmonic component.
        # We'll use a large margin for separating harmonics from percussives
        chroma_os_harm = librosa.feature.chroma_cqt(y=self.y, sr=self.sr, bins_per_octave=12 * 3)

        ###########################################
        # There's still some noise in there though.
        # We can clean it up using non-local filtering.
        # This effectively removes any sparse additive noise from the features.
        chroma_filter = np.minimum(chroma_os_harm,
                                   librosa.decompose.nn_filter(chroma_os_harm,
                                                               aggregate=np.median,
                                                               metric='cosine'))

        ###########################################################
        # Local discontinuities and transients can be suppressed by
        # using a horizontal median filter.
        self.chroma_smooth = scipy.ndimage.median_filter(chroma_filter, size=(1, 9))

        return self.chroma_smooth

    def analyse_beat_sync(self) -> (float, float, np.array):
        chroma_smooth = self.analyse();
        tempo, beat_f = librosa.beat.beat_track(y=self.y, sr=self.sr, trim=False)
        beat_f = librosa.util.fix_frames(beat_f)
        beat_t = librosa.frames_to_time(beat_f, sr=self.sr)
        chroma_smooth_sync = librosa.util.sync(chroma_smooth, beat_f, aggregate=np.median)

        return beat_t, tempo, chroma_smooth_sync


###########################################################
# Plot 2 graphs
# First graph is without beat synchronization

chroma = Chroma("./PianoChords.wav", duration=120)

chroma_smooth = chroma.analyse()
beat_t, tempo, chroma_smooth_sync = chroma.analyse_beat_sync()

plt.figure()

ax1 = plt.subplot(2, 1, 1)
librosa.display.specshow(chroma_smooth, y_axis='chroma', x_axis='time')
plt.title('Processed Chroma (linear time)')

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
librosa.display.specshow(chroma_smooth_sync, y_axis='chroma', x_axis='time',
                         x_coords=beat_t)
plt.title('Processed Chroma (beat time)' + (' tempo: %.2f ' % tempo) + ' bps')
plt.tight_layout()
plt.show()
