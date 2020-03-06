#  Copyright 2020 Matúš Škerlík
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this
#  software and associated documentation files (the "Software"), to deal in the Software
#  without restriction, including without limitation the rights to use, copy, modify, merge,
#  publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
#  to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or
#  substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
#  PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
#  LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
#  OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
#  OTHER DEALINGS IN THE SOFTWARE.
#

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.signal import find_peaks

# define zoom of graphs (sec)
from chordify.hcdf import hcdf

ZOOM = np.array([0, 5])

y, sr = librosa.load("./song.m4a")
y_harm = librosa.effects.harmonic(y=y, margin=8)

idx = tuple([slice(None), slice(*list(librosa.time_to_frames(ZOOM, sr=sr)))])

C = librosa.cqt(y, sr=sr, bins_per_octave=12 * 3, hop_length=4096)
C_harm = librosa.cqt(y_harm, sr=sr, bins_per_octave=12 * 3, hop_length=4096)

chroma_os = np.abs(librosa.feature.chroma_cqt(C=C, sr=sr, bins_per_octave=12 * 3))
chroma_os_harm = np.abs(librosa.feature.chroma_cqt(C=C_harm, sr=sr, bins_per_octave=12 * 3))

chroma_filter = np.minimum(chroma_os_harm,
                           librosa.decompose.nn_filter(chroma_os_harm,
                                                       aggregate=np.median,
                                                       metric='cosine'))
chroma_smooth = scipy.ndimage.median_filter(chroma_filter, size=(1, 8))

D = librosa.amplitude_to_db(np.abs(C_harm), ref=np.max)

""" SPECTROGRAM'S AND CHROMAGRAMS"""
plt.figure(figsize=(20, 4 * 10))

plt.subplot(10, 1, 1)
librosa.display.specshow(D[idx], y_axis='cqt_note', cmap='viridis')
plt.ylabel("Notes")
plt.xlabel("Time")
plt.title("Spectrogram")

plt.subplot(10, 1, 2)
librosa.display.specshow(chroma_os[idx], y_axis='chroma', x_axis='time', cmap='viridis')
plt.ylabel("Chroma")
plt.xlabel("Time")
plt.title("Chromagram")

plt.subplot(10, 1, 3)
librosa.display.specshow(chroma_os_harm[idx], y_axis='chroma', x_axis='time', cmap='viridis')
plt.ylabel("Chroma")
plt.xlabel("Time")
plt.title("Chromagram Harmonic")

plt.subplot(10, 1, 4)
librosa.display.specshow(chroma_filter[idx], y_axis='chroma', x_axis='time', cmap='viridis')
plt.ylabel("Chroma")
plt.xlabel("Time")
plt.title("Chromagram Harmonic, NN Filter")

plt.subplot(10, 1, 5)
librosa.display.specshow(chroma_smooth[idx], y_axis='chroma', x_axis='time', cmap='viridis')
plt.ylabel("Chroma")
plt.xlabel("Time")
plt.title("Chromagram Harmonic, NN + Median Filter")

""" HCDF graphs """
idx = slice(*list(librosa.time_to_frames(ZOOM, sr=sr)))

_hcdf_basic = hcdf(chroma_os)[idx]
_hcdf_harmonic = hcdf(chroma_os_harm)[idx]
_hcdf_filter = hcdf(chroma_filter)[idx]
_hcdf_smooth = hcdf(chroma_smooth)[idx]

# MIN MAX SCALING
_hcdf_basic = (_hcdf_basic - min(_hcdf_basic)) / (max(_hcdf_basic) - min(_hcdf_basic))
_hcdf_harmonic = (_hcdf_harmonic - min(_hcdf_harmonic)) / (max(_hcdf_harmonic) - min(_hcdf_harmonic))
_hcdf_filter = (_hcdf_filter - min(_hcdf_filter)) / (max(_hcdf_filter) - min(_hcdf_filter))
_hcdf_smooth = (_hcdf_smooth - min(_hcdf_smooth)) / (max(_hcdf_smooth) - min(_hcdf_smooth))

_peeks_basic, _ = find_peaks(_hcdf_basic, prominence=0.3)
_peeks_harmonic, _ = find_peaks(_hcdf_harmonic, prominence=0.3)
_peeks_filter, _ = find_peaks(_hcdf_filter, prominence=0.3)
_peeks_smooth, _ = find_peaks(_hcdf_smooth, prominence=0.3)

plt.subplot(10, 1, 6)
plt.plot(_hcdf_basic, '--b', label='Basic', linewidth=3)
plt.plot(_hcdf_harmonic, '--g', label='Harmonic', linewidth=3)
plt.plot(_hcdf_filter, '--r', label='Harminic, NN', linewidth=3)
plt.plot(_hcdf_smooth, '--c', label='Harmonic, NN + Median', linewidth=3)
plt.ylabel("HCDF")
plt.xlabel("Frame")
plt.ylim(ymin=0)
plt.xlim(xmin=0)
plt.legend()

plt.subplot(10, 1, 7)
plt.plot(_hcdf_basic, '--b', label='Basic', linewidth=3)
plt.plot(_peeks_basic, tuple(_hcdf_basic[i] for i in _peeks_basic), 'ok', markersize=5)
plt.ylabel("HCDF")
plt.xlabel("Frame")
plt.ylim(ymin=0)
plt.xlim(xmin=0)
plt.legend()

plt.subplot(10, 1, 8)
plt.plot(_hcdf_harmonic, '--g', label='Harmonic', linewidth=3)
plt.plot(_peeks_harmonic, tuple(_hcdf_harmonic[i] for i in _peeks_harmonic), 'ok', markersize=5)
plt.ylabel("HCDF")
plt.xlabel("Frame")
plt.ylim(ymin=0)
plt.xlim(xmin=0)
plt.legend()

plt.subplot(10, 1, 9)
plt.plot(_hcdf_filter, '--r', label='Harmonic, NN', linewidth=3)
plt.plot(_peeks_filter, tuple(_hcdf_filter[i] for i in _peeks_filter), 'ok', markersize=5)
plt.ylabel("HCDF")
plt.xlabel("Frame")
plt.ylim(ymin=0)
plt.xlim(xmin=0)
plt.legend()

plt.subplot(10, 1, 10)
plt.plot(_hcdf_smooth, '--c', label='Harmonic, NN + Median', linewidth=3)
plt.plot(_peeks_smooth, tuple(_hcdf_smooth[i] for i in _peeks_smooth), 'ok', markersize=5)
plt.ylabel("HCDF")
plt.xlabel("Frame")
plt.ylim(ymin=0)
plt.xlim(xmin=0)
plt.legend()

plt.show()
