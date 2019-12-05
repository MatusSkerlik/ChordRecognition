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

import operator
import statistics
from enum import Enum
from typing import List

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy


def plot(ch, t, bt):
    librosa.display.specshow(ch, y_axis='chroma', x_axis='time',
                             x_coords=bt)
    plt.title('Chroma (beat time)' + (' tempo: %.2f ' % t) + ' bps')
    plt.tight_layout()
    plt.show()


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

    def analyse_beat_sync(self) -> (np.array, float, np.array, np.array):
        chroma_smooth = self.analyse()
        tempo, beat_f = librosa.beat.beat_track(y=self.y, sr=self.sr, trim=False)
        beat_f = librosa.util.fix_frames(beat_f)
        beat_t = librosa.frames_to_time(beat_f, sr=self.sr)
        chroma_smooth_sync = librosa.util.sync(chroma_smooth, beat_f, aggregate=np.median)

        return chroma_smooth_sync, tempo, beat_t, beat_f


class ChordType(Enum):
    MAJOR = "major"
    MINOR = "minor"
    UNKNOWN = "unknown"

    def __str__(self):
        return self.value


class Chord(Enum):
    # TODO update frequencies to 4 float points
    C = (0, "C", 16.35159883)
    Cs = (1, "C#", 17.32391444)
    D = (2, "D", 18.35404799)
    Ds = (3, "D#", 19.44543648)
    E = (4, "E", 20.60172231)
    F = (5, "F", 21.82676446)
    Fs = (6, "F#", 23.12465142)
    G = (7, "G", 24.49971475)
    Gs = (8, "G#", 25.9565436)
    A = (9, "A", 27.50)
    As = (10, "A#", 29.13523509)
    B = (11, "B", 30.86770633)
    UNKNOWN = (-1, "?", .0)

    def __str__(self):
        return self.value[1]

    @classmethod
    def index(cls, index: int):
        for _chord in cls:
            if _chord.value[0] == index:
                return _chord
        raise ValueError("%d is not a valid index for %s" % (index, cls.__name__))

    @classmethod
    def frequency(cls, index: int, octave: int) -> float:
        return cls.index(index).value[2] * (2 ** octave)

    @classmethod
    def harmonics(cls, root: 'Chord', s_octave: int = 2, depth: int = 8) -> List['Chord']:
        bf = cls.frequency(root.value[0], s_octave)
        frequencies = list(bf * i for i in range(1, depth + 1))

        all_fq = list()
        max_frequency = max(frequencies)
        for octave in range(s_octave, 8):
            for note in range(0, 12):
                frequency = cls.frequency(note, octave)
                if frequency > max_frequency:
                    all_fq.append(frequency)
                    break
                else:
                    all_fq.append(frequency)
            else:
                continue  # only executed if the inner loop did NOT break
            break  # only executed if the inner loop DID break

        harms = list()
        for frequency in frequencies:
            fq_diff = list()
            for fq in all_fq:
                fq_diff.append(fq - frequency)
            minimal = min(fq_diff, key=abs)
            min_index = fq_diff.index(minimal)
            harms.append(cls.index(min_index % 12))

        return harms


class ChordTemplate:

    def __init__(self, chord_type: ChordType, shift: int) -> None:
        super().__init__()
        if chord_type is ChordType.UNKNOWN:
            self.vector = np.zeros(12, dtype=float)
        else:
            self.vector = ChordTemplate.chord_triad(chord_type, shift)
        self.chord_type = chord_type
        self.chord_key = Chord.index(shift)

    def __repr__(self) -> str:
        return '%s %s' % (self.chord_key, self.chord_type)

    @staticmethod
    def chord(chord_type: ChordType, shift: int, alpha: float = 0.25, depth: int = 12) -> np.array:
        vector = np.zeros(12, dtype=float)
        harmonics = Chord.harmonics(Chord.index(shift), depth=depth)

        vector[0] = 1 + sum(alpha ** i for i in range(1, depth) if harmonics[i] is Chord.index(shift))
        if chord_type is ChordType.MINOR:
            vector[3] = sum(alpha ** i for i in range(1, depth) if harmonics[i] is Chord.index((shift + 3) % 12))
        elif chord_type is ChordType.MAJOR:
            vector[4] = sum(alpha ** i for i in range(1, depth) if harmonics[i] is Chord.index((shift + 4) % 12))
        else:
            raise TypeError
        vector[7] = sum(alpha ** i for i in range(1, depth) if harmonics[i] is Chord.index((shift + 7) % 12))
        # vector[10] = sum(alpha ** i for i in range(depth) if harmonics[i] is Chord.index((shift + 10) % 12))

        return np.roll(vector, shift)

    @staticmethod
    def chord_triad(chord_type: ChordType, shift: int, alpha: float = 0.25, depth: int = 12):
        if chord_type is ChordType.MINOR:
            return np.sum((ChordTemplate.chord(chord_type, shift, alpha, depth),
                           ChordTemplate.chord(chord_type, (shift + 3) % 12, alpha, depth),
                           ChordTemplate.chord(chord_type, (shift + 7) % 12, alpha, depth)), axis=0)
        elif chord_type is ChordType.MAJOR:
            return np.sum((ChordTemplate.chord(chord_type, shift, alpha, depth),
                           ChordTemplate.chord(chord_type, (shift + 4) % 12, alpha, depth),
                           ChordTemplate.chord(chord_type, (shift + 7) % 12, alpha, depth)), axis=0)
        else:
            raise TypeError

    def dot_product(self, vector: np.array):
        return np.dot(self.vector, vector)


chord_templates = tuple(ChordTemplate(chord_type, shift) for chord_type in ChordType for shift in range(0, 12) if
                        chord_type is not ChordType.UNKNOWN)
chroma = Chroma("./PianoChords.wav", 30)
chroma_smooth_sync, tempo, beat_t, beat_f = chroma.analyse_beat_sync()

plot(chroma_smooth_sync, tempo, beat_t)

chord_progression = list()
for chroma_vector in chroma_smooth_sync.T:
    template_dot_map = {}
    dot_products = list()
    for chord_template in chord_templates:
        dot_product = chord_template.dot_product(chroma_vector)
        dot_products.append(dot_product)
        template_dot_map.update({chord_template: dot_product})
    median = statistics.median(dot_products)
    # median is big if there is chroma vector witch is fulfilled, average is about 0.05
    print(template_dot_map)
    print('Mean: %f, Median: %f' % (statistics.mean(dot_products), median))
    if median < 5:
        chord_progression.append(max(template_dot_map.items(), key=operator.itemgetter(1))[0])
    else:
        chord_progression.append(ChordTemplate(ChordType.UNKNOWN, -1))

print(chord_progression)
