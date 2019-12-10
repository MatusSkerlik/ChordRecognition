import operator
import statistics
from copy import deepcopy
from enum import Enum
from typing import List

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy


def evaluate(original: List, predicted: List) -> (float, float):
    original_copy = deepcopy(original)
    predicted_copy = deepcopy(predicted)

    original_copy.reverse()
    predicted_copy.reverse()

    sums = 0
    total = 1

    if len(original_copy) > 0 and len(predicted_copy) > 0:

        org = original_copy.pop()
        pred = predicted_copy.pop()

        while len(original_copy) > 0 and len(predicted_copy) > 0:

            if pred[1] < org[0]:
                pred = predicted_copy.pop()
                continue
            elif pred[0] > org[1]:
                org = original_copy.pop()
                total += 1
                continue

            pred_len = pred[1] - pred[0]
            org_len = org[1] - org[0]

            if org[2].find(pred[2]) > -1:
                if pred[0] <= org[0]:
                    if pred[1] <= org[1]:
                        sums += (pred_len - (org[0] - pred[0])) / org_len
                    else:
                        sums += 1
                elif pred[0] >= org[0]:
                    if pred[1] <= org[1]:
                        sums += pred_len / org_len
                    else:
                        sums += (pred_len - (pred[1] - org[1])) / org_len
            pred = predicted_copy.pop()
    return sums / total


def c2n(to: int) -> int:
    n = 2
    while to > n:
        n *= 2
    return n


def plot_chroma(chroma, tempo, beat_time):
    plt.figure(0)
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time',
                             x_coords=beat_time)
    plt.title('Chroma (beat time)' + (' tempo: %.2f ' % tempo) + ' bps')
    plt.tight_layout()
    plt.show()


def all_chords_str() -> List[str]:
    all_chords = list()
    all_chords.append("N")
    all_chords.extend(
        ('%s%s' % (chord_key.__str__(), chord_type.__str__())) for chord_type in ChordType for chord_key in
        ChordKey if chord_type is not ChordType.UNKNOWN if chord_key is not ChordKey.UNKNOWN
    )
    return all_chords


def normalize_chord_str(chord_str: str):
    normalized = deepcopy(chord_str)[:5]

    if normalized.find("maj") > -1:
        normalized = normalized[:1]

    return normalized


def index_chord_str(chord_str: str, normalize=False):
    normalized = chord_str
    if normalize:
        normalized = normalize_chord_str(chord_str)

    chord_map = {'N': 0}
    for chord_str in all_chords_str():
        chord_map[chord_str] = len(chord_map)

    return chord_map.get(normalized) or 0


def plot_prediction(original: List, predicted: List):
    x_p = list(chord_time[0] for chord_time in predicted for i in range(2))[1:]
    y_p = list(index_chord_str(chord_time[2]) for chord_time in predicted for i in range(2))[:-1]
    x_o = list(chord_time[0] for chord_time in original for i in range(2))[1:]
    y_o = list(index_chord_str(chord_time[2], normalize=True) for chord_time in original for i in range(2))[:-1]
    plt.figure(1, figsize=(30, 15))
    plt.plot(x_o, y_o, 'g--')
    plt.plot(x_p, y_p, 'r-')
    plt.ylabel('Chords')
    plt.xlabel('Time (s)')
    plt.xlim(0, original[len(original) - 1][1])
    plt.yticks(range(0, (4 * 12) + 1), all_chords_str())
    # plt.xticks(x_o, list(str(datetime.timedelta(seconds=chord_time[0]))[2:7] for chord_time in original for i in
    # range(2))[1:])
    plt.axes().xaxis.set_major_locator(plt.LinearLocator())
    plt.show()


class Chroma:

    def __init__(self, filename: str, duration: int = None, sr: int = 22050, f_min: float = 110,
                 n_octaves: int = 4) -> None:
        super().__init__()

        self.n_octaves = n_octaves
        self.filename = filename
        self.duration = duration
        self.f_min = f_min
        self.sr = sr

    def analyse(self) -> (np.array, np.array, float):
        y, sr = librosa.load(self.filename, duration=self.duration, sr=self.sr)
        y_harm = librosa.effects.harmonic(y=y, margin=(8, 8))

        chroma_os_harm = librosa.feature.chroma_cqt(y=y_harm,
                                                    sr=sr,
                                                    bins_per_octave=12 * 3,
                                                    fmin=self.f_min,
                                                    n_octaves=self.n_octaves,
                                                    )
        chroma_nn_filter = np.minimum(chroma_os_harm,
                                      librosa.decompose.nn_filter(chroma_os_harm,
                                                                  aggregate=np.median,
                                                                  metric='cosine'))
        chroma_median = scipy.ndimage.median_filter(chroma_nn_filter, size=(1, 9))

        tempo, beat_f = librosa.beat.beat_track(y=y_harm, sr=sr, trim=True)
        beat_f = librosa.util.fix_frames(beat_f)
        beat_t = librosa.frames_to_time(beat_f, sr=sr)

        return librosa.util.sync(chroma_median, beat_f, aggregate=np.median), beat_t, tempo


class ChordType(Enum):
    MAJOR = ""
    MINOR = ":min"
    AUGMENTED = ":aug"
    DIMINISHED = ":dim"
    UNKNOWN = "unknown"

    def __str__(self):
        return self.value


class ChordKey(Enum):
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
    UNKNOWN = (-1, "N", .0)

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
    def harmonics(cls, root: 'ChordKey', s_octave: int = 2, depth: int = 8) -> List['ChordKey']:
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


class Chord:

    def __init__(self, chord_type: ChordType, shift: int, alpha: float = 0.5, depth: int = 8) -> None:
        super().__init__()
        if chord_type is ChordType.UNKNOWN:
            self.vector = np.zeros(12, dtype=float)
        else:
            self.vector = Chord._triad_energy(chord_type, shift, alpha, depth)
        self.chord_type = chord_type
        self.chord_key = ChordKey.index(shift)

    def __repr__(self) -> str:
        return '%s%s' % (self.chord_key, self.chord_type)

    @staticmethod
    def energy(chord_key: ChordKey, chord_type: ChordType, alpha: float = 0.25, depth: int = 8) -> np.array:
        return Chord._energy(chord_type, chord_key.value[0], alpha, depth)

    @staticmethod
    def _energy(chord_type: ChordType, shift: int, alpha: float = 0.5, depth: int = 8) -> np.array:
        vector = np.zeros(12, dtype=float)
        harmonics = ChordKey.harmonics(ChordKey.index(shift), depth=depth)

        vector[0] = 1 + sum(alpha ** i for i in range(1, depth) if harmonics[i] is ChordKey.index(shift))
        if chord_type is ChordType.MINOR or chord_type is ChordType.DIMINISHED:
            vector[3] = sum(alpha ** i for i in range(1, depth) if harmonics[i] is ChordKey.index((shift + 3) % 12))
        elif chord_type is ChordType.MAJOR or chord_type is ChordType.AUGMENTED:
            vector[4] = sum(alpha ** i for i in range(1, depth) if harmonics[i] is ChordKey.index((shift + 4) % 12))

        if chord_type is ChordType.MINOR or chord_type is ChordType.MAJOR:
            vector[7] = sum(alpha ** i for i in range(1, depth) if harmonics[i] is ChordKey.index((shift + 7) % 12))
        elif chord_type is ChordType.DIMINISHED:
            vector[6] = sum(alpha ** i for i in range(1, depth) if harmonics[i] is ChordKey.index((shift + 6) % 12))
        elif chord_type is ChordType.AUGMENTED:
            vector[8] = sum(alpha ** i for i in range(1, depth) if harmonics[i] is ChordKey.index((shift + 8) % 12))

        vector[10] = sum(alpha ** i for i in range(depth) if harmonics[i] is ChordKey.index((shift + 10) % 12))
        vector[11] = sum(alpha ** i for i in range(depth) if harmonics[i] is ChordKey.index((shift + 11) % 12))

        return np.roll(vector, shift)

    @staticmethod
    def triad_energy(chord_key: ChordKey, chord_type: ChordType, alpha: float = 0.5, depth: int = 8):
        return Chord._triad_energy(chord_type, chord_key.value[0], alpha, depth)

    @staticmethod
    def _triad_energy(chord_type: ChordType, shift: int, alpha: float = 0.5, depth: int = 8):
        if chord_type is ChordType.MINOR:
            return np.sum((Chord._energy(chord_type, shift, alpha, depth),
                           Chord._energy(chord_type, (shift + 3) % 12, alpha, depth),
                           Chord._energy(chord_type, (shift + 7) % 12, alpha, depth)), axis=0)
        elif chord_type is ChordType.MAJOR:
            return np.sum((Chord._energy(chord_type, shift, alpha, depth),
                           Chord._energy(chord_type, (shift + 4) % 12, alpha, depth),
                           Chord._energy(chord_type, (shift + 7) % 12, alpha, depth)), axis=0)
        elif chord_type is ChordType.AUGMENTED:
            return np.sum((Chord._energy(chord_type, shift, alpha, depth),
                           Chord._energy(chord_type, (shift + 4) % 12, alpha, depth),
                           Chord._energy(chord_type, (shift + 8) % 12, alpha, depth)), axis=0)
        elif chord_type is ChordType.DIMINISHED:
            return np.sum((Chord._energy(chord_type, shift, alpha, depth),
                           Chord._energy(chord_type, (shift + 3) % 12, alpha, depth),
                           Chord._energy(chord_type, (shift + 6) % 12, alpha, depth)), axis=0)
        else:
            raise TypeError

    def dot_product(self, vector: np.array):
        return np.dot(self.vector, vector)


class ChordMapper:

    @staticmethod
    def map(chromes: np.array, bottom_threshold: float = 0, upper_threshold: float = 2, alpha: float = 0.5,
            depth: int = 8) -> list:
        chord_prg = list()
        chord_templates = tuple(
            Chord(chord_type, chord_key.value[0], alpha, depth) for chord_type in ChordType for chord_key in ChordKey
            if chord_type is not ChordType.UNKNOWN if chord_key is not ChordKey.UNKNOWN)
        for chroma_vector in chromes.T:
            template_dot_map = {}
            dot_products = list()
            for chord_template in chord_templates:
                dot_product = chord_template.dot_product(chroma_vector)
                dot_products.append(dot_product)
                template_dot_map.update({chord_template: dot_product})
            median = statistics.median(dot_products)
            # median is big if there is chroma vector witch is fulfilled
            # print('Mean: %f, Median: %f' % (statistics.mean(dot_products), median))
            if bottom_threshold < median < upper_threshold:
                chord_prg.append(max(template_dot_map.items(), key=operator.itemgetter(1))[0])
            else:
                chord_prg.append(Chord(ChordType.UNKNOWN, -1))

        return chord_prg


chords = pd.read_csv("./ReferenceAnnotations/Beatles/Let_It_Be/chords.lab", delimiter=' ', header=None).values.tolist()
p_chords = list()

chroma = Chroma("./ReferenceAnnotations/Beatles/Let_It_Be/song.m4a",
                duration=None,
                sr=11025,
                f_min=110,
                n_octaves=5
                )
chroma_smooth_sync, beat_t, tempo = chroma.analyse()
chord_progression = ChordMapper.map(chroma_smooth_sync,
                                    bottom_threshold=0.005,
                                    upper_threshold=2.0,
                                    alpha=0.2,
                                    depth=128
                                    )

for i in range(len(beat_t) - 1):
    p_chords.append([beat_t[i], beat_t[i + 1], chord_progression[i].__repr__()])

# compare chords and predicted chords
print("Prediction: %f" % evaluate(chords, p_chords))
plot_chroma(chroma_smooth_sync, tempo, beat_t)
plot_prediction(chords, p_chords)

print(chord_progression)
print(list(cho[2] for cho in chords))
