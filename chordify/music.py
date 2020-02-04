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

#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this
#  software and associated documentation files (the "Software"), to deal in the Software
#  without restriction, including without limitation the rights to use, copy, modify, merge,
#  publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
#  to whom the Software is furnished to do so, subject to the following conditions:
#
#
#

from enum import Enum
from typing import List

import numpy as np


class ChordType(Enum):
    MAJOR = ""
    MINOR = ":min"
    AUGMENTED = ":aug"
    DIMINISHED = ":dim"
    UNKNOWN = "unknown"

    def __str__(self):
        return self.value


class ChordKey(Enum):
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
    def index(cls, index):
        for enum in ChordKey:
            if enum.value[0] == index:
                return enum
        raise ValueError

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
        self.chord_type = chord_type
        self.chord_key = ChordKey.index(shift)
        self.shift = shift
        self.depth = depth
        self.alpha = alpha

        if chord_type is ChordType.UNKNOWN:
            self.vector = np.zeros(12, dtype=float)
        else:
            self.vector = self.triad_energy()

    def __repr__(self) -> str:
        return '%s%s' % (self.chord_key, self.chord_type)

    @staticmethod
    def energy(chord_type: ChordType, shift: int, alpha: float = 0.5, depth: int = 8) -> np.array:
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

    def triad_energy(self):

        chord_type = self.chord_type
        shift = self.shift
        alpha = self.alpha
        depth = self.depth

        if chord_type is ChordType.MINOR:
            return np.sum((Chord.energy(chord_type, shift, alpha, depth),
                           Chord.energy(chord_type, (shift + 3) % 12, alpha, depth),
                           Chord.energy(chord_type, (shift + 7) % 12, alpha, depth)), axis=0)
        elif chord_type is ChordType.MAJOR:
            return np.sum((Chord.energy(chord_type, shift, alpha, depth),
                           Chord.energy(chord_type, (shift + 4) % 12, alpha, depth),
                           Chord.energy(chord_type, (shift + 7) % 12, alpha, depth)), axis=0)
        elif chord_type is ChordType.AUGMENTED:
            return np.sum((Chord.energy(chord_type, shift, alpha, depth),
                           Chord.energy(chord_type, (shift + 4) % 12, alpha, depth),
                           Chord.energy(chord_type, (shift + 8) % 12, alpha, depth)), axis=0)
        elif chord_type is ChordType.DIMINISHED:
            return np.sum((Chord.energy(chord_type, shift, alpha, depth),
                           Chord.energy(chord_type, (shift + 3) % 12, alpha, depth),
                           Chord.energy(chord_type, (shift + 6) % 12, alpha, depth)), axis=0)
        else:
            raise TypeError

    def dot_product(self, vector: np.array):
        return np.dot(self.vector, vector)
