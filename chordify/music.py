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

from abc import abstractmethod, ABCMeta
from collections import deque
from enum import Enum
from itertools import cycle
from math import log
from typing import Tuple, List, Iterable, Collection, Sized

#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this
#  software and associated documentation files (the "Software"), to deal in the Software
#  without restriction, including without limitation the rights to use, copy, modify, merge,
#  publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
#  to whom the Software is furnished to do so, subject to the following conditions:
#
#
#
import numpy as np

from chordify.exceptions import IllegalStateError, IllegalArgumentError


def _rotate_right(vector: 'Vector', r: int) -> 'Vector':
    _vector = deque(vector)
    _vector.rotate(r)
    return Vector(_vector)


def _frequency(pitch: int) -> float:
    if pitch < 0:
        raise IllegalArgumentError
    if pitch > 127:
        raise IllegalArgumentError
    return pow(2, (pitch - 69) / 12) * 440


def _harmonics(start: 'ChordKey', length=19) -> Tuple['ChordKey']:
    fq = list(reversed(tuple(i * start.frequency() for i in range(1, length + 1))))

    chk_seq: List['ChordKey'] = list()

    while len(fq) > 0:
        subject = fq.pop()

        before: ChordKey = start
        for i, k in zip(range(12 * (int(log(length, 2)) + 1)), cycle(ChordKey.__iter__())):
            if _frequency(i) < subject:
                before = k
            else:
                d_bs = subject - _frequency(i - 1 if i - 1 >= 0 else 0)
                d_cs = _frequency(i) - subject

                if d_bs < d_cs:
                    chk_seq.append(before)
                else:
                    chk_seq.append(k)
                break

    return tuple(chk_seq)


def _harm_to_vector(harms: Collection) -> 'Vector':
    _vector = ZeroVector()

    for key in harms:
        _vector[key.pos()] += 1

    return _vector + ZeroVector()


class ChordType(Enum):
    MAJOR = ""
    MINOR = ":min"
    AUGMENTED = ":aug"
    DIMINISHED = ":dim"


class ChordKey(Enum):
    C = "C"
    Cs = "C#"
    D = "D"
    Ds = "D#"
    E = "E"
    F = "F"
    Fs = "F#"
    G = "G"
    Gs = "G#"
    A = "A"
    As = "A#"
    B = "B"

    def frequency(self):
        for i, k in enumerate(self.__class__.__iter__()):
            if k == self:
                return _frequency(i)
        raise IllegalStateError

    def pos(self):
        for i, k in enumerate(self.__class__.__iter__()):
            if k == self:
                return i
        raise IllegalStateError


class Vector(Sized, Iterable):
    _vector: tuple = None

    def __init__(self, vector: Collection) -> None:
        super().__init__()

        if len(vector) != 12:
            raise IllegalArgumentError

        self._vector = tuple(vector)

    def __iter__(self):
        return iter(self._vector)

    def __len__(self):
        return 12

    def __getitem__(self, item):
        return self._vector[item]

    def __sub__(self, other):
        if isinstance(other, Vector):
            _r = tuple(a - b for a, b in zip(self._vector, other._vector))
            m = max(_r)
            return Vector(tuple(a / m for a in _r))
        raise NotImplementedError

    def __add__(self, other):
        if isinstance(other, Vector):
            _r = tuple(a + b for a, b in zip(self._vector, other._vector))
            m = max(_r)
            return Vector(tuple(a / m for a in _r))
        raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, Vector):
            return sum(a * b for a, b in zip(self._vector, other._vector))
        raise NotImplementedError

    def __repr__(self):
        return '(' + ', '.join(map(str, self._vector)) + ')'


class MutableVector(Vector):

    def __setitem__(self, key, value):
        if 0 <= key < 12:
            _vector = list(self._vector)
            _vector[key] = value
            self._vector = tuple(_vector)
        else:
            raise IllegalArgumentError


class ZeroVector(MutableVector):

    def __init__(self) -> None:
        super().__init__(tuple(0.0 for i in range(12)))


class Chord(object, metaclass=ABCMeta):
    _chord_type: ChordType = None
    _chord_key: ChordKey = None

    def __init__(self, chord_key: ChordKey, chord_type: ChordType):
        super().__init__()

        self._chord_type = chord_type
        self._chord_key = chord_key

    def __repr__(self) -> str:
        return '%s%s' % (self._chord_key.value, self._chord_type.value)

    @property
    def key(self):
        return self._chord_key

    @property
    def type(self):
        return self._chord_type

    @property
    @abstractmethod
    def vector(self) -> Vector:
        pass


class TemplateChord(Chord):
    _MAJOR: Vector = (1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0)
    _MINOR: Vector = (1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0)
    _DIMINISHED: Vector = (1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0)
    _AUGMENTED: Vector = (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)

    def shift(self):
        for i, key in enumerate(ChordKey.__iter__()):
            if key == self._chord_key:
                return i
        raise IllegalStateError

    @property
    def vector(self) -> Vector:

        _shift = self.shift()

        if self.type == ChordType.MAJOR:
            return _rotate_right(self._MAJOR, _shift)
        if self.type == ChordType.MINOR:
            return _rotate_right(self._MINOR, _shift)
        if self.type == ChordType.DIMINISHED:
            return _rotate_right(self._DIMINISHED, _shift)
        if self.type == ChordType.AUGMENTED:
            return _rotate_right(self._AUGMENTED, _shift)
        raise IllegalStateError


class TemplateChordList(object):
    MAJOR = tuple(TemplateChord(chord_key, ChordType.MAJOR) for chord_key in ChordKey)
    MINOR = tuple(TemplateChord(chord_key, ChordType.MINOR) for chord_key in ChordKey)
    AUGMENTED = tuple(TemplateChord(chord_key, ChordType.AUGMENTED) for chord_key in ChordKey)
    DIMINISHED = tuple(TemplateChord(chord_key, ChordType.DIMINISHED) for chord_key in ChordKey)
    ALL = tuple(np.array([MAJOR, MINOR, AUGMENTED, DIMINISHED]).flatten())


class HarmonicChord(TemplateChord):

    @property
    def vector(self) -> Vector:
        return super().vector + _harm_to_vector(_harmonics(self.key))


class HarmonicChordList(object):
    MAJOR = tuple(HarmonicChord(chord_key, ChordType.MAJOR) for chord_key in ChordKey)
    MINOR = tuple(HarmonicChord(chord_key, ChordType.MINOR) for chord_key in ChordKey)
    AUGMENTED = tuple(HarmonicChord(chord_key, ChordType.AUGMENTED) for chord_key in ChordKey)
    DIMINISHED = tuple(HarmonicChord(chord_key, ChordType.DIMINISHED) for chord_key in ChordKey)
    ALL = tuple(np.array([MAJOR, MINOR, AUGMENTED, DIMINISHED]).flatten())
