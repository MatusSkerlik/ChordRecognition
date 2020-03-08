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
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this
#  software and associated documentation files (the "Software"), to deal in the Software
#  without restriction, including without limitation the rights to use, copy, modify, merge,
#  publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
#  to whom the Software is furnished to do so, subject to the following conditions:
#
#
#
from abc import abstractmethod
from pathlib import Path
from typing import Tuple, List, Sequence

from pandas import read_csv

from .ctx import _chord_resolution
from .exceptions import IllegalArgumentError
from .music import IChord, ChordKey
from .strategy import Strategy


class ChordTimeline(Sequence):
    _start: List[float]
    _stop: List[float]
    _chords: List[IChord]
    _len: int = 0
    _counter: int = 0

    def __init__(self):
        super().__init__()
        self._start = list()
        self._stop = list()
        self._chords = list()

    def __iter__(self):
        self._counter = 0
        return self

    def __next__(self) -> Tuple[float, float, IChord]:
        if self._counter < self._len:
            self._counter += 1
            return self._start[self._counter - 1], self._stop[self._counter - 1], self._chords[self._counter - 1]
        else:
            raise StopIteration

    def __getitem__(self, item) -> Tuple[float, float, IChord]:
        return self._start[item], self._stop[item], self._chords[item]

    def __len__(self) -> int:
        return self._len

    def append(self, start: float, stop: float, chord: IChord):
        if start is None or stop is None or start < 0 or stop <= 0 or chord is None:
            raise IllegalArgumentError

        if len(self._start) > 0 and max(self._start) > start:
            raise IllegalArgumentError
        if len(self._stop) > 0 and max(self._stop) > stop:
            raise IllegalArgumentError
        self._start.append(float(start))
        self._stop.append(float(stop))
        self._chords.append(chord)
        self._len += 1

    def start(self) -> Tuple[float, ...]:
        return tuple(self._start)

    def stop(self) -> Tuple[float, ...]:
        return tuple(self._stop)

    def chords(self) -> Tuple[IChord, ...]:
        return tuple(self._chords)

    def duration(self) -> float:
        return self._stop[-1]


class AnnotationParser(Strategy):

    @staticmethod
    @abstractmethod
    def accept(ext: str) -> bool:
        pass

    @abstractmethod
    def parse(self, absolute_path) -> ChordTimeline:
        pass


class LabParser(AnnotationParser):

    @staticmethod
    def factory(config, *args, **kwargs):
        return LabParser()

    @staticmethod
    def accept(ext: Path) -> bool:
        return ext.match("*.lab")

    def parse(self, absolute_path) -> ChordTimeline:
        assert self.__class__.accept(absolute_path)

        csv = read_csv(filepath_or_buffer=absolute_path, header=None, skip_blank_lines=True, delimiter=" ")

        _timeline = ChordTimeline()
        for i, row in csv.iterrows():
            start, stop, chord = row
            _timeline.append(float(start), float(stop), parse_chord(str(chord)))

        return _timeline


def get_parser(config, annotation_path: Path) -> AnnotationParser:
    for annotation_processor in __processors__:
        if annotation_processor.accept(annotation_path):
            return annotation_processor.factory(config)
    raise NotImplementedError("Not supported file format.")


def parse_annotation(config, annotation_path: Path) -> ChordTimeline:
    return get_parser(config, annotation_path).parse(annotation_path)


def parse_chord(chord_label: str) -> IChord:
    for i_chord in reversed(_chord_resolution()):
        if chord_label == str(i_chord.__repr__()):
            return i_chord
    return IChord(ChordKey.N, None)


def make_timeline(beat_time: Sequence[float], annotation: Sequence[IChord]) -> ChordTimeline:
    start = 0.0
    _timeline = ChordTimeline()
    for stop, chord in zip(beat_time[1:], annotation):
        _timeline.append(start, stop, chord)
        start = stop

    return _timeline


__processors__ = [
    LabParser
]
