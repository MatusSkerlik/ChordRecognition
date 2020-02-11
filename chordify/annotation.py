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
from abc import abstractmethod
from typing import Iterable, Sized, Iterator

from pandas import read_csv

from chordify.ctx import Context
from chordify.exceptions import AnnotationParsingError
from chordify.music import TemplateChordList
from chordify.strategy import Strategy


class ChordTimeline(Iterator, Sized):
    _start: tuple = tuple()
    _stop: tuple = tuple()
    _chords: tuple = tuple()
    _len: int = 0
    _counter: int = 0

    def __init__(self) -> None:
        super().__init__()

    def __iter__(self):
        self._counter = 0
        return self

    def __next__(self):
        if self._counter < self._len:
            self._counter += 1
            return self._start[self._counter - 1], self._stop[self._counter - 1], self._chords[self._counter - 1]
        else:
            raise StopIteration

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        return self._start[item], self._stop[item], self._chords[item]

    def append(self, start: float, stop: float, chord: str):
        if start is None or stop is None or start < 0 or stop <= 0 or chord is None or len(chord) == 0:
            raise AnnotationParsingError

        _start = list(self._start)
        _start.append(float(start))
        self._start = tuple(_start)

        _stop = list(self._stop)
        _stop.append(float(stop))
        self._stop = tuple(_stop)

        _chords = list(self._chords)
        _chords.append(str(chord))
        self._chords = tuple(_chords)

        self._len += 1

    def start(self) -> Iterable:
        return iter(self._start)

    def stop(self) -> Iterable:
        return iter(self._stop)

    def chords(self) -> Iterable:
        return iter(self._chords)

    def duration(self):
        return self._stop[-1]


class ParsingStrategy(Strategy):

    @staticmethod
    @abstractmethod
    def factory(ctx: Context):
        pass

    @staticmethod
    @abstractmethod
    def accept(ext: str) -> bool:
        pass

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def parse(self, absolute_path) -> ChordTimeline:
        pass


class LabParser(ParsingStrategy):

    @staticmethod
    def factory(ctx: Context):
        return LabParser()

    @staticmethod
    def accept(ext: str) -> bool:
        return ext.endswith(".lab")

    def parse(self, absolute_path) -> ChordTimeline:
        assert self.__class__.accept(absolute_path)

        csv = read_csv(filepath_or_buffer=absolute_path, header=None, skip_blank_lines=True, delimiter=" ")

        _timeline = ChordTimeline()
        for i, row in csv.iterrows():
            start = float(row[0])
            stop = float(row[1])
            chord = normalize_chord(str(row[2]))
            _timeline.append(start, stop, chord)

        return _timeline


def get_parser(ctx: Context, annotation_path: str) -> ParsingStrategy:
    for annotation_processor in __processors__:
        if annotation_processor.accept(annotation_path):
            return annotation_processor.factory(ctx)
    raise NotImplementedError("Not supported file format.")


def parse_annotation(ctx: Context, annotation_path: str) -> ChordTimeline:
    return get_parser(ctx, annotation_path).parse(annotation_path)


def normalize_chord(chord_label: str) -> str:
    for chord in reversed(TemplateChordList.ALL):
        if str(chord_label).startswith(str(chord)):
            return str(chord)
    return chord_label


def make_timeline(beat_time: tuple, annotation: tuple) -> ChordTimeline:
    start = 0.0
    _timeline = ChordTimeline()
    for stop, chord in zip(beat_time[1:], annotation):
        _timeline.append(start, stop, str(chord))
        start = stop

    return _timeline


__processors__ = [
    LabParser
]
