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

from os import listdir
from pathlib import Path
from typing import Tuple

from .annotation import ChordTimeline
from .music import IChord, ChordKey


class SupervisedDirectoryAdapter(object):
    _chord: IChord
    _files: Tuple[str]
    _len: int
    _n: int

    def __init__(self, directory: Path, chord: IChord) -> None:
        super().__init__()
        self._chord = chord
        self._directory = directory
        self._files = listdir(directory)
        self._len = len(self._files)
        self._n = 0

    def __iter__(self):
        self._n = 0
        return self

    def __next__(self) -> Tuple[Path, IChord]:
        if self._n < self._len:
            self._n += 1
            return self._directory / Path(self._files[self._n - 1]), self._chord
        else:
            raise StopIteration

    def __len__(self):
        return self._len


def score(prediction: ChordTimeline, annotation: ChordTimeline) -> float:
    _correct = 0
    _skip = 0
    _i_annotation = 0
    _i_prediction = 0
    while _i_annotation < len(annotation) and _i_prediction < len(prediction):
        start, stop, chord = annotation[_i_annotation]
        p_start, p_stop, p_chord = prediction[_i_prediction]

        if chord == IChord(ChordKey.N, None):
            _i_annotation += 1
            _skip += 1
            continue

        if p_stop < start:
            _i_prediction += 1
            continue
        elif stop < p_start:
            _i_annotation += 1
            continue

        if p_start <= start and p_stop < stop and chord == p_chord:
            _correct += (p_stop - start) / (stop - start)
        elif p_start >= start and p_stop <= stop and chord == p_chord:
            _correct += (p_stop - p_start) / (stop - start)
        elif p_start > start and p_stop >= stop and chord == p_chord:
            _correct += (stop - p_start) / (stop - start)
        elif p_start <= start and p_stop >= stop and chord == p_chord:
            _correct += (stop - start) / (p_stop - p_start)
        _i_prediction += 1
    return _correct / (len(annotation) - _skip)
