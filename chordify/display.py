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

from types import FunctionType
from typing import Iterable

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from chordify.annotation import ChordTimeline
from chordify.music import TemplateChordList


class ChartBuilder(object):
    _n_cols: int = 1
    _n_rows: int = 0
    _auto = False
    _n_func = list()
    _default_width = 6
    _default_height = 8

    def __init__(self, n_rows: int = None, n_cols: int = None, width: int = None, height: int = None) -> None:

        self.height = height or self._default_width
        self.width = width or self._default_width

        if n_rows is not None and n_cols is not None and n_rows > 0 and n_cols > 0:
            self._n_rows = n_rows
            self._n_cols = n_cols
        else:
            self._auto = True

    def _add(self, func):
        assert isinstance(func, FunctionType)
        if self._auto:
            self.height += 2
            self._n_rows += 1
        self._n_func.append(func)

    def chromagram(self, chroma: np.ndarray, beat_time: np.ndarray):
        def plot(ax: Axes):
            ax.set_title("Chromagram")
            librosa.display.specshow(chroma,
                                     y_axis='chroma',
                                     x_axis='time',
                                     x_coords=beat_time,
                                     ax=ax)

        self._add(plot)
        return self

    def prediction(self, predicted: ChordTimeline, annotation: ChordTimeline):
        _ch_str = ["N"]
        _ch_str.extend(list(map(lambda t: str(t), reversed(TemplateChordList.ALL))))

        def index(chord: str):
            return _ch_str.index(chord)

        def plot(ax: Axes):
            x_p = list(np.array(list((start, stop) for start, stop, chord in predicted)).flatten())
            y_p = list(np.array(list((index(chord), index(chord)) for start, stop, chord in predicted)).flatten())
            x_o = list(np.array(list((start, stop) for start, stop, chord in annotation)).flatten())
            y_o = list(np.array(list((index(chord), index(chord)) for start, stop, chord in annotation)).flatten())

            ax.set_yticklabels(_ch_str)
            ax.set_yticks(range(0, len(TemplateChordList.ALL)))
            ax.set_ylabel('Chords')
            ax.set_xlim(0, annotation.duration())
            ax.set_xlabel('Time (s)')

            ax.xaxis.set_major_locator(plt.LinearLocator())

            ax.bar(x_p, y_p, color="darkgrey")
            ax.plot(x_o, y_o, 'r-')

            ax.set_title("Chord Prediction")

        self._add(plot)
        return self

    def show(self):
        fig, axes = plt.subplots(nrows=self._n_rows, ncols=self._n_cols)
        fig.set_size_inches(w=self.width, h=self.height)
        self._n_func.reverse()

        if not isinstance(axes, Iterable):
            _axes = list()
            _axes.append(axes)
            axes = _axes

        if self._auto:

            for ax in axes:
                self._n_func.pop()(ax)

        else:
            n_rows = 0
            n_cols = 0

            while n_rows < self._n_rows:
                while n_cols < self._n_cols:
                    self._n_func.pop()(axes[n_rows, n_cols])
                    n_cols += 1
                n_rows += 1
        fig.show()
