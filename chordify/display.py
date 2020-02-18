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

from itertools import chain
from types import FunctionType
from typing import Iterable, List

import librosa.display
import matplotlib.pyplot as plt
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
from matplotlib.axes import Axes

from .annotation import ChordTimeline
from .ctx import Context, _chord_resolution
from .music import BasicResolution, IChord
from .utils import score


class Plotter(object):
    _n_cols: int
    _n_rows: int
    _auto: bool
    _n_func: List[FunctionType]
    _default_width: int = 6
    _default_height: int = 8

    @staticmethod
    def factory(ctx: Context):
        return Plotter(
            ctx.config["CHARTS_ROWS"],
            ctx.config["CHARTS_COLS"],
            ctx.config["CHARTS_WIDTH"],
            ctx.config["CHARTS_HEIGHT"]
        )

    def __init__(self, n_rows: int = None, n_cols: int = None, width: int = None, height: int = None) -> None:

        self._n_func = list()
        self.height = height or self._default_width
        self.width = width or self._default_width

        if n_rows is not None and n_cols is not None and n_rows > 0 and n_cols > 0:
            self._n_rows = n_rows
            self._n_cols = n_cols
            self._auto = False
        else:
            self._n_rows = 0
            self._n_cols = 1
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
        self.width += 6

        _ch_str = ["N"]
        _ch_str.extend(list(map(lambda t: str(t), reversed(_chord_resolution()))))

        def index(chord: IChord):
            try:
                return _ch_str.index(str(chord))
            except ValueError:
                return 0

        def plot(ax: Axes):
            x_p = list(chain(*((start, stop) for start, stop, chord in predicted)))
            y_p = list(chain(*((index(chord), index(chord)) for start, stop, chord in predicted)))
            x_a = list(chain(*((start, stop) for start, stop, chord in annotation)))
            y_a = list(chain(*((index(chord), index(chord)) for start, stop, chord in annotation)))

            ax.set_yticklabels(_ch_str)
            ax.set_yticks(range(0, len(tuple(BasicResolution())) + 1))
            ax.set_ylabel('Chords')
            ax.set_xlim(0, annotation.duration())
            ax.set_xlabel('Time (s)')

            ax.xaxis.set_major_locator(plt.LinearLocator())

            ax.bar(x_p, y_p, color="darkgrey", linewidth=1, ecolor=None)
            ax.plot(x_a, y_a, 'r-', linewidth=1)

            ax.set_title("Chord Prediction " + str(round(score(predicted, annotation), 2)) + "%")

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
