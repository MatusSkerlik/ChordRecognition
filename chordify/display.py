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
from .ctx import _chord_resolution
from .logger import log
from .music import BasicResolution, IChord
from .utils import score


class Plotter(object):
    _u_cols: int
    _u_rows: int
    _u_width: int
    _u_height: int
    _n_cols: int
    _n_rows: int
    _auto: bool
    _n_func: List[FunctionType]
    _default_width: int = 6
    _default_height: int = 8

    @classmethod
    def factory(cls, config, *args, **kwargs):
        log(cls, "Init")
        return Plotter(
            config["CHARTS_ROWS"],
            config["CHARTS_COLS"],
            config["CHARTS_WIDTH"],
            config["CHARTS_HEIGHT"]
        )

    def __init__(self, n_rows: int = None, n_cols: int = None, width: int = None, height: int = None) -> None:

        self._u_rows = n_rows
        self._u_cols = n_cols
        self._u_width = width
        self._u_height = height

        self._reset()

    def _reset(self):
        self._n_func = list()
        self.height = self._u_width or self._default_width
        self.width = self._u_height or self._default_width

        if self._u_rows is not None and self._u_cols is not None and self._u_rows > 0 and self._u_cols > 0:
            self._n_rows = self._u_rows
            self._n_cols = self._u_cols
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
            librosa.display.specshow(chroma,
                                     y_axis='chroma',
                                     x_axis='time',
                                     x_coords=beat_time,
                                     ax=ax)
            ax.set_title("Chromagram")

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

        def plot_prediction(ax: Axes):
            x = list(chain(*((start, stop) for start, stop, chord in predicted)))
            y = list(chain(*((index(chord), index(chord)) for start, stop, chord in predicted)))

            ax.set_yticklabels(_ch_str)
            ax.set_yticks(range(0, len(tuple(BasicResolution())) + 1))
            ax.set_ylabel('Chords')
            ax.set_xlim(0, predicted.duration())
            ax.set_xlabel('Time (s)')

            ax.xaxis.set_major_locator(plt.LinearLocator())
            ax.xaxis.set_major_formatter(librosa.display.TimeFormatter())

            ax.fill_between(x, y, 'r-', color="darkgrey", linewidth=1)
            for vl in predicted.stop():
                ax.axvline(vl, color="green")

        def plot_annotation(ax: Axes):
            x = list(chain(*((start, stop) for start, stop, chord in annotation)))
            y = list(chain(*((index(chord), index(chord)) for start, stop, chord in annotation)))

            ax.set_yticklabels(_ch_str)
            ax.set_yticks(range(0, len(tuple(BasicResolution())) + 1))
            ax.set_ylabel('Chords')
            ax.set_xlim(0, annotation.duration())
            ax.set_xlabel('Time (s)')

            ax.xaxis.set_major_locator(plt.LinearLocator())
            ax.xaxis.set_major_formatter(librosa.display.TimeFormatter())

            ax.plot(x, y, 'r-', linewidth=1)
            for vl in annotation.stop():
                ax.axvline(vl, color="red")

            ax.set_title("Chord Prediction " + str(round(score(predicted, annotation), 2)) + "%")

        def plot(ax: Axes):
            if annotation is not None:
                plot_annotation(ax)
            plot_prediction(ax)

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
        self._reset()
