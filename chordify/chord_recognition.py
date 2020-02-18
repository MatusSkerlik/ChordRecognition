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
from abc import *
from typing import Sequence

import numpy as np

from .ctx import Context
from .music import TemplateChords, HarmonicChords, Resolution, BasicResolution, IChord
from .strategy import Strategy


class PredictStrategy(Strategy):

    @abstractmethod
    def predict(self, chroma) -> Sequence[IChord]:
        pass

    @property
    @abstractmethod
    def resolution(self) -> Resolution:
        pass


class TemplatePredictStrategy(PredictStrategy):

    @staticmethod
    def factory(ctx: Context, **kwargs):
        return TemplatePredictStrategy()

    @property
    def resolution(self) -> Resolution:
        return BasicResolution()

    def predict(self, chroma: np.ndarray, filter_func=lambda d: d) -> tuple:
        _chord_prg = list()
        _ch_v = np.array(list(map(lambda c: c.vector, TemplateChords.ALL)))

        for ch in chroma.T:
            _dots = filter_func(_ch_v.dot(ch))
            _chord_prg.append(TemplateChords.ALL[np.argmax(_dots)])

        return tuple(_chord_prg)


class HarmonicPredictStrategy(PredictStrategy):

    @staticmethod
    def factory(ctx: Context, **kwargs):
        return HarmonicPredictStrategy()

    @property
    def resolution(self) -> Resolution:
        return BasicResolution()

    def predict(self, chroma: np.ndarray, filter_func=lambda d: d) -> tuple:
        _chord_prg = list()
        _ch_v = np.array(list(map(lambda c: c.vector, HarmonicChords.ALL)))

        for ch in chroma.T:
            _dots = filter_func(_ch_v.dot(ch))
            _chord_prg.append(HarmonicChords.ALL[np.argmax(_dots)])

        return tuple(_chord_prg)
