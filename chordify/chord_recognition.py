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

from abc import *

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

from chordify.ctx import Context
from chordify.music import TemplateChordList, HarmonicChordList
from chordify.strategy import Strategy


class PredictStrategy(Strategy, metaclass=ABCMeta):

    @staticmethod
    @abstractmethod
    def factory(ctx: Context):
        pass

    @abstractmethod
    def predict(self, chroma) -> tuple:
        pass


class TemplatePredictStrategy(PredictStrategy):

    @staticmethod
    def factory(ctx: Context):
        return TemplatePredictStrategy()

    def __init__(self):
        super().__init__()

    def predict(self, chroma: np.ndarray, filter_func=lambda d: d) -> tuple:
        _chord_prg = list()
        _ch_v = np.array(list(map(lambda c: c.vector, TemplateChordList.ALL)))

        for ch in chroma.T:
            _dots = filter_func(_ch_v.dot(ch))
            _chord_prg.append(TemplateChordList.ALL[np.argmax(_dots)])

        return tuple(_chord_prg)


class HarmonicPredictStrategy(PredictStrategy):

    @staticmethod
    def factory(ctx: Context):
        return HarmonicPredictStrategy()

    def __init__(self):
        super().__init__()

    def predict(self, chroma: np.ndarray, filter_func=lambda d: d) -> tuple:
        _chord_prg = list()
        _ch_v = np.array(list(map(lambda c: c.vector, HarmonicChordList.ALL)))

        for ch in chroma.T:
            _dots = filter_func(_ch_v.dot(ch))
            _chord_prg.append(HarmonicChordList.ALL[np.argmax(_dots)])

        return tuple(_chord_prg)
