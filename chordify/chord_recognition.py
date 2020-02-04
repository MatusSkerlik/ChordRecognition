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
import operator
import statistics
from abc import *

from .ctx import *
from .exceptions import *
from .music import Chord, ChordKey, ChordType


class Strategy(metaclass=ABCMeta):
    _context: Context = None

    def __init__(self, context: Context):
        super().__init__()
        self.context = context

    @property
    def context(self):
        if self._context is None:
            raise IllegalStateError
        return self._context

    @context.setter
    def context(self, value):
        if not isinstance(value, Context):
            raise IllegalArgumentError

        self._context = value


class PredictStrategy(Strategy, metaclass=ABCMeta):

    @abstractmethod
    def run(self, chroma) -> tuple:
        pass


class TemplatePredictStrategy(PredictStrategy):

    @staticmethod
    def factory(ctx: Context):
        return TemplatePredictStrategy(ctx)

    def __init__(self, context: Context, alpha=None, depth=None, bottom_threshold=None, upper_threshold=None):
        super().__init__(context)
        self.alpha = alpha or self.context.config["PREDICT_STRATEGY_ALPHA"]
        self.depth = depth or self.context.config["PREDICT_STRATEGY_DEPTH"]
        self.upper_threshold = upper_threshold or self.context.config["PREDICT_STRATEGY_UPPER_THRESHOLD"]
        self.bottom_threshold = bottom_threshold or self.context.config["PREDICT_STRATEGY_BOTTOM_THRESHOLD"]

    def run(self, chroma) -> tuple:
        chord_prg = list()
        chord_templates = tuple(
            Chord(chord_type, chord_key.value[0], self.alpha, self.depth) for chord_type in ChordType
            for chord_key in
            ChordKey
            if chord_type is not ChordType.UNKNOWN if chord_key is not ChordKey.UNKNOWN)
        for chroma_vector in chroma.T:
            template_dot_map = {}
            dot_products = list()
            for chord_template in chord_templates:
                dot_product = chord_template.dot_product(chroma_vector)
                dot_products.append(dot_product)
                template_dot_map.update({chord_template: dot_product})
            median = statistics.median(dot_products)
            # median is big if there is chroma vector witch is fulfilled
            # print('Mean: %f, Median: %f' % (statistics.mean(dot_products), median))
            if self.bottom_threshold < median < self.upper_threshold:
                chord_prg.append(max(template_dot_map.items(), key=operator.itemgetter(1))[0])
            else:
                chord_prg.append(Chord(ChordType.UNKNOWN, -1))

        return tuple(chord_prg)
