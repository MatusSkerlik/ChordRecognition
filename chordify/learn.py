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
from abc import abstractmethod
from collections import Sized, Iterator
from pathlib import Path
from pickle import dump, load
from typing import List, Tuple, Dict, Any

import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from .chord_recognition import PredictStrategy
from .ctx import _ctx_state, State
from .exceptions import IllegalArgumentError, IllegalStateError
from .music import Vector, IChord, Resolution, StrictResolution


class RGridSearchCV(GridSearchCV):

    def fit(self, x: Tuple[Vector], y: Tuple[IChord] = None, groups=None, **fit_params):
        _y = LabelEncoder().fit_transform(tuple(map(str, y)))
        _x = np.array(x)

        return super().fit(x, _y, groups, **fit_params)

    def r_predict(self, vectors: np.ndarray, chord_resolution: Resolution) -> Tuple[IChord]:
        _l_ch_map: Dict[str, IChord] = {str(r): r for r in chord_resolution}
        y = self.predict(vectors)
        encoder = LabelEncoder().fit(tuple(map(str, chord_resolution)))

        return tuple(map(lambda l: _l_ch_map[l], encoder.inverse_transform(y)))


class SupervisedVectors(Sized, Iterator):
    _vectors: List[Vector]
    _labels: List[IChord]

    _n: int
    _len: int

    def __init__(self) -> None:
        super().__init__()
        self._vectors = list()
        self._labels = list()
        self._n = 0
        self._len = 0

    def __iter__(self):
        self._n = 0
        return self

    def __next__(self):
        if self._n < self._len:
            self._n += 1
            return self._vectors[self._n - 1], self._labels[self._n - 1]
        else:
            raise StopIteration

    def __getitem__(self, item):
        return self._vectors[item], self._labels[item]

    def __len__(self) -> int:
        return self._len

    def append(self, vector: Vector, label: IChord):
        if vector is None or label is None:
            raise IllegalArgumentError

        self._vectors.append(vector)
        self._labels.append(label)
        self._len += 1

    def vectors(self) -> Tuple[Vector]:
        return tuple(self._vectors)

    def labels(self) -> Tuple[IChord]:
        return tuple(self._labels)


class LearnStrategy(PredictStrategy):

    @staticmethod
    def factory(*args, **kwargs):
        pass

    @abstractmethod
    def learn(self, supervised_vectors: SupervisedVectors):
        pass


class ScikitLearnStrategy(LearnStrategy):
    ch_resolution: StrictResolution

    classifier: RGridSearchCV
    __unpickling__ = False  # HACK

    def __new__(cls, *args, **kwargs) -> Any:
        if not args and not kwargs and not cls.__unpickling__:
            try:
                cls.__unpickling__ = True
                with Path(cls.__name__ + ".pickle").open(mode='rb') as fd:
                    return load(fd)
            except FileNotFoundError:
                raise IllegalStateError("Model not trained, train first !")
        elif cls.__unpickling__:
            cls.__unpickling__ = False
            return object.__new__(cls)
        else:
            o = object.__new__(cls)
            o.__init__(*args, **kwargs)
            return o

    def __init__(self, core, **kwargs) -> None:
        self.classifier = RGridSearchCV(core, kwargs, cv=5, n_jobs=-1)

    @property
    def resolution(self) -> Resolution:
        return self.ch_resolution

    def learn(self, supervised_vectors: SupervisedVectors):
        self.ch_resolution = StrictResolution(supervised_vectors.labels())
        self.classifier.fit(supervised_vectors.vectors(), supervised_vectors.labels())

        with Path(self.__class__.__name__ + ".pickle").open(mode='wb') as fd:
            dump(self, fd)

    def predict(self, vectors: np.ndarray) -> Tuple[IChord]:
        return self.classifier.r_predict(vectors.T, self.ch_resolution)


class SVCLearn(ScikitLearnStrategy):

    @staticmethod
    def factory(*args, **kwargs):
        if _ctx_state() == State.LEARNING:
            return SVCLearn.__new__(SVCLearn, svm.SVC(), C=[1, 50])
        else:
            return SVCLearn.__new__(SVCLearn)
