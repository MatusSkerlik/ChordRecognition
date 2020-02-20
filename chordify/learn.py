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
from pickle import dump, load
from typing import List, Tuple, Dict, ContextManager

import numpy as np
from sklearn import svm
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from chordify.strategy import Strategy
from .chord_recognition import PredictStrategy
from .exceptions import IllegalArgumentError
from .logger import log
from .music import Vector, IChord, Resolution, StrictResolution
from .state import AppState


class RGridSearchCV(GridSearchCV):
    _encoder: LabelEncoder

    def __init__(self, estimator, param_grid, scoring=None, n_jobs=None, iid='deprecated', refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs', error_score=np.nan, return_train_score=False):
        super().__init__(estimator, param_grid, scoring, n_jobs, iid, refit, cv, verbose, pre_dispatch, error_score,
                         return_train_score)
        self._encoder = LabelEncoder()

    def fit(self, x: Tuple[Vector], y: Tuple[IChord] = None, groups=None, **fit_params):
        _y = self._encoder.fit_transform(tuple(map(str, y)))
        return super().fit(np.array(x), _y, groups, **fit_params)

    def r_predict(self, vectors: np.ndarray, chord_resolution: Resolution) -> Tuple[IChord]:
        _l_ch_map: Dict[str, IChord] = {str(r): r for r in chord_resolution}
        _y = self.predict(vectors)
        return tuple(map(lambda l: _l_ch_map[l], self._encoder.inverse_transform(_y)))


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

    def __init__(self, estimator: BaseEstimator, file: ContextManager, **kwargs) -> None:
        self.classifier = RGridSearchCV(estimator, kwargs, cv=5, n_jobs=-1)
        self.output_file = file

    @property
    def resolution(self) -> Resolution:
        return self.ch_resolution

    def learn(self, supervised_vectors: SupervisedVectors):
        log(self.__class__, "Learning...")
        self.ch_resolution = StrictResolution(supervised_vectors.labels())
        self.classifier.fit(supervised_vectors.vectors(), supervised_vectors.labels())
        log(self.__class__, "Learning done...")

        output_file = self.output_file
        del self.output_file

        with output_file as f:
            log(self.__class__, "Dumping model = " + str(f))
            dump(self, f)

    def predict(self, vectors: np.ndarray) -> Tuple[IChord]:
        log(self.__class__, "Predicting...")
        return self.classifier.r_predict(vectors.T, self.ch_resolution)


class SVCLearn(Strategy):

    def __new__(cls, estimator: BaseEstimator, state: AppState, file: ContextManager, *args,
                **kwargs) -> ScikitLearnStrategy:
        if state == AppState.LEARNING:
            return ScikitLearnStrategy(estimator, file, **kwargs)
        else:
            with file as f:
                return load(f)

    @classmethod
    def factory(cls, config: dict, state: AppState, file: ContextManager, *args, **kwargs) -> ScikitLearnStrategy:
        log(cls, "Init")
        return SVCLearn(svm.SVC(), state, file, C=[1, 50])
