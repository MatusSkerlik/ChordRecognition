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
from abc import ABCMeta, abstractmethod
from functools import lru_cache
from typing import Any

import librosa
import numpy as np
import scipy.ndimage

from .ctx import Context
from .exceptions import IllegalStateError, IllegalArgumentError


class Strategy(object):
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
    def context(self, value: Context):
        self._context = value


class LoadStrategy(Strategy, metaclass=ABCMeta):

    @staticmethod
    @abstractmethod
    def factory(context: Context) -> 'LoadStrategy':
        pass

    @abstractmethod
    @lru_cache
    def run(self, absolute_path) -> np.ndarray:
        pass


class STFTStrategy(Strategy, metaclass=ABCMeta):

    @staticmethod
    @abstractmethod
    def factory(context: Context) -> 'STFTStrategy':
        pass

    @abstractmethod
    def run(self, y: np.ndarray) -> np.ndarray:
        pass


class ChromaStrategy(Strategy, metaclass=ABCMeta):

    @staticmethod
    @abstractmethod
    def factory(context: Context) -> 'ChromaStrategy':
        pass

    @abstractmethod
    def run(self, c: np.ndarray) -> np.ndarray:
        pass


class BeatStrategy(Strategy, metaclass=ABCMeta):

    @staticmethod
    @abstractmethod
    def factory(context: Context) -> 'BeatStrategy':
        pass

    @abstractmethod
    def run(self, y: np.ndarray, chroma: np.ndarray) -> (np.ndarray, Any):
        pass


class URILoadStrategy(LoadStrategy):

    @staticmethod
    def factory(context) -> LoadStrategy:
        return URILoadStrategy(context)

    def run(self, absolute_path) -> np.ndarray:
        y, sr = librosa.load(absolute_path, self.context.config["SAMPLING_FREQUENCY"])
        return y


class CQTStrategy(STFTStrategy):

    @staticmethod
    def factory(context: Context) -> 'STFTStrategy':
        return CQTStrategy(context)

    def run(self, y: np.ndarray) -> np.ndarray:
        return np.abs(librosa.cqt(y, sr=self.context.config["SAMPLING_FREQUENCY"]))


class FilteringChromaStrategy(ChromaStrategy):

    @staticmethod
    def factory(context: Context) -> 'ChromaStrategy':
        return FilteringChromaStrategy(context)

    def run(self, c: np.ndarray) -> np.ndarray:
        chroma = librosa.feature.chroma_cqt(
            C=c,
            hop_length=self.context.config["HOP_LENGTH"],
            fmin=self.context.config["MIN_FREQ"],
            bins_per_octave=self.context.config["BINS_PER_OCTAVE"],
            n_octaves=self.context.config["N_OCTAVES"]
        )

        chroma = np.minimum(chroma,
                            librosa.decompose.nn_filter(chroma,
                                                        aggregate=np.median,
                                                        metric='cosine'))
        return scipy.ndimage.median_filter(chroma, size=(1, 9))


class SyncBeatStrategy(BeatStrategy):

    @staticmethod
    def factory(context: Context) -> 'BeatStrategy':
        return SyncBeatStrategy(context)

    def run(self, y: np.ndarray, chroma: np.ndarray) -> (np.ndarray, Any):
        tempo, beat_f = librosa.beat.beat_track(y=y, sr=self.context.config["SAMPLING_FREQUENCY"])
        beat_f = librosa.util.fix_frames(beat_f)

        return librosa.util.sync(chroma, beat_f, aggregate=np.median), beat_f


class NoneBeatStrategy(BeatStrategy):

    @staticmethod
    def factory(context: Context) -> 'BeatStrategy':
        return NoneBeatStrategy(context)

    def run(self, y: np.ndarray, chroma: np.ndarray) -> (np.ndarray, Any):
        return y, None


class AudioProcessing(object):

    def __init__(self, load_strategy: LoadStrategy, stft_strategy: STFTStrategy, chroma_strategy: ChromaStrategy,
                 beat_strategy: BeatStrategy) -> None:
        super().__init__()

        if load_strategy is None:
            raise IllegalArgumentError
        if stft_strategy is None:
            raise IllegalArgumentError
        if chroma_strategy is None:
            raise IllegalArgumentError
        if beat_strategy is None:
            raise IllegalArgumentError

        self.load_strategy = load_strategy
        self.stft_strategy = stft_strategy
        self.chroma_strategy = chroma_strategy
        self.beat_strategy = beat_strategy

    def process(self, absolute_path) -> (np.ndarray, Any):
        y = self.load_strategy.run(absolute_path)
        c = self.stft_strategy.run(y)
        chroma = self.chroma_strategy.run(c)
        return self.beat_strategy.run(y, chroma)


class AudioProcessingBuilder(object):
    _context: Context = None
    _load_strategy: LoadStrategy = None
    _stft_strategy: STFTStrategy = None
    _chroma_strategy: ChromaStrategy = None
    _beat_strategy: BeatStrategy = None

    def __init__(self, context: Context) -> None:
        super().__init__()
        self._context = context

    def define_load_strategy(self, load_strategy_factory: LoadStrategy.factory):
        if load_strategy_factory is not None:
            self._load_strategy = load_strategy_factory(self._context)

    def define_stft_strategy(self, stft_strategy_factory: STFTStrategy.factory):
        if stft_strategy_factory is not None:
            self._stft_strategy = stft_strategy_factory(self._context)

    def define_chroma_strategy(self, chroma_strategy_factory: LoadStrategy.factory):
        if chroma_strategy_factory is not None:
            self._chroma_strategy = chroma_strategy_factory(self._context)

    def define_beat_strategy(self, beat_strategy_factory: STFTStrategy.factory):
        if beat_strategy_factory is not None:
            self._beat_strategy = beat_strategy_factory(self._context)

    def build(self) -> AudioProcessing:
        return AudioProcessing(
            self._load_strategy or self._context.config["AP_LOAD_STRATEGY_FACTORY"](self._context),
            self._stft_strategy or self._context.config["AP_STFT_STRATEGY_FACTORY"](self._context),
            self._chroma_strategy or self._context.config["AP_CHROMA_STRATEGY_FACTORY"](self._context),
            self._beat_strategy or self._context.config["AP_BEAT_STRATEGY_FACTORY"](self._context)
        )
