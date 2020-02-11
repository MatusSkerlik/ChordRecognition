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

from chordify.strategy import Strategy
from .ctx import Context
from .exceptions import IllegalArgumentError


class LoadStrategy(Strategy, metaclass=ABCMeta):

    @staticmethod
    @abstractmethod
    def factory(context: Context) -> 'LoadStrategy':
        pass

    @abstractmethod
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
        return URILoadStrategy(context.config["SAMPLING_FREQUENCY"])

    def __init__(self, sampling_frequency: int):
        super().__init__()

        self._sr = sampling_frequency

    @lru_cache(maxsize=None)
    def run(self, absolute_path) -> np.ndarray:
        y, sr = librosa.load(absolute_path, self._sr)
        y_harm = librosa.effects.harmonic(y=y, margin=8)
        return y_harm


class CQTStrategy(STFTStrategy):

    @staticmethod
    def factory(context: Context) -> 'STFTStrategy':
        return CQTStrategy(context.config["SAMPLING_FREQUENCY"],
                           context.config["HOP_LENGTH"],
                           context.config["MIN_FREQ"],
                           context.config["N_BINS"],
                           context.config["BINS_PER_OCTAVE"])

    def __init__(self, sampling_frequency: int, hop_length: int, min_freq: int, n_bins: int,
                 bins_per_octave: int) -> None:
        super().__init__()

        self.bins_per_octave = bins_per_octave
        self._n_bins = n_bins
        self._min_freq = min_freq
        self._hop_length = hop_length
        self._sr = sampling_frequency

    def run(self, y: np.ndarray) -> np.ndarray:
        return np.abs(librosa.cqt(y,
                                  sr=self._sr,
                                  hop_length=self._hop_length,
                                  fmin=self._min_freq,
                                  bins_per_octave=self.bins_per_octave,
                                  n_bins=self._n_bins)
                      )


class FilteringChromaStrategy(ChromaStrategy):

    @staticmethod
    def factory(context: Context) -> 'ChromaStrategy':
        return FilteringChromaStrategy(
            context.config["HOP_LENGTH"],
            context.config["MIN_FREQ"],
            context.config["BINS_PER_OCTAVE"],
            context.config["N_OCTAVES"]
        )

    def __init__(self, hop_length: int, min_freq: int, bins_per_octave: int, n_octaves: int) -> None:
        super().__init__()
        self._hop_length = hop_length
        self._min_freq = min_freq
        self._bins_per_octave = bins_per_octave
        self._n_octaves = n_octaves

    def run(self, c: np.ndarray) -> np.ndarray:
        chroma = librosa.feature.chroma_cqt(
            C=c,
            hop_length=self._hop_length,
            fmin=self._min_freq,
            bins_per_octave=self._bins_per_octave,
            n_octaves=self._n_octaves
        )

        chroma = np.minimum(chroma,
                            librosa.decompose.nn_filter(chroma,
                                                        aggregate=np.median,
                                                        metric='cosine'))
        return scipy.ndimage.median_filter(chroma, size=(1, 9))


class SyncBeatStrategy(BeatStrategy):

    @staticmethod
    def factory(context: Context) -> 'BeatStrategy':
        return SyncBeatStrategy(context.config["SAMPLING_FREQUENCY"],
                                context.config["HOP_LENGTH"])

    def __init__(self, sampling_frequency: int, hop_length: int) -> None:
        super().__init__()

        self._hop_length = hop_length
        self._sr = sampling_frequency

    def run(self, y: np.ndarray, chroma: np.ndarray) -> (np.ndarray, Any):
        tempo, beat_f = librosa.beat.beat_track(y=y, sr=self._sr, hop_length=self._hop_length, trim=False)
        beat_f = librosa.util.fix_frames(beat_f)
        frames = librosa.util.sync(chroma, beat_f, aggregate=np.median)

        beat_t = librosa.frames_to_time(beat_f, sr=self._sr, hop_length=self._hop_length)
        return frames, beat_t


class NoBeatStrategy(BeatStrategy):

    @staticmethod
    def factory(context: Context) -> 'BeatStrategy':
        return NoBeatStrategy()

    def run(self, y: np.ndarray, chroma: np.ndarray) -> (np.ndarray, Any):
        return y, None


class AudioProcessing(object):

    @staticmethod
    def factory(ctx: Context):
        return AudioProcessing(
            ctx.config["AP_LOAD_STRATEGY_FACTORY"](ctx),
            ctx.config["AP_STFT_STRATEGY_FACTORY"](ctx),
            ctx.config["AP_CHROMA_STRATEGY_FACTORY"](ctx),
            ctx.config["AP_BEAT_STRATEGY_FACTORY"](ctx)
        )

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
