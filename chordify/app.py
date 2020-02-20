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
from typing import Iterator, Union

from joblib import Parallel, delayed

from chordify.exceptions import IllegalConfigError
from .annotation import parse_annotation, make_timeline
from .audio_processing import *
from .chord_recognition import *
from .config import Config, ImmutableDict
from .ctx import Context, _ctx_stack, ContextAttribute, ConfigAttribute
from .display import Plotter
from .learn import SupervisedVectors, SVCLearn
from .music import Vector
from .state import AppState


def check_config(config):
    if "AUDIO_PROCESSING_CLASS" in config and not issubclass(config["AUDIO_PROCESSING_CLASS"], AudioProcessing):
        raise IllegalConfigError
    if "AP_LOAD_STRATEGY_CLASS" in config and not issubclass(config["AP_LOAD_STRATEGY_CLASS"], LoadStrategy):
        raise IllegalConfigError
    if "AP_STFT_STRATEGY_CLASS" in config and not issubclass(config["AP_STFT_STRATEGY_CLASS"], STFTStrategy):
        raise IllegalConfigError
    if "AP_CHROMA_STRATEGY_CLASS" in config and not issubclass(config["AP_CHROMA_STRATEGY_CLASS"],
                                                               ChromaStrategy):
        raise IllegalConfigError
    if "AP_BEAT_STRATEGY_CLASS" in config and not issubclass(config["AP_BEAT_STRATEGY_CLASS"], BeatStrategy):
        raise IllegalConfigError
    if "CHORD_RECOGNITION_CLASS" in config and not issubclass(config["CHORD_RECOGNITION_CLASS"],
                                                              Strategy):
        raise IllegalConfigError
    if "CHORD_LEARNING_CLASS" in config and not issubclass(config["CHORD_LEARNING_CLASS"], Strategy):
        raise IllegalConfigError


class Chordify(object):
    default_config: Config = ImmutableDict({
        "DEBUG": False,

        "PLOT_CLASS": Plotter,
        "CHARTS": True,
        "CHARTS_HEIGHT": None,
        "CHARTS_WIDTH": 6,
        "CHARTS_ROWS": None,
        "CHARTS_COLS": None,
        "CHART_CHROMAGRAM": True,
        "CHART_PREDICTION": True,

        "AUDIO_PROCESSING_CLASS": AudioProcessing,
        "AP_LOAD_STRATEGY_CLASS": PathLoadStrategy,
        "AP_STFT_STRATEGY_CLASS": CQTStrategy,
        "AP_CHROMA_STRATEGY_CLASS": FilteringChromaStrategy,
        "AP_BEAT_STRATEGY_CLASS": SyncBeatStrategy,

        "SAMPLING_FREQUENCY": 44100,
        "N_OCTAVES": 84 // 12,
        "N_BINS": 84,
        "BINS_PER_OCTAVE": 12 * 3,
        "MIN_FREQ": 440,
        "HOP_LENGTH": 4096,

        "CHORD_RECOGNITION_CLASS": TemplatePredictStrategy,
        "CHORD_LEARNING_CLASS": SVCLearn,
    })

    debug = ConfigAttribute("DEBUG")
    charts = ConfigAttribute("CHARTS")
    chart_chromagram = ConfigAttribute("CHART_CHROMAGRAM")
    chart_prediction = ConfigAttribute("CHART_PREDICTION")

    audio_processing = ContextAttribute("audio_processing")
    chord_recognition = ContextAttribute("chord_recognition")
    chord_learner = ContextAttribute("chord_recognition")
    plotter = ContextAttribute("plotter")
    config = ContextAttribute("config")

    def app_context(self, config=None) -> Context:
        if _ctx_stack.top is not None:
            return _ctx_stack.top
        else:
            return self.with_config(config)

    def with_config(self, config=None):
        if config is None:
            config = dict()
        check_config(config)
        return Context(self, config)

    def from_path(self, absolute_path: Union[Path, str], annotation_path: Union[Path, str] = None):
        log(self.__class__, "Start predicting")
        ctx = self.app_context()
        try:
            ctx.push()
            ctx.transition_to(AppState.PREDICTING)

            _absolute_path = Path(absolute_path) if isinstance(absolute_path, str) else absolute_path
            _annotation_path = Path(annotation_path) if isinstance(annotation_path, str) else annotation_path

            log(self.__class__, "File = " + str(_absolute_path.resolve()))
            if annotation_path is not None:
                log(self.__class__, "Annotation = " + str(_annotation_path.resolve()))

            chroma_sync, beat_t = self.audio_processing.process(_absolute_path)
            prediction = self.chord_recognition.predict(chroma_sync)

            if self.charts:
                if self.chart_chromagram:
                    self.plotter.chromagram(chroma_sync, beat_t)
                if self.chart_prediction and annotation_path is not None:
                    annotation_timeline = parse_annotation(ctx, _annotation_path)
                    prediction_timeline = make_timeline(beat_t, prediction)
                    self.plotter.prediction(prediction_timeline, annotation_timeline)
            self.plotter.show()

            log(self.__class__, "Result: " + str(prediction))
            return prediction
        finally:
            log(self.__class__, "Stop predicting")
            ctx.pop()

    def from_samples(self, paths: Iterator[Path] = None, labels: Iterator[IChord] = None,
                     iterable: Iterator = None):
        log(self.__class__, "Start learning")
        ctx = self.with_config({"AP_BEAT_STRATEGY_CLASS": VectorBeatStrategy})
        try:
            ctx.push()
            ctx.transition_to(AppState.LEARNING)

            if iterable is None and (paths is None or labels is None):
                raise IllegalArgumentError

            _iter = tuple(zip(paths, labels) if iterable is None else iterable)
            _supervised_vectors = SupervisedVectors()

            with Parallel(n_jobs=-3) as parallel:
                out = parallel(delayed(self.audio_processing.process)(path) for path, label in _iter)
                for vector_beat, path_label in zip(out, _iter):
                    _supervised_vectors.append(Vector(vector_beat[0]), path_label[1])

            self.chord_learner.learn(_supervised_vectors)
        finally:
            log(self.__class__, "Stop learning")
            ctx.pop()
