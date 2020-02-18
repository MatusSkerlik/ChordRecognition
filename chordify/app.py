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

from .annotation import parse_annotation, make_timeline
from .audio_processing import *
from .chord_recognition import *
from .config import ConfigAttribute, Config, ImmutableDict
from .ctx import Context, _ctx_stack, ContextAttribute, State
from .display import Plotter
from .learn import SupervisedVectors, SVCLearn
from .music import Vector


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

    def __init__(self) -> None:
        super().__init__()

    def app_context(self) -> Context:
        return _ctx_stack.top or self.with_config(None)

    def with_config(self, config):
        return Context(self, config)

    def from_path(self, absolute_path: Union[Path, str], annotation_path: Union[Path, str] = None):

        ctx = self.app_context()
        try:
            ctx.push(State.PREDICTING)

            _absolute_path = Path(absolute_path) if isinstance(absolute_path, str) else absolute_path
            _annotation_path = Path(annotation_path) if isinstance(annotation_path, str) else annotation_path

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

            return prediction
        finally:
            ctx.pop()

    def from_samples(self, paths: Iterator[Path] = None, labels: Iterator[IChord] = None,
                     iterable: Iterator = None):

        ctx = self.with_config({
            "AP_BEAT_STRATEGY_CLASS": VectorBeatStrategy,
        })

        try:
            ctx.push(State.LEARNING)

            if iterable is None and (paths is None or labels is None):
                raise IllegalArgumentError

            _supervised_vectors = SupervisedVectors()

            for absolute_path, label in (zip(paths, labels) if iterable is None else iterable):
                vector, e = self.audio_processing.process(absolute_path)
                _supervised_vectors.append(Vector(vector), label)

            self.chord_learner.learn(_supervised_vectors)
        finally:
            ctx.pop()
