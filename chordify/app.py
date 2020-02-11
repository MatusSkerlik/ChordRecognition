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
from chordify.display import ChartBuilder
from .annotation import parse_annotation, make_timeline
from .audio_processing import *
from .chord_recognition import *
from .config import ConfigAttribute, Config
from .ctx import Context


class Chordify(Context):
    debug = ConfigAttribute("DEBUG")
    charts = ConfigAttribute("CHARTS")
    charts_rows = ConfigAttribute("CHARTS_ROWS")
    charts_cols = ConfigAttribute("CHARTS_COLS")
    charts_width = ConfigAttribute("CHARTS_WIDTH")
    charts_height = ConfigAttribute("CHARTS_HEIGHT")
    chart_chromagram = ConfigAttribute("CHART_CHROMAGRAM")
    chart_prediction = ConfigAttribute("CHART_PREDICTION")

    default_config: Config = {
        "DEBUG": False,

        "CHARTS": True,
        "CHARTS_HEIGHT": None,
        "CHARTS_WIDTH": 6,
        "CHARTS_ROWS": None,
        "CHARTS_COLS": None,
        "CHART_CHROMAGRAM": True,
        "CHART_PREDICTION": True,

        "AP_FACTORY": AudioProcessing.factory,
        "AP_LOAD_STRATEGY_FACTORY": URILoadStrategy.factory,
        "AP_STFT_STRATEGY_FACTORY": CQTStrategy.factory,
        "AP_CHROMA_STRATEGY_FACTORY": FilteringChromaStrategy.factory,
        "AP_BEAT_STRATEGY_FACTORY": SyncBeatStrategy.factory,

        "SAMPLING_FREQUENCY": 44100,
        "N_OCTAVES": 84 // 12,
        "N_BINS": 84,
        "BINS_PER_OCTAVE": 12 * 3,
        "MIN_FREQ": 110,
        "HOP_LENGTH": 4096,

        "PREDICT_STRATEGY_FACTORY": TemplatePredictStrategy.factory,
        "PREDICT_STRATEGY_ALPHA": 0.5,
        "PREDICT_STRATEGY_DEPTH": 8,
        "PREDICT_STRATEGY_BOTTOM_THRESHOLD": 0,
        "PREDICT_STRATEGY_UPPER_THRESHOLD": 2
    }

    _audio_processing: AudioProcessing = None
    _chord_recognition: PredictStrategy = None
    _display: ChartBuilder = None

    def __init__(self, config=None) -> None:

        if config is None:
            config = {}
        _cfg = dict()
        _cfg.update(self.default_config)
        _cfg.update(config)

        super().__init__(_cfg)

        self._audio_processing = self.config["AP_FACTORY"](self)
        self._chord_recognition = self.config["PREDICT_STRATEGY_FACTORY"](self)
        self._display = ChartBuilder(
            self.charts_rows,
            self.charts_cols,
            self.charts_width,
            self.charts_height
        )

    def from_path(self, absolute_path: str, annotation_path: str = None,
                  chord_recognition_factory: PredictStrategy.factory = None):

        chroma_sync, beat_t = self._audio_processing.process(absolute_path)

        if chord_recognition_factory is not None:
            prediction = chord_recognition_factory(self).predict(chroma_sync)
        else:
            prediction = self._chord_recognition.predict(chroma_sync)

        if self.charts:
            if self.chart_chromagram:
                self._display.chromagram(chroma_sync, beat_t)
            if self.chart_prediction and annotation_path is not None:
                annotation_timeline = parse_annotation(self, annotation_path)
                prediction_timeline = make_timeline(beat_t, prediction)
                self._display.prediction(prediction_timeline, annotation_timeline)

        self._display.show()
        return prediction
