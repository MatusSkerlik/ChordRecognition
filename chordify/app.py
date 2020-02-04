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

from .audio_processing import *
from .chord_recognition import *
from .config import ConfigAttribute
from .ctx import Context


class Chordify(Context):
    sampling_freq = ConfigAttribute("SAMPLING_FREQUENCY")
    n_octaves = ConfigAttribute("N_OCTAVES")
    f_min = ConfigAttribute("MIN_FREQ")

    default_config: Config = {
        "SAMPLING_FREQUENCY": 44100,
        "N_OCTAVES": 84 / 12,
        "N_BINS": 84,
        "BINS_PER_OCTAVE": 12,
        "MIN_FREQ": 110,
        "HOP_LENGTH": 512,

        "AP_LOAD_STRATEGY_FACTORY": URILoadStrategy.factory,
        "AP_STFT_STRATEGY_FACTORY": CQTStrategy.factory,
        "AP_CHROMA_STRATEGY_FACTORY": FilteringChromaStrategy.factory,
        "AP_BEAT_STRATEGY_FACTORY": SyncBeatStrategy.factory,

        "PREDICT_STRATEGY_FACTORY": TemplatePredictStrategy.factory,

        "PREDICT_STRATEGY_ALPHA": 0.5,
        "PREDICT_STRATEGY_DEPTH": 8,
        "PREDICT_STRATEGY_BOTTOM_THRESHOLD": 0,
        "PREDICT_STRATEGY_UPPER_THRESHOLD": 2
    }

    _audio_processing: AudioProcessing = None
    _chord_recognition: PredictStrategy = None
    _absolute_path = None

    def __init__(self, config=None) -> None:

        if config is None:
            config = {}
        _cfg = dict()
        _cfg.update(self.default_config)
        _cfg.update(config)

        super().__init__(_cfg)

        self._audio_processing = AudioProcessingBuilder(self).build()
        self._chord_recognition = self.config["PREDICT_STRATEGY_FACTORY"](self)

    @property
    def audio_processing(self):
        return self._audio_processing

    def define_audio_processing(self, audio_processing_builder: AudioProcessingBuilder):
        if audio_processing_builder is None:
            raise IllegalArgumentError

        self._audio_processing = audio_processing_builder.build()

    @property
    def chord_recognition(self):
        return self._chord_recognition

    def chords_from(self, absolute_path):
        chroma_sync, beat_f = self.audio_processing.process(absolute_path)
        chord_progression = self.chord_recognition.run(chroma_sync)
        return chord_progression
