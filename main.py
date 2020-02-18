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

from chordify.app import Chordify
from chordify.audio_processing import HPSSChromaStrategy
from chordify.learn import SVCLearn

app = Chordify()
with app.with_config({
    "CHORD_RECOGNITION_CLASS": SVCLearn,
    "AP_CHROMA_STRATEGY_CLASS": HPSSChromaStrategy
}) as m_app:
    # m_app.from_samples(iterable=chain(
    #    SupervisedDirectoryAdapter(Path("./chords/Guitar_Only/a"), IChord(ChordKey.A, ChordType.MAJOR)),
    #    SupervisedDirectoryAdapter(Path("./chords/Guitar_Only/am"), IChord(ChordKey.A, ChordType.MINOR)),
    #    SupervisedDirectoryAdapter(Path("./chords/Guitar_Only/bm"), IChord(ChordKey.B, ChordType.MINOR)),
    #    SupervisedDirectoryAdapter(Path("./chords/Guitar_Only/c"), IChord(ChordKey.C, ChordType.MAJOR)),
    #    SupervisedDirectoryAdapter(Path("./chords/Guitar_Only/d"), IChord(ChordKey.D, ChordType.MAJOR)),
    #    SupervisedDirectoryAdapter(Path("./chords/Guitar_Only/dm"), IChord(ChordKey.D, ChordType.MINOR)),
    #    SupervisedDirectoryAdapter(Path("./chords/Guitar_Only/e"), IChord(ChordKey.E, ChordType.MAJOR)),
    #    SupervisedDirectoryAdapter(Path("./chords/Guitar_Only/em"), IChord(ChordKey.E, ChordType.MINOR)),
    #    SupervisedDirectoryAdapter(Path("./chords/Guitar_Only/f"), IChord(ChordKey.F, ChordType.MAJOR)),
    #    SupervisedDirectoryAdapter(Path("./chords/Guitar_Only/g"), IChord(ChordKey.G, ChordType.MAJOR))
    # ))
    p = m_app.from_path("./song.m4a", "./chords.lab")
    print(p)
