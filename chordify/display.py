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

from copy import deepcopy
from typing import List

import librosa.display
import matplotlib.pyplot as plt

from .app import ChordKey, ChordType


def c2n(to: int) -> int:
    n = 2
    while to > n:
        n *= 2
    return n


def plot_chroma(chroma, tempo, beat_time):
    plt.figure(0)
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time',
                             x_coords=beat_time)
    plt.title('Chroma (beat time)' + (' tempo: %.2f ' % tempo) + ' bps')
    plt.tight_layout()
    plt.show()


def all_chords_str() -> List[str]:
    all_chords = list()
    all_chords.append("N")
    all_chords.extend(
        ('%s%s' % (chord_key.__str__(), chord_type.__str__())) for chord_type in ChordType for chord_key in
        ChordKey if chord_type is not ChordType.UNKNOWN if chord_key is not ChordKey.UNKNOWN
    )
    return all_chords


def normalize_chord_str(chord_str: str):
    normalized = deepcopy(chord_str)[:5]

    if normalized.find("maj") > -1:
        normalized = normalized[:1]

    return normalized


def index_chord_str(chord_str: str, normalize=False):
    normalized = chord_str
    if normalize:
        normalized = normalize_chord_str(chord_str)

    chord_map = {'N': 0}
    for chord_str in all_chords_str():
        chord_map[chord_str] = len(chord_map)

    return chord_map.get(normalized) or 0


def plot_prediction(original: List, predicted: List):
    x_p = list(chord_time[0] for chord_time in predicted for i in range(2))[1:]
    y_p = list(index_chord_str(chord_time[2]) for chord_time in predicted for i in range(2))[:-1]
    x_o = list(chord_time[0] for chord_time in original for i in range(2))[1:]
    y_o = list(index_chord_str(chord_time[2], normalize=True) for chord_time in original for i in range(2))[:-1]
    plt.figure(1, figsize=(30, 15))
    plt.plot(x_o, y_o, 'g--')
    plt.plot(x_p, y_p, 'r-')
    plt.ylabel('Chords')
    plt.xlabel('Time (s)')
    plt.xlim(0, original[len(original) - 1][1])
    plt.yticks(range(0, (4 * 12) + 1), all_chords_str())
    # plt.xticks(x_o, list(str(datetime.timedelta(seconds=chord_time[0]))[2:7] for chord_time in original for i in
    # range(2))[1:])
    plt.axes().xaxis.set_major_locator(plt.LinearLocator())
    plt.show()
