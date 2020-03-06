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
from functools import lru_cache
from itertools import tee
from math import sin, pi, cos

import numpy as np
from numpy.linalg import norm


def _pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    next(b, None)
    return zip(a, b)


@lru_cache
def hcdf_matrix(r1: float = 1.0, r2: float = 1.0, r3: float = .5):
    matrix = np.zeros((6, 12))
    matrix[0] = tuple((r1 * sin((l * 7 * pi) / 6) for l in range(12)))
    matrix[1] = tuple((r1 * cos((l * 7 * pi) / 6) for l in range(12)))
    matrix[2] = tuple((r2 * sin((l * 3 * pi) / 2) for l in range(12)))
    matrix[3] = tuple((r2 * cos((l * 3 * pi) / 2) for l in range(12)))
    matrix[4] = tuple((r3 * sin((l * 2 * pi) / 3) for l in range(12)))
    matrix[5] = tuple((r3 * cos((l * 2 * pi) / 3) for l in range(12)))

    return matrix


def hcdf(frames: np.ndarray):
    """ Harmonic change detect function, #TODO implement M frames solution """
    matrix = hcdf_matrix()

    l2_seq = list()
    for f0, f1 in _pairwise(frames.T):
        tv0 = np.dot(matrix, f0) / norm(f0, ord=1)  # manhattan distance
        tv1 = np.dot(matrix, f1) / norm(f0, ord=1)  # manhattan distance
        d = norm(tv0 - tv1, ord=2)  # euclidean distance
        l2_seq.append(d)
    return np.array(l2_seq)
