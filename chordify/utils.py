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


def evaluate(original: List, predicted: List) -> (float, float):
    original_copy = deepcopy(original)
    predicted_copy = deepcopy(predicted)

    original_copy.reverse()
    predicted_copy.reverse()

    sums = 0
    total = 1

    if len(original_copy) > 0 and len(predicted_copy) > 0:

        org = original_copy.pop()
        pred = predicted_copy.pop()

        while len(original_copy) > 0 and len(predicted_copy) > 0:

            if pred[1] < org[0]:
                pred = predicted_copy.pop()
                continue
            elif pred[0] > org[1]:
                org = original_copy.pop()
                total += 1
                continue

            pred_len = pred[1] - pred[0]
            org_len = org[1] - org[0]

            if org[2].find(pred[2]) > -1:
                if pred[0] <= org[0]:
                    if pred[1] <= org[1]:
                        sums += (pred_len - (org[0] - pred[0])) / org_len
                    else:
                        sums += 1
                elif pred[0] >= org[0]:
                    if pred[1] <= org[1]:
                        sums += pred_len / org_len
                    else:
                        sums += (pred_len - (pred[1] - org[1])) / org_len
            pred = predicted_copy.pop()
    return sums / total
