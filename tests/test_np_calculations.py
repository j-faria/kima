from math import floor
from dataclasses import dataclass

import pytest
from hypothesis import given, settings
from hypothesis.strategies import integers, lists, composite
import numpy as np

from pykima.analysis import np_bayes_factor_threshold


@dataclass
class dummy:
    max_components: int
    Np: np.ndarray


@composite
def gen(draw, threshold=150):
    Np = draw(integers(0, 5))
    max_samples = draw(integers(min_value=threshold + 1, max_value=10000))
    min_size = max(2, Np + 1)
    maxv = floor(max_samples / threshold)
    lst = draw(
        lists(integers(min_value=0, max_value=maxv), min_size=min_size,
              max_size=6))
    lst[Np] = max_samples
    return lst, Np


# @pytest.mark.parametrize("threshold", [0.5, 1.0, 2.0, 10, 50, 150, 500])
@given(gen())
# @settings(max_examples=5)
def test(ex):
    n, Np = ex
    n = np.atleast_1d(n)
    Np_samples = np.concatenate([np.full(aa, i) for i, aa in enumerate(n)])
    res = dummy(max_components=Np, Np=Np_samples)
    resNp = np_bayes_factor_threshold(res)
    # print(ex, resNp)
    assert Np == resNp
