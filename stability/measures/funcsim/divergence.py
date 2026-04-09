from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.spatial.distance
import scipy.special
import torch

from ..utils import to_numpy_if_needed

from ._base import PairwiseFuncSimMeasure
from .utils import check_has_two_axes


# adapted implementation from scipy to always return non-negative values, and to change to log base 2
def jensen_shannon_divergence(p: npt.NDArray, q: npt.NDArray, axis=0, keepdims=False) -> float:
    p = p / np.sum(p, axis=axis, keepdims=True)
    q = q / np.sum(q, axis=axis, keepdims=True)
    m = (p + q) / 2.0
    left = scipy.special.rel_entr(p, m)
    right = scipy.special.rel_entr(q, m)
    left_sum = np.sum(left, axis=axis, keepdims=keepdims)
    right_sum = np.sum(right, axis=axis, keepdims=keepdims)
    js = left_sum + right_sum
    js /= np.log(2)
    return np.clip(js / 2.0, a_min=0, a_max=1)


class JSD(PairwiseFuncSimMeasure):
    def __init__(self):
        super().__init__(larger_is_more_similar=False, is_symmetric=True)

    def __call__(self, output_a: torch.Tensor | npt.NDArray, output_b: torch.Tensor | npt.NDArray) -> Any:
        output_a, output_b = to_numpy_if_needed(check_has_two_axes(output_a), check_has_two_axes(output_b))

        output_a = scipy.special.softmax(output_a, axis=1)
        output_b = scipy.special.softmax(output_b, axis=1)
        return np.nanmean(
            [jensen_shannon_divergence(output_a_i, output_b_i) for output_a_i, output_b_i in zip(output_a, output_b)]
        )
