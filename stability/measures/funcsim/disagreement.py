import warnings
from typing import Any, List

import numpy as np
import numpy.typing as npt
import torch

from ..utils import to_numpy_if_needed
from ._base import (
    PairwiseFuncSimMeasure,
    GroupwiseFuncSimMeasure,
    PerformanceBasedFuncSimMeasure,
)
from .utils import check_has_two_axes


class Disagreement(PairwiseFuncSimMeasure):
    def __init__(self):
        super().__init__(larger_is_more_similar=False, is_symmetric=True)

    def __call__(self, output_a: torch.Tensor | npt.NDArray, output_b: torch.Tensor | npt.NDArray) -> Any:
        check_has_two_axes(output_a)
        check_has_two_axes(output_b)

        output_a, output_b = to_numpy_if_needed(output_a, output_b)
        if output_a.shape[0] != output_b.shape[0]:
            raise ValueError(
                f"output_a and output_b must have the same number of samples, got {output_a.shape[0]} and {output_b.shape[0]}."
            )
        if output_a.shape[0] == 0:
            raise ValueError("Disagreement is undefined for empty outputs (zero samples).")
        if not np.isfinite(output_a).all() or not np.isfinite(output_b).all():
            raise ValueError("output_a and output_b must contain only finite values.")
        return (output_a.argmax(axis=1) != output_b.argmax(axis=1)).sum() / len(output_a)


class MinMaxNormalizedDisagreement(PerformanceBasedFuncSimMeasure):
    def __init__(self):
        super().__init__(larger_is_more_similar=False, is_symmetric=True)

    def __call__(
        self,
        output_a: torch.Tensor | npt.NDArray,
        output_b: torch.Tensor | npt.NDArray,
        accuracy_a: float,
        accuracy_b: float,
    ) -> Any:

        da = Disagreement()
        disagreement = da(output_a, output_b)
        if not np.isfinite(accuracy_a) or not np.isfinite(accuracy_b):
            raise ValueError(f"Accuracies must be finite, got accuracy_a={accuracy_a}, accuracy_b={accuracy_b}.")
        if accuracy_a < 0.0 or accuracy_a > 1.0 or accuracy_b < 0.0 or accuracy_b > 1.0:
            raise ValueError(f"Accuracies must be in [0, 1], got accuracy_a={accuracy_a}, accuracy_b={accuracy_b}.")

        min_dis = abs(accuracy_a - accuracy_b)
        # At least max(0, a+b-1) samples must be jointly correct -> those samples always agree.
        max_dis = 1.0 - max(0.0, accuracy_a + accuracy_b - 1.0)

        if max_dis <= min_dis:
            warnings.warn(
                "Degenerate normalization interval in MinMaxNormalizedDisagreement; returning 0.0 by convention."
            )
            return 0.0

        if disagreement < min_dis or disagreement > max_dis:
            warnings.warn(
                f"Observed disagreement={disagreement} outside theoretical bounds [{min_dis}, {max_dis}]; clipping."
            )
            disagreement = float(np.clip(disagreement, min_dis, max_dis))

        score = (disagreement - min_dis) / (max_dis - min_dis)
        return float(np.clip(score, 0.0, 1.0))


class StableCore(GroupwiseFuncSimMeasure):
    def __init__(self):
        super().__init__(larger_is_more_similar=True)

    def __call__(self, output_list: List[torch.Tensor] | List[npt.NDArray]) -> Any:
        output_list = [to_numpy_if_needed(check_has_two_axes(output))[0] for output in output_list]
        label_matrix = np.array([output.argmax(axis=1) for output in output_list]).T
        return np.sum(np.all(label_matrix[:, [0]] == label_matrix, axis=1)) / label_matrix.shape[0]
