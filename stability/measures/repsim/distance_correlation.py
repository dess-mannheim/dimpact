import warnings
from typing import Optional
from typing import Union

import numpy as np
import numpy.typing as npt
import sklearn.metrics
import torch
from .utils import double_center
from .utils import flatten
from ._base import RSMSimilarityMeasure
from ._base import SHAPE_TYPE
from ..utils import to_numpy_if_needed


def distance_correlation(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
    n_jobs: Optional[int] = None,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)
    if not np.isfinite(R).all() or not np.isfinite(Rp).all():
        raise ValueError("R and Rp must contain only finite values for distance correlation.")
    if R.shape[0] < 2 or Rp.shape[0] < 2:
        raise ValueError("distance_correlation requires at least 2 samples.")

    # Promote precision
    R = R.astype(np.float64, copy=False)
    Rp = Rp.astype(np.float64, copy=False)

    S = sklearn.metrics.pairwise_distances(R, metric="euclidean", n_jobs=n_jobs)
    Sp = sklearn.metrics.pairwise_distances(Rp, metric="euclidean", n_jobs=n_jobs)

    # rescale to avoid overflows
    S_max = S.max()
    Sp_max = Sp.max()
    if S_max > 0:
        S /= S_max
    else:
        warnings.warn("All pairwise distances in R are zero; distance correlation is undefined. Returning 0.0.")
        return 0.0
    if Sp_max > 0:
        Sp /= Sp_max
    else:
        warnings.warn("All pairwise distances in Rp are zero; distance correlation is undefined. Returning 0.0.")
        return 0.0

    S = double_center(S)
    Sp = double_center(Sp)

    def dCov2(x: npt.NDArray, y: npt.NDArray) -> np.floating:
        return np.multiply(x, y).mean()

    dcov_xx = float(dCov2(S, S))
    dcov_yy = float(dCov2(Sp, Sp))
    dcov_xy = float(dCov2(S, Sp))
    denom = np.sqrt(max(dcov_xx, 0.0) * max(dcov_yy, 0.0))
    if denom <= 0:
        warnings.warn("Zero distance variance encountered; returning 0.0 for distance correlation.")
        return 0.0

    ratio = dcov_xy / denom
    # Numerical noise can make the ratio slightly negative/greater-than-one.
    ratio = float(np.clip(ratio, 0.0, 1.0))
    return float(np.sqrt(ratio))


class DistanceCorrelation(RSMSimilarityMeasure):
    def __init__(self):
        super().__init__(
            sim_func=distance_correlation,
            larger_is_more_similar=True,
            is_metric=False,
            is_symmetric=True,
            invariant_to_affine=False,
            invariant_to_invertible_linear=False,
            invariant_to_ortho=True,
            invariant_to_permutation=True,
            invariant_to_isotropic_scaling=False,
            invariant_to_translation=True,
        )
