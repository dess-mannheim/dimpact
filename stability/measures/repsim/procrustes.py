import warnings
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import numpy.typing as npt
import scipy.linalg
import scipy.optimize
import torch
from .utils import adjust_dimensionality
from .utils import align_spatial_dimensions
from .utils import center_columns
from .utils import flatten
from .utils import normalize_matrix_norm
from ._base import RepresentationalSimilarityMeasure
from ._base import SHAPE_TYPE
from ..utils import to_numpy_if_needed


def _validate_aligned_cossim_inputs(
    R: Union[torch.Tensor, npt.NDArray], Rp: Union[torch.Tensor, npt.NDArray], shape: SHAPE_TYPE
) -> tuple[npt.NDArray, npt.NDArray]:
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)
    if not np.isfinite(R).all() or not np.isfinite(Rp).all():
        raise ValueError("Aligned cosine similarity requires finite-valued inputs.")
    if R.shape[0] == 0 or Rp.shape[0] == 0:
        raise ValueError("Aligned cosine similarity is undefined for empty inputs.")
    if not np.any(R) or not np.any(Rp):
        raise ValueError("Aligned cosine similarity is undefined, since one of the inputs only contains zeroes.")
    return R.astype(np.float64, copy=False), Rp.astype(np.float64, copy=False)


def _safe_row_cosine(r: npt.NDArray, rp: npt.NDArray, eps: float = 1e-12) -> float:
    nr = float(np.linalg.norm(r))
    nrp = float(np.linalg.norm(rp))
    if nr <= eps or nrp <= eps:
        return np.nan
    val = float(r.dot(rp) / (nr * nrp))
    return float(np.clip(val, -1.0, 1.0))


def orthogonal_procrustes(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)
    R, Rp = adjust_dimensionality(R, Rp)
    nucnorm = scipy.linalg.orthogonal_procrustes(R, Rp)[1]
    squared_dist = -2 * nucnorm + np.linalg.norm(R, ord="fro") ** 2 + np.linalg.norm(Rp, ord="fro") ** 2
    if squared_dist < 0:
        warnings.warn(
            f"Squared Orthogonal Procrustes distance is less than 0, but small, likely due to numerical errors. "
            f"Exact value={squared_dist}. Rounding to zero."
        )
        squared_dist = 0
    return np.sqrt(squared_dist)


def procrustes_size_and_shape_distance(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
) -> float:
    """Same setup as Williams et al., 2021 for the rotation invariant metric"""
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)
    R, Rp = center_columns(R), center_columns(Rp)
    return orthogonal_procrustes(R, Rp, "nd")


def orthogonal_procrustes_centered_and_normalized(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
) -> float:
    """Same setup as Ding et al., 2021"""
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)
    R, Rp = center_columns(R), center_columns(Rp)
    R, Rp = normalize_matrix_norm(R), normalize_matrix_norm(Rp)
    return orthogonal_procrustes(R, Rp, "nd")


def permutation_procrustes(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
    optimal_permutation_alignment: Optional[Tuple[npt.NDArray, npt.NDArray]] = None,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)
    R, Rp = adjust_dimensionality(R, Rp)

    if not optimal_permutation_alignment:
        PR, PRp = scipy.optimize.linear_sum_assignment(R.T @ Rp, maximize=True)  # returns column assignments
        optimal_permutation_alignment = (PR, PRp)
    PR, PRp = optimal_permutation_alignment
    return float(np.linalg.norm(R[:, PR] - Rp[:, PRp], ord="fro"))


def permutation_angular_shape_metric(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)
    R, Rp = adjust_dimensionality(R, Rp)
    R, Rp = normalize_matrix_norm(R), normalize_matrix_norm(Rp)

    PR, PRp = scipy.optimize.linear_sum_assignment(R.T @ Rp, maximize=True)  # returns column assignments

    aligned_R = R[:, PR]
    aligned_Rp = Rp[:, PRp]

    # matrices are already normalized so no division necessary
    corr = np.trace(aligned_R.T @ aligned_Rp)

    # From https://github.com/ahwillia/netrep/blob/0f3d825aad58c6d998b44eb0d490c0c5c6251fc9/netrep/utils.py#L107  # noqa: E501
    # numerical precision issues require us to clip inputs to arccos
    return np.arccos(np.clip(corr, -1.0, 1.0))


def orthogonal_angular_shape_metric(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)
    R, Rp = adjust_dimensionality(R, Rp)
    R, Rp = normalize_matrix_norm(R), normalize_matrix_norm(Rp)

    Qstar, nucnorm = scipy.linalg.orthogonal_procrustes(R, Rp)
    # matrices are already normalized so no division necessary
    corr = np.trace(Qstar.T @ R.T @ Rp)  # = \langle RQ, R' \rangle

    # From https://github.com/ahwillia/netrep/blob/0f3d825aad58c6d998b44eb0d490c0c5c6251fc9/netrep/utils.py#L107  # noqa: E501
    # numerical precision issues require us to clip inputs to arccos
    return float(np.arccos(np.clip(corr, -1.0, 1.0)))


def orthogonal_angular_shape_metric_centered(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
) -> float:
    """Williams et al., 2021 version"""
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)
    R, Rp = center_columns(R), center_columns(Rp)
    return orthogonal_angular_shape_metric(R, Rp, "nd")


def aligned_cossim(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
) -> float:
    R, Rp = _validate_aligned_cossim_inputs(R, Rp, shape)

    R, Rp = adjust_dimensionality(R, Rp)
    try:
        Q, _ = scipy.linalg.orthogonal_procrustes(R, Rp)
    except np.linalg.LinAlgError:
        try:
            U, _, Vt = scipy.linalg.svd(R.T @ Rp, lapack_driver="gesvd")
            Q = U @ Vt
        except np.linalg.LinAlgError:
            warnings.warn("Numerical issues with SVD in aligned_cossim; retrying with light regularization.")
            # last-resort: lightly regularized SVD
            eps = 1e-8
            M = R.T @ Rp + eps * np.eye(R.shape[1])
            U, _, Vt = scipy.linalg.svd(M, lapack_driver="gesvd")
            Q = U @ Vt

    R_aligned = R @ Q
    row_cossims = np.array([_safe_row_cosine(r, rp) for r, rp in zip(R_aligned, Rp)], dtype=np.float64)
    valid = np.isfinite(row_cossims)
    nan_ct = int((~valid).sum())
    if nan_ct == R.shape[0]:
        raise ValueError(
            "Aligned cosine similarity undefined since full-zero representations occurred in all instances."
        )
    elif nan_ct > 0:
        warnings.warn(
            f"In {nan_ct} instance(s), full-zero instance representations have been detected, yielding "
            f"undefined cosine similarity for these. These rows were left out when aggregating cosine "
            f"similarities."
        )

    return float(row_cossims[valid].mean())


def permutation_aligned_cossim(R: Union[torch.Tensor, npt.NDArray], Rp: Union[torch.Tensor, npt.NDArray]) -> float:
    R, Rp = _validate_aligned_cossim_inputs(R, Rp, "nd")

    R, Rp = adjust_dimensionality(R, Rp)

    PR, PRp = scipy.optimize.linear_sum_assignment(R.T @ Rp, maximize=True)  # returns column assignments
    R_aligned = R[:, PR]
    Rp_aligned = Rp[:, PRp]

    row_cossims = np.array([_safe_row_cosine(r, rp) for r, rp in zip(R_aligned, Rp_aligned)], dtype=np.float64)
    valid = np.isfinite(row_cossims)
    nan_ct = int((~valid).sum())
    if nan_ct == R.shape[0]:
        raise ValueError(
            "Aligned cosine similarity undefined since full-zero representations occurred in all instances."
        )
    elif nan_ct > 0:
        warnings.warn(
            f"In {nan_ct} instance(s), full-zero instance representations have been detected, yielding "
            f"undefined cosine similarity for these. These rows were left out when aggregating cosine "
            f"similarities."
        )
    return float(row_cossims[valid].mean())


class ProcrustesSizeAndShapeDistance(RepresentationalSimilarityMeasure):
    def __init__(self):
        super().__init__(
            sim_func=procrustes_size_and_shape_distance,
            larger_is_more_similar=False,
            is_metric=True,
            is_symmetric=True,
            invariant_to_affine=False,  # because default lambda=0
            invariant_to_invertible_linear=False,
            invariant_to_ortho=True,
            invariant_to_permutation=True,
            invariant_to_isotropic_scaling=False,
            invariant_to_translation=True,
        )

    def __call__(self, R: torch.Tensor | npt.NDArray, Rp: torch.Tensor | npt.NDArray, shape: SHAPE_TYPE) -> float:
        if shape == "nchw":
            # Move spatial dimensions into the sample dimension
            # If not the same spatial dimension, resample via FFT.
            R, Rp = align_spatial_dimensions(R, Rp)
            shape = "nd"

        return self.sim_func(R, Rp, shape)


class OrthogonalProcrustesCenteredAndNormalized(RepresentationalSimilarityMeasure):
    def __init__(self):
        super().__init__(
            sim_func=orthogonal_procrustes_centered_and_normalized,
            larger_is_more_similar=False,
            is_metric=True,
            is_symmetric=True,
            invariant_to_affine=False,  # because default lambda=0
            invariant_to_invertible_linear=False,
            invariant_to_ortho=True,
            invariant_to_permutation=True,
            invariant_to_isotropic_scaling=True,
            invariant_to_translation=True,
        )

    def __call__(self, R: torch.Tensor | npt.NDArray, Rp: torch.Tensor | npt.NDArray, shape: SHAPE_TYPE) -> float:
        if shape == "nchw":
            # Move spatial dimensions into the sample dimension
            # If not the same spatial dimension, resample via FFT.
            R, Rp = align_spatial_dimensions(R, Rp)
            shape = "nd"

        return self.sim_func(R, Rp, shape)


class PermutationProcrustes(RepresentationalSimilarityMeasure):
    def __init__(self):
        super().__init__(
            sim_func=permutation_procrustes,
            larger_is_more_similar=False,
            is_metric=True,
            is_symmetric=True,
            invariant_to_affine=False,  # because default lambda=0
            invariant_to_invertible_linear=False,
            invariant_to_ortho=False,
            invariant_to_permutation=True,
            invariant_to_isotropic_scaling=False,
            invariant_to_translation=False,
        )

    def __call__(self, R: torch.Tensor | npt.NDArray, Rp: torch.Tensor | npt.NDArray, shape: SHAPE_TYPE) -> float:
        if shape == "nchw":
            # Move spatial dimensions into the sample dimension
            # If not the same spatial dimension, resample via FFT.
            R, Rp = align_spatial_dimensions(R, Rp)
            shape = "nd"

        return self.sim_func(R, Rp, shape)


class OrthogonalAngularShapeMetricCentered(RepresentationalSimilarityMeasure):
    def __init__(self):
        super().__init__(
            sim_func=orthogonal_angular_shape_metric_centered,
            larger_is_more_similar=False,
            is_metric=True,
            is_symmetric=True,
            invariant_to_affine=False,  # because default lambda=0
            invariant_to_invertible_linear=False,
            invariant_to_ortho=True,
            invariant_to_permutation=True,
            invariant_to_isotropic_scaling=True,
            invariant_to_translation=True,
        )

    def __call__(self, R: torch.Tensor | npt.NDArray, Rp: torch.Tensor | npt.NDArray, shape: SHAPE_TYPE) -> float:
        if shape == "nchw":
            # Move spatial dimensions into the sample dimension
            # If not the same spatial dimension, resample via FFT.
            R, Rp = align_spatial_dimensions(R, Rp)
            shape = "nd"

        return self.sim_func(R, Rp, shape)


class AlignedCosineSimilarity(RepresentationalSimilarityMeasure):
    def __init__(self):
        super().__init__(
            sim_func=aligned_cossim,
            larger_is_more_similar=True,
            is_metric=False,
            is_symmetric=True,
            invariant_to_affine=False,  # because default lambda=0
            invariant_to_invertible_linear=False,
            invariant_to_ortho=True,
            invariant_to_permutation=True,
            invariant_to_isotropic_scaling=False,
            invariant_to_translation=False,
        )

    def __call__(self, R: torch.Tensor | npt.NDArray, Rp: torch.Tensor | npt.NDArray, shape: SHAPE_TYPE) -> float:
        if shape == "nchw":
            # Move spatial dimensions into the sample dimension
            # If not the same spatial dimension, resample via FFT.
            R, Rp = align_spatial_dimensions(R, Rp)
            shape = "nd"

        return self.sim_func(R, Rp, shape)
