import numpy.typing as npt
import torch


def check_has_two_axes(x: npt.NDArray | torch.Tensor) -> npt.NDArray | torch.Tensor:
    if len(x.shape) != 2:
        raise ValueError(f"Matrix must have two dimensions, but has {len(x.shape)}")
    return x
