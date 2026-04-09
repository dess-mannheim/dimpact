from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Any, List
import numpy.typing as npt
import torch
from dataclasses import field


@dataclass
class FunctionalSimilarityMeasure(ABC):
    larger_is_more_similar: bool
    name: str = field(init=False)

    def __post_init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError


@dataclass
class PairwiseFuncSimMeasure(FunctionalSimilarityMeasure):
    is_symmetric: bool

    @abstractmethod
    def __call__(self, output_a: torch.Tensor | npt.NDArray, output_b: torch.Tensor | npt.NDArray) -> float:
        raise NotImplementedError


@dataclass
class PerformanceBasedFuncSimMeasure(FunctionalSimilarityMeasure):
    is_symmetric: bool

    @abstractmethod
    def __call__(
        self,
        output_a: torch.Tensor | npt.NDArray,
        output_b: torch.Tensor | npt.NDArray,
        accuracy_a: float,
        accuracy_b: float,
    ) -> float:
        raise NotImplementedError


@dataclass
class GroupwiseFuncSimMeasure(FunctionalSimilarityMeasure):

    @abstractmethod
    def __call__(self, output_list: List[torch.Tensor] | List[npt.NDArray]) -> float:
        raise NotImplementedError
