from typing import Dict

from .repsim import PWCCA
from .repsim import SVCCA
from .repsim import CKA
from .repsim import HardCorrelationMatch
from .repsim import SoftCorrelationMatch
from .repsim import DistanceCorrelation
from .repsim import EigenspaceOverlapScore
from .repsim import GeometryScore
from .repsim import Gulp
from .repsim import LinearRegression
from .repsim import IMDScore
from .repsim import JaccardSimilarity
from .repsim import RankSimilarity
from .repsim import SecondOrderCosineSimilarity
from .repsim import AlignedCosineSimilarity
from .repsim import OrthogonalAngularShapeMetricCentered
from .repsim import OrthogonalProcrustesCenteredAndNormalized
from .repsim import PermutationProcrustes
from .repsim import ProcrustesSizeAndShapeDistance
from .repsim import RSA
from .repsim import RSMNormDifference
from .repsim import RTD
from .repsim import ConcentricityDifference
from .repsim import MagnitudeDifference
from .repsim import UniformityDifference
from .repsim import RepresentationalSimilarityMeasure

from .funcsim import (
    FunctionalSimilarityMeasure,
    PairwiseFuncSimMeasure,
    PerformanceBasedFuncSimMeasure,
    GroupwiseFuncSimMeasure,
)
from .funcsim import Disagreement, MinMaxNormalizedDisagreement, StableCore
from .funcsim import JSD

CLASSES = [
    PWCCA,
    SVCCA,
    HardCorrelationMatch,
    SoftCorrelationMatch,
    DistanceCorrelation,
    EigenspaceOverlapScore,
    GeometryScore,
    IMDScore,
    Gulp,
    LinearRegression,
    JaccardSimilarity,
    RankSimilarity,
    SecondOrderCosineSimilarity,
    AlignedCosineSimilarity,
    OrthogonalAngularShapeMetricCentered,
    OrthogonalProcrustesCenteredAndNormalized,
    PermutationProcrustes,
    ProcrustesSizeAndShapeDistance,
    RSA,
    RSMNormDifference,
    ConcentricityDifference,
    MagnitudeDifference,
    UniformityDifference,
    CKA,
    RTD,
]


ALL_MEASURES: Dict[str, RepresentationalSimilarityMeasure] = {m().name: m() for m in CLASSES}

PAIRWISE_FUNCTIONAL_SIMILARITY_MEASURES: Dict[str, PairwiseFuncSimMeasure] = {
    m().name: m() for m in [JSD, Disagreement]
}
PERFORMANCE_BASED_FUNCTIONAL_SIMILARITY_MEASURES: Dict[str, PerformanceBasedFuncSimMeasure] = {
    m().name: m() for m in [MinMaxNormalizedDisagreement]
}
GROUPWISE_FUNCTIONAL_SIMILARITY_MEASURES: Dict[str, GroupwiseFuncSimMeasure] = {m().name: m() for m in [StableCore]}

ALL_FUNCSIM_MEASURES: Dict[str, FunctionalSimilarityMeasure] = {
    **PAIRWISE_FUNCTIONAL_SIMILARITY_MEASURES,
    **PERFORMANCE_BASED_FUNCTIONAL_SIMILARITY_MEASURES,
    **GROUPWISE_FUNCTIONAL_SIMILARITY_MEASURES,
}
