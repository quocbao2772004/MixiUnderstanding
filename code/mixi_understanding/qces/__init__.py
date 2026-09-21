"""Question-conditioned acoustic evidence separation.

The package exposes a checkpoint-free complementary-mask backbone and an
optional adapter for a separately installed AudioSep checkout.
"""

from mixi_understanding.qces.config import QCESConfig
from mixi_understanding.qces.counterfactual import (
    CounterfactualBatchSampler,
    CounterfactualGroupPlan,
    build_counterfactual_group_plan,
    counterfactual_objectives,
)
from mixi_understanding.qces.losses import LossWeights, QCESLoss
from mixi_understanding.qces.model import (
    QCESModel,
    QCESOutput,
    load_qces_checkpoint,
)
from mixi_understanding.qces.separators import (
    AudioSepConditionedAdapter,
    ComplementaryMaskSeparator,
    PhaseAwareComplexMaskSeparator,
    SeparatorAwareTemporalRefiner,
)
from mixi_understanding.qces.tokenization import StableHashTokenizer, TokenBatch

__all__ = [
    "AudioSepConditionedAdapter",
    "ComplementaryMaskSeparator",
    "CounterfactualBatchSampler",
    "CounterfactualGroupPlan",
    "PhaseAwareComplexMaskSeparator",
    "SeparatorAwareTemporalRefiner",
    "LossWeights",
    "QCESConfig",
    "QCESLoss",
    "QCESModel",
    "QCESOutput",
    "build_counterfactual_group_plan",
    "counterfactual_objectives",
    "load_qces_checkpoint",
    "StableHashTokenizer",
    "TokenBatch",
]
