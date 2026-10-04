"""Person 3: twin Q, value network and bounded stochastic continuous policy."""

from typing import TYPE_CHECKING

from .config import ProjectConfig
from .contracts import DatasetInfo

if TYPE_CHECKING:
    from torch import nn


def build_networks(config: ProjectConfig, info: DatasetInfo) -> dict[str, "nn.Module"]:
    """Return named modules q1, q2, value, policy and target critics; no shared optimizers."""
    raise NotImplementedError("Person 3: networks are not implemented")
