"""Person 2: Minari loading, episode flattening, normalization and batch sampler."""

from .config import ProjectConfig
from .contracts import OfflineDataset


def load_offline_dataset(config: ProjectConfig, *, download: bool = False) -> OfflineDataset:
    """Return validated normalized transitions. Download requires explicit opt-in."""
    raise NotImplementedError("Person 2: dataset loader is not implemented; see PROJECT_PLAN.md")
