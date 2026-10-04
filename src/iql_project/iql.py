"""Person 3: implement losses, action selection and complete checkpoint state."""

from pathlib import Path

from .config import ProjectConfig
from .contracts import DatasetInfo, FloatArray, Metrics, TransitionBatch


class IQLAgent:
    def __init__(self, config: ProjectConfig, info: DatasetInfo) -> None:
        raise NotImplementedError("Person 3: IQL agent is not implemented")

    def update(self, batch: TransitionBatch) -> Metrics:
        """Return q_loss, v_loss, policy_loss, advantage_mean as finite Python floats."""
        raise NotImplementedError("Person 3: IQL update is not implemented")

    def act(self, observation: FloatArray, *, deterministic: bool = True) -> FloatArray:
        raise NotImplementedError("Person 3: policy action selection is not implemented")

    def save(self, path: Path) -> None:
        """Save model, optimizers, preprocessing, config, step, RNGs and provenance."""
        raise NotImplementedError("Person 3: checkpoint saving is not implemented")

    def load(self, path: Path) -> None:
        """Restore complete state into an agent configured for the same dataset contract."""
        raise NotImplementedError("Person 3: checkpoint loading is not implemented")
