"""Person 4: reproducible episodic evaluation, curves and video recording."""

from pathlib import Path

from .config import ProjectConfig
from .contracts import DatasetInfo, EvaluationResult, Policy


def evaluate(policy: Policy, config: ProjectConfig, info: DatasetInfo) -> EvaluationResult:
    """Use raw observations -> saved normalization -> policy; report raw rewards."""
    raise NotImplementedError("Person 4: episodic evaluation is not implemented")


def record_video(
    policy: Policy, config: ProjectConfig, info: DatasetInfo, *, output_dir: Path, seed: int
) -> Path:
    """Record one deterministic policy episode via rgb_array; return playable MP4 path."""
    raise NotImplementedError("Person 4: demo recording is not implemented")
