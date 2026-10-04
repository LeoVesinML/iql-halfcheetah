"""Person 3 implements orchestration; Person 1 reviews end-to-end compatibility."""

from pathlib import Path

from .config import ProjectConfig


def train(config: ProjectConfig, *, run_dir: Path) -> Path:
    """Train on fixed offline data; return final checkpoint path, write manifest + JSONL."""
    raise NotImplementedError("Person 3: training orchestration is not implemented")
