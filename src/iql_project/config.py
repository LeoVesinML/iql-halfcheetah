"""Shared, validated configuration. No side effects or device initialization."""

import math
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    env_id: str = "HalfCheetah-v5"
    dataset_id: str = "mujoco/halfcheetah/medium-v0"
    observation_dim: int = 17
    action_dim: int = 6
    discount: float = 0.99
    expectile: float = 0.7
    inverse_temperature: float = 3.0
    max_weight: float = 100.0
    target_tau: float = 0.005
    learning_rate: float = 3e-4
    batch_size: int = 256
    total_steps: int = 500_000
    eval_interval: int = 5_000
    checkpoint_interval: int = 50_000
    seed: int = 0
    device: str = "cpu"
    eval_episodes: int = 10
    eval_seed: int = 10_000
    output_dir: str = "results"

    def __post_init__(self) -> None:
        for name in (
            "observation_dim",
            "action_dim",
            "batch_size",
            "total_steps",
            "eval_interval",
            "checkpoint_interval",
            "eval_episodes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("seed", "eval_seed"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in (
            "discount",
            "expectile",
            "inverse_temperature",
            "max_weight",
            "target_tau",
            "learning_rate",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not 0 <= self.discount <= 1:
            raise ValueError("discount must lie in [0, 1]")
        if not 0.5 < self.expectile < 1:
            raise ValueError("expectile must lie in (0.5, 1) for upper-expectile IQL")
        if not 0 < self.target_tau <= 1:
            raise ValueError("target_tau must lie in (0, 1]")
        if min(self.inverse_temperature, self.max_weight, self.learning_rate) <= 0:
            raise ValueError("inverse_temperature, max_weight and learning_rate must be positive")
        for name in ("env_id", "dataset_id", "device", "output_dir"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if self.device not in ("cpu", "cuda", "auto"):
            raise ValueError("device must be cpu, cuda or auto")
        if (self.env_id, self.observation_dim, self.action_dim) != ("HalfCheetah-v5", 17, 6):
            raise ValueError("this project's environment contract is HalfCheetah-v5 / 17 / 6")


def load_config(path: str | Path) -> ProjectConfig:
    """Read flat TOML; reject misspelled fields instead of silently ignoring them."""
    with Path(path).open("rb") as file:
        values = tomllib.load(file)
    unknown = set(values) - {field.name for field in fields(ProjectConfig)}
    if unknown:
        raise ValueError(f"Unknown config fields: {', '.join(sorted(unknown))}")
    return ProjectConfig(**values)
