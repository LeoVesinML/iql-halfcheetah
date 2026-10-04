"""Stable boundaries shared by all four feature branches; see docs/INTERFACES.md."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]
Metrics = dict[str, float]


@dataclass(frozen=True)
class TransitionBatch:
    observations: FloatArray  # (B, observation_dim), normalized consistently
    actions: FloatArray  # (B, action_dim), original environment bounds
    rewards: FloatArray  # (B, 1)
    next_observations: FloatArray  # (B, observation_dim)
    terminated: BoolArray  # (B, 1), masks bootstrapping
    truncated: BoolArray  # (B, 1), does not mask bootstrapping


@dataclass(frozen=True)
class DatasetInfo:
    dataset_id: str
    observation_dim: int
    action_dim: int
    action_low: FloatArray
    action_high: FloatArray
    observation_mean: FloatArray
    observation_std: FloatArray  # includes epsilon convention; record in metadata
    reward_scale: float
    reward_shift: float


class OfflineDataset(Protocol):
    info: DatasetInfo

    def sample(self, batch_size: int, rng: np.random.Generator) -> TransitionBatch: ...

    def __len__(self) -> int: ...


class Policy(Protocol):
    def act(self, observation: FloatArray, *, deterministic: bool = True) -> FloatArray:
        """One preprocessed observation (D,) -> bounded action (A,)."""
        ...


class Learner(Policy, Protocol):
    def update(self, batch: TransitionBatch) -> Metrics: ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...


@dataclass(frozen=True)
class EvaluationResult:
    episode_returns: tuple[float, ...]
    episode_lengths: tuple[int, ...]
    episode_seeds: tuple[int, ...]
