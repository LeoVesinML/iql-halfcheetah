"""Validated Minari transitions, offline normalization and seeded NumPy sampling."""

import copy
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from .config import ProjectConfig
from .contracts import DatasetInfo, TransitionBatch
from .environment import _load_minari_dataset

OBSERVATION_STD_FLOOR = 1e-3


class ArrayOfflineDataset:
    """In-memory transitions implementing the shared OfflineDataset protocol.

    Sampling is uniform with replacement, controlled solely by the caller's RNG.
    ``metadata`` and ``save_metadata`` provide JSON-safe statistics and preprocessing
    for manifests/checkpoints without changing the shared DatasetInfo contract.
    """

    def __init__(
        self, transitions: TransitionBatch, info: DatasetInfo, metadata: dict[str, Any]
    ) -> None:
        self.info = info
        self._transitions = transitions
        self._metadata = metadata
        for field in fields(transitions):
            getattr(transitions, field.name).setflags(write=False)
        for field in fields(info):
            value = getattr(info, field.name)
            if isinstance(value, np.ndarray):
                value.setflags(write=False)

    def __len__(self) -> int:
        return len(self._transitions.observations)

    def sample(self, batch_size: int, rng: np.random.Generator) -> TransitionBatch:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        indices = rng.integers(len(self), size=batch_size)
        return TransitionBatch(
            **{
                field.name: getattr(self._transitions, field.name)[indices]
                for field in fields(self._transitions)
            }
        )

    @property
    def metadata(self) -> dict[str, Any]:
        """A copy: callers cannot accidentally change the preprocessing record."""
        return copy.deepcopy(self._metadata)

    def save_metadata(self, path: str | Path) -> None:
        """Save raw-data statistics, EnvSpec and exact float32 normalization values."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


def _numeric_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be a numeric array with shape {shape}")
    if np.iscomplexobj(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} contains invalid or non-finite values")
    with np.errstate(over="ignore"):
        array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains values outside the float32 range")
    return array


def _flags(value: Any, length: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (length,) or array.dtype != np.bool_:
        raise ValueError(f"{name} must be a bool array with shape ({length},)")
    return array


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values, dtype=np.float64)),
        "std": float(np.std(values, dtype=np.float64)),
    }


def load_offline_dataset(config: ProjectConfig, *, download: bool = False) -> ArrayOfflineDataset:
    """Load every episode without cross-episode links; downloading requires opt-in.

    Fit mean and population std on float32 offline ``observations[:-1]`` only,
    accumulating in float64. Store float32 mean/std with std floored at 1e-3;
    use those exact values for both current and next observations. Rewards and
    actions stay in environment units. Truncations do not disable bootstrapping.
    """
    dataset = _load_minari_dataset(config, download=download)
    total_steps, total_episodes = dataset.total_steps, dataset.total_episodes
    if total_steps <= 0 or total_episodes <= 0:
        raise ValueError("Dataset must contain episodes and transitions")

    observations = np.empty((total_steps, config.observation_dim), dtype=np.float32)
    next_observations = np.empty_like(observations)
    actions = np.empty((total_steps, config.action_dim), dtype=np.float32)
    rewards = np.empty((total_steps, 1), dtype=np.float32)
    terminated = np.empty((total_steps, 1), dtype=np.bool_)
    truncated = np.empty_like(terminated)
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    offset = 0
    for episode in dataset.iterate_episodes():
        prefix = f"Episode {episode.id}"
        length = len(episode)
        if length <= 0 or offset + length > total_steps:
            raise ValueError(f"{prefix}: invalid length or inconsistent dataset transition count")
        obs = _numeric_array(
            episode.observations, (length + 1, config.observation_dim), f"{prefix} observations"
        )
        act = _numeric_array(episode.actions, (length, config.action_dim), f"{prefix} actions")
        rew = _numeric_array(episode.rewards, (length,), f"{prefix} rewards")
        term = _flags(episode.terminations, length, f"{prefix} terminations")
        trunc = _flags(episode.truncations, length, f"{prefix} truncations")
        if np.any((term | trunc)[:-1]):
            raise ValueError(f"{prefix}: ending flag before the last transition")
        # A final unflagged transition is valid for a partial collected episode.
        if np.any(act < dataset.action_space.low) or np.any(act > dataset.action_space.high):
            raise ValueError(f"{prefix}: actions outside recorded bounds")
        if np.any(obs < dataset.observation_space.low) or np.any(
            obs > dataset.observation_space.high
        ):
            raise ValueError(f"{prefix}: observations outside recorded bounds")
        rows = slice(offset, offset + length)
        observations[rows] = obs[:-1]
        next_observations[rows] = obs[1:]
        actions[rows] = act
        rewards[rows, 0] = rew
        terminated[rows, 0] = term
        truncated[rows, 0] = trunc
        episode_returns.append(float(np.sum(rew, dtype=np.float64)))
        episode_lengths.append(length)
        offset += length
    if offset != total_steps or len(episode_lengths) != total_episodes:
        raise ValueError("Episode/transition counts do not match Minari metadata")

    mean = observations.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(observations.std(axis=0, dtype=np.float64), OBSERVATION_STD_FLOOR).astype(
        np.float32
    )
    info = DatasetInfo(
        dataset_id=config.dataset_id,
        observation_dim=config.observation_dim,
        action_dim=config.action_dim,
        action_low=dataset.action_space.low.astype(np.float32),
        action_high=dataset.action_space.high.astype(np.float32),
        observation_mean=mean,
        observation_std=std,
        reward_scale=1.0,
        reward_shift=0.0,
    )
    preprocessing = {
        field.name: value.tolist() if isinstance(value, np.ndarray) else value
        for field in fields(info)
        for value in (getattr(info, field.name),)
    }
    source_metadata = dataset.storage.metadata
    source = {
        name: source_metadata.get(name)
        for name in ("author", "algorithm_name", "code_permalink", "license")
    }
    if isinstance(source["author"], set):
        source["author"] = sorted(source["author"])
    metadata = {
        "schema_version": 1,
        "dataset_id": dataset.id,
        "total_episodes": total_episodes,
        "total_transitions": total_steps,
        "observation_dim": config.observation_dim,
        "action_dim": config.action_dim,
        "source": source,
        "dataset_minari_version": dataset.minari_version,
        "collection_requirements": source_metadata.get("requirements", []),
        "env_spec": json.loads(dataset.env_spec.to_json()),
        "preprocessing": {
            **preprocessing,
            "observation_std_floor": OBSERVATION_STD_FLOOR,
            "normalization_fit": "offline observations[:-1]; float64 accumulation; population std",
            "reward_transform": "reward * 1.0 + 0.0 (unchanged)",
        },
        "statistics": {
            "raw_reward": _summary(rewards),
            "raw_episode_return": _summary(np.asarray(episode_returns)),
            "episode_length": _summary(np.asarray(episode_lengths)),
            "terminated_transitions": int(terminated.sum()),
            "truncated_transitions": int(truncated.sum()),
            "partial_episodes": int(total_episodes - np.count_nonzero(terminated | truncated)),
        },
    }
    for array in (observations, next_observations):
        with np.errstate(over="ignore", invalid="ignore"):
            array -= mean
            array /= std
        if not np.isfinite(array).all():
            raise ValueError("Observation normalization produced non-finite values")
    return ArrayOfflineDataset(
        TransitionBatch(observations, actions, rewards, next_observations, terminated, truncated),
        info,
        metadata,
    )
