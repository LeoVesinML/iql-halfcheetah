"""Recover the collection environment, including its recorded kwargs and horizon."""

from typing import TYPE_CHECKING

import gymnasium as gym
import minari
import numpy as np

from .config import ProjectConfig

if TYPE_CHECKING:
    from minari import MinariDataset


def _load_minari_dataset(config: ProjectConfig, *, download: bool = False) -> "MinariDataset":
    try:
        dataset = minari.load_dataset(config.dataset_id, download=download)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Minari dataset {config.dataset_id!r} is missing. Download explicitly with "
            f"`minari download {config.dataset_id}` or load_offline_dataset(config, download=True)."
        ) from error
    if dataset.id != config.dataset_id:
        raise ValueError(f"Dataset ID mismatch: expected {config.dataset_id}, got {dataset.id}")
    if dataset.env_spec is None or dataset.env_spec.id != config.env_id:
        raise ValueError(f"Dataset must record an EnvSpec for {config.env_id}")
    _validate_spaces(dataset.observation_space, dataset.action_space, config)
    return dataset


def _validate_spaces(
    observation_space: gym.Space, action_space: gym.Space, config: ProjectConfig
) -> None:
    for name, space, dimension in (
        ("observation", observation_space, config.observation_dim),
        ("action", action_space, config.action_dim),
    ):
        if (
            not isinstance(space, gym.spaces.Box)
            or space.shape != (dimension,)
            or not np.issubdtype(space.dtype, np.floating)
        ):
            raise ValueError(f"{name} space must be a continuous Box with shape ({dimension},)")
        if np.isnan(space.low).any() or np.isnan(space.high).any():
            raise ValueError(f"{name} bounds contain NaN")
        if np.any(space.low >= space.high):
            raise ValueError(f"{name} bounds must have low < high")
    if not np.isfinite(action_space.low).all() or not np.isfinite(action_space.high).all():
        raise ValueError("Action bounds must be finite")


def make_evaluation_env(config: ProjectConfig, *, render_mode: str | None = None) -> gym.Env:
    """Recover the exact collection EnvSpec; never download or fall back to defaults."""
    dataset = _load_minari_dataset(config)
    env = dataset.recover_environment(render_mode=render_mode)
    try:
        _validate_spaces(env.observation_space, env.action_space, config)
        for name in ("observation_space", "action_space"):
            recorded, recovered = getattr(dataset, name), getattr(env, name)
            if not (
                np.array_equal(recorded.low, recovered.low)
                and np.array_equal(recorded.high, recovered.high)
            ):
                raise ValueError(f"Recovered {name} bounds differ from the dataset")
        if env.spec is None or env.spec.id != config.env_id:
            raise ValueError("Recovered environment ID differs from the configured environment")
        if env.spec.max_episode_steps != dataset.env_spec.max_episode_steps:
            raise ValueError("Recovered environment horizon differs from the dataset")
    except Exception:
        env.close()
        raise
    return env
