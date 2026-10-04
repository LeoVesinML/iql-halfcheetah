"""Person 2: recover the evaluation environment from the Minari dataset EnvSpec."""

from typing import TYPE_CHECKING

from .config import ProjectConfig

if TYPE_CHECKING:
    import gymnasium as gym


def make_evaluation_env(config: ProjectConfig, *, render_mode: str | None = None) -> "gym.Env":
    """Recover exact dataset environment; accept rgb_array for video recording."""
    raise NotImplementedError("Person 2: environment factory is not implemented")
