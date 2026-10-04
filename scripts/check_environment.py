"""Dependency smoke check only: no dataset loader, learner or rendering."""

import gymnasium as gym
import numpy as np


def main() -> None:
    env = gym.make("HalfCheetah-v5")
    try:
        observation, _ = env.reset(seed=0)
        assert observation.shape == (17,)
        assert env.action_space.shape == (6,)
        env.action_space.seed(0)
        for _ in range(10):
            observation, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            assert np.isfinite(observation).all() and np.isfinite(reward)
            if terminated or truncated:
                env.reset()
        print("HalfCheetah-v5: reset + 10 finite steps passed (no rendering or learning)")
    finally:
        env.close()


if __name__ == "__main__":
    main()
