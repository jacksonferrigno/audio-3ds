import os

import numpy as np
from stable_baselines3 import PPO

from src.agent.cnn.config import DEFAULT_RL_CHECKPOINT, CACHE_STEPS_PER_SAMPLE, STEPS_PER_SAMPLE
from src.env.echo_env import EchoEnv
from src.sim.room import Room


def load_rl_agent(checkpoint_path: str = DEFAULT_RL_CHECKPOINT) -> PPO:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"RL checkpoint not found: {checkpoint_path}\n"
            "Train the RL agent first with: uv run python main.py"
        )
    return PPO.load(checkpoint_path)


def generate_occupancy_map_with_agent(
    room: Room,
    rl_model: PPO,
    n_steps: int = STEPS_PER_SAMPLE,
    seed: int | None = None,
    deterministic: bool = True,
    fast: bool = False,
) -> np.ndarray:
    env = EchoEnv(room=room, max_steps=n_steps + 5)
    if fast:
        obs = env.reset_for_cache(seed=seed)
        for _ in range(n_steps - 1):
            action, _ = rl_model.predict(obs, deterministic=deterministic)
            obs = env.cache_step(action)
            if env.current_step >= env.max_steps:
                break
        return env.occupancy_map.copy()

    obs, _ = env.reset(seed=seed)

    for _ in range(n_steps):
        action, _ = rl_model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    return env.occupancy_map.copy()


def generate_occupancy_map_for_cache(
    room: Room,
    rl_model: PPO,
    n_steps: int = CACHE_STEPS_PER_SAMPLE,
    seed: int | None = None,
) -> np.ndarray:
    return generate_occupancy_map_with_agent(
        room,
        rl_model,
        n_steps=n_steps,
        seed=seed,
        deterministic=True,
        fast=True,
    )
