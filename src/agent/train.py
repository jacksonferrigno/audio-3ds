# src/agent/train.py
import os

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from tqdm import tqdm

from src.agent.cnn.config import MAX_OBJECTS, MAX_PEOPLE
from src.agent.cnn.rooms import make_random_room
from src.env.echo_env import EchoEnv

# --- training config ---
TOTAL_TIMESTEPS = 100_000
MOVING_PEOPLE_AFTER = 50_000
N_ENVS = 4
MAX_STEPS = 125
CHECKPOINT_FREQ = 5_000
EVAL_FREQ = 2_500
EVAL_EPISODES = 10
DEFAULT_RESUME_CHECKPOINT = "checkpoints/best_model/best_model.zip"
EPISODE_MEAN_WINDOW = 10

# --- PPO hyperparams ---
LEARNING_RATE = 3e-4
N_STEPS = 64
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2

LOG_DIR = "runs/ppo_echo"
CHECKPOINT_DIR = "checkpoints/"
BEST_MODEL_PATH = "checkpoints/best_model"


def make_env():
    """Random room each episode — objects + 0-N static people."""

    def _init():
        room = make_random_room(max_objects=MAX_OBJECTS, max_people=MAX_PEOPLE)
        return EchoEnv(room=room, max_steps=MAX_STEPS)

    return _init


class TqdmCallback(BaseCallback):
    """Progress bar for the current learn() run."""

    def __init__(self, total_timesteps: int):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.pbar: tqdm | None = None
        self._start_steps = 0

    def _on_training_start(self) -> None:
        self._start_steps = self.num_timesteps
        target = self._start_steps + self.total_timesteps
        self.pbar = tqdm(
            total=target,
            initial=self._start_steps,
            unit="step",
            desc="PPO train",
            dynamic_ncols=True,
        )

    def _on_step(self) -> bool:
        if self.pbar is not None:
            self.pbar.n = self.num_timesteps
            info = self.locals["infos"][0]
            if self.locals["dones"][0]:
                self.pbar.set_postfix(
                    reward=f"{info.get('map_coverage', 0):.2f} cov",
                    moving=info.get("moving_people", False),
                )
            self.pbar.refresh()
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()


class EchoTensorBoardCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_coverages = []
        self.current_episode_reward = 0.0

    def _on_step(self) -> bool:
        self.current_episode_reward += float(self.locals["rewards"][0])
        info = self.locals["infos"][0]

        self.logger.record("echo/map_coverage", info.get("map_coverage", 0))
        self.logger.record("echo/f_start", info.get("f_start", 0))
        self.logger.record("echo/f_end", info.get("f_end", 0))
        self.logger.record("echo/direction", info.get("direction", 0))
        self.logger.record("echo/sweep_width", info.get("sweep_width", 0))
        self.logger.record("echo/moving_people", float(info.get("moving_people", False)))

        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_coverages.append(info.get("map_coverage", 0))
            self.logger.record("echo/episode_reward", self.current_episode_reward)
            self.logger.record("echo/episode_coverage", info.get("map_coverage", 0))
            self.logger.record(
                "echo/mean_reward",
                np.mean(self.episode_rewards[-EPISODE_MEAN_WINDOW:]),
            )
            self.logger.record(
                "echo/mean_coverage",
                np.mean(self.episode_coverages[-EPISODE_MEAN_WINDOW:]),
            )
            self.current_episode_reward = 0.0

        return True


class MovingPeopleCallback(BaseCallback):
    """After N global steps, people drift slightly each env step."""

    def __init__(self, enable_after: int = MOVING_PEOPLE_AFTER, verbose: int = 0):
        super().__init__(verbose)
        self.enable_after = enable_after
        self._enabled = False

    def _on_step(self) -> bool:
        if not self._enabled and self.num_timesteps >= self.enable_after:
            self.training_env.env_method("set_moving_people", True)
            self._enabled = True
            if self.verbose:
                print(f"moving people enabled at step {self.num_timesteps}")
        return True


def resolve_resume_checkpoint(resume_from: str | None) -> str | None:
    if resume_from == "":
        return None
    if resume_from is None:
        if os.path.exists(DEFAULT_RESUME_CHECKPOINT):
            return DEFAULT_RESUME_CHECKPOINT
        return None
    if not os.path.exists(resume_from):
        raise FileNotFoundError(f"checkpoint not found: {resume_from}")
    return resume_from


def load_or_create_model(vec_env, checkpoint_path: str | None):
    if checkpoint_path:
        print(f"loading checkpoint: {checkpoint_path}")
        return PPO.load(checkpoint_path, env=vec_env)

    print("starting fresh PPO model")
    return PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )


def train(resume_from: str | None = None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    resume_checkpoint = resolve_resume_checkpoint(resume_from)
    if resume_checkpoint:
        print(f"resuming from: {resume_checkpoint}")
    else:
        print("no checkpoint found — starting fresh")

    vec_env = make_vec_env(make_env(), n_envs=N_ENVS)
    model = load_or_create_model(vec_env, resume_checkpoint)
    model.set_logger(configure(LOG_DIR, ["stdout", "tensorboard"]))

    print(f"tensorboard: tensorboard --logdir {LOG_DIR}")

    eval_env = make_vec_env(make_env(), n_envs=1)
    callbacks = CallbackList([
        TqdmCallback(total_timesteps=TOTAL_TIMESTEPS),
        CheckpointCallback(
            save_freq=CHECKPOINT_FREQ,
            save_path=CHECKPOINT_DIR,
            name_prefix="echo",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=BEST_MODEL_PATH,
            log_path=LOG_DIR,
            eval_freq=EVAL_FREQ,
            n_eval_episodes=EVAL_EPISODES,
            deterministic=True,
        ),
        EchoTensorBoardCallback(),
        MovingPeopleCallback(enable_after=MOVING_PEOPLE_AFTER, verbose=1),
    ])

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        reset_num_timesteps=resume_checkpoint is None,
        progress_bar=False,
    )

    final_path = f"{CHECKPOINT_DIR}echo_final"
    model.save(final_path)
    print(f"training complete — saved to {final_path}")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    train()
