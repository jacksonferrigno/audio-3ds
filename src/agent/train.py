# src/agent/train.py
from pathlib import Path

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
from src.env.echo_env import DEFAULT_MAX_STEPS, EchoEnv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- training config ---
TOTAL_TIMESTEPS = 100_000
MOVING_PEOPLE_AFTER: int | None = None  # off for now; set e.g. 50_000 to re-enable
N_ENVS = 4
MAX_STEPS = DEFAULT_MAX_STEPS
CHECKPOINT_FREQ = 5_000
EVAL_FREQ = 2_500
EVAL_EPISODES = 10
DEFAULT_RESUME_CHECKPOINT = PROJECT_ROOT / "checkpoints/best_model/best_model.zip"
LEGACY_CHECKPOINT_DIR = PROJECT_ROOT / "src/checkpoints"
EPISODE_MEAN_WINDOW = 10

# --- PPO hyperparams ---
LEARNING_RATE = 3e-4
N_STEPS = 64
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2

LOG_DIR = PROJECT_ROOT / "runs/ppo_echo"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
BEST_MODEL_PATH = PROJECT_ROOT / "checkpoints/best_model"


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

    def __init__(self, enable_after: int | None = MOVING_PEOPLE_AFTER, verbose: int = 0):
        super().__init__(verbose)
        self.enable_after = enable_after
        self._enabled = False

    def _on_step(self) -> bool:
        if self.enable_after is None:
            return True
        if not self._enabled and self.num_timesteps >= self.enable_after:
            self.training_env.env_method("set_moving_people", True)
            self._enabled = True
            if self.verbose:
                print(f"moving people enabled at step {self.num_timesteps}")
        return True


def _latest_step_checkpoint(checkpoint_dir: Path) -> Path | None:
    latest_path: Path | None = None
    latest_steps = -1

    if not checkpoint_dir.is_dir():
        return None

    for path in checkpoint_dir.glob("*_steps.zip"):
        try:
            steps = int(path.stem.removesuffix("_steps").rsplit("_", 1)[-1])
        except ValueError:
            continue
        if steps > latest_steps:
            latest_steps = steps
            latest_path = path

    return latest_path


def resolve_resume_checkpoint(resume_from: str | None) -> str | None:
    if resume_from == "":
        return None

    if resume_from is not None:
        path = Path(resume_from)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {resume_from}")
        return str(path)

    if DEFAULT_RESUME_CHECKPOINT.exists():
        return str(DEFAULT_RESUME_CHECKPOINT)

    latest = _latest_step_checkpoint(CHECKPOINT_DIR)
    if latest is not None:
        return str(latest)

    legacy_latest = _latest_step_checkpoint(LEGACY_CHECKPOINT_DIR)
    if legacy_latest is not None:
        print(f"found legacy checkpoint: {legacy_latest}")
        return str(legacy_latest)

    return None


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
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_PATH.mkdir(parents=True, exist_ok=True)

    resume_checkpoint = resolve_resume_checkpoint(resume_from)
    if resume_checkpoint:
        print(f"resuming from: {resume_checkpoint}")
    else:
        print("no checkpoint found — starting fresh")
    print(f"checkpoints save to: {CHECKPOINT_DIR.resolve()}")
    if MOVING_PEOPLE_AFTER is None:
        print("moving people: disabled")
    else:
        print(f"moving people: enabled after step {MOVING_PEOPLE_AFTER}")

    vec_env = make_vec_env(make_env(), n_envs=N_ENVS)
    model = load_or_create_model(vec_env, resume_checkpoint)
    model.set_logger(configure(str(LOG_DIR), ["stdout", "tensorboard"]))

    remaining_timesteps = TOTAL_TIMESTEPS
    if resume_checkpoint:
        remaining_timesteps = max(0, TOTAL_TIMESTEPS - model.num_timesteps)
        print(f"continuing for {remaining_timesteps} steps (at {model.num_timesteps}/{TOTAL_TIMESTEPS})")

    print(f"tensorboard: tensorboard --logdir {LOG_DIR.resolve()}")

    eval_env = make_vec_env(make_env(), n_envs=1)
    callbacks = CallbackList([
        TqdmCallback(total_timesteps=remaining_timesteps),
        CheckpointCallback(
            # save_freq counts env.step() calls; divide by N_ENVS for real timesteps
            save_freq=max(1, CHECKPOINT_FREQ // N_ENVS),
            save_path=str(CHECKPOINT_DIR),
            name_prefix="echo",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(BEST_MODEL_PATH),
            log_path=str(LOG_DIR / "evaluations"),
            eval_freq=max(1, EVAL_FREQ // N_ENVS),
            n_eval_episodes=EVAL_EPISODES,
            deterministic=True,
        ),
        EchoTensorBoardCallback(),
        MovingPeopleCallback(enable_after=MOVING_PEOPLE_AFTER, verbose=1),
    ])

    model.learn(
        total_timesteps=remaining_timesteps,
        callback=callbacks,
        reset_num_timesteps=resume_checkpoint is None,
        progress_bar=False,
    )

    final_path = CHECKPOINT_DIR / "echo_final"
    model.save(str(final_path))
    print(f"training complete — saved to {final_path.resolve()}")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    train()
