# src/agent/train.py
import re
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.logger import configure
from src.env.echo_env import EchoEnv
from src.sim.room import Room, Object, Person
import os

# --- training config ---
TOTAL_TIMESTEPS  = 100_000
N_ENVS           = 4          # parallel envs, speeds up data collection
MAX_STEPS        = 50         # steps per episode
CHECKPOINT_FREQ  = 5_000      # save model every N steps
EVAL_FREQ        = 2_500      # evaluate every N steps
EVAL_EPISODES    = 10         # episodes per evaluation
DEFAULT_RESUME_CHECKPOINT = "checkpoints/phase_1_120000_steps.zip"
EPISODE_MEAN_WINDOW = 10      # episodes averaged for metrics + phase advancement

# --- PPO hyperparams ---
LEARNING_RATE    = 3e-4
N_STEPS          = 64       # steps per PPO update (64 * N_ENVS must divide BATCH_SIZE)
BATCH_SIZE       = 64
N_EPOCHS         = 10
GAMMA            = 0.99       # discount factor
GAE_LAMBDA       = 0.95       # generalized advantage estimation
CLIP_RANGE       = 0.2        # PPO clip

# --- curriculum thresholds ---
# mean reward needed to unlock next phase (kept low — phases differ by room complexity)
PHASE_THRESHOLDS = [0.1, 0.2, 0.3]

# --- paths ---
LOG_DIR          = "runs/ppo_echo"
CHECKPOINT_DIR   = "checkpoints/"
BEST_MODEL_PATH  = "checkpoints/best_model"

CHECKPOINT_STEP_PATTERN = re.compile(r"phase_(\d+)_(\d+)_steps\.zip$")
CHECKPOINT_MODEL_PATTERN = re.compile(r"phase_(\d+)_model\.zip$")

def make_env(phase: int = 1):
    """
    Factory function that returns an env constructor for a given curriculum phase.
    stable-baselines3 needs a callable, not an env instance directly.
    """
    def _init():
        room = Room(width=10.0, height=8.0)

        # scale complexity with phase
        max_objects = phase * 2
        max_people = phase

        n_objects = np.random.randint(0, max_objects + 1)
        n_people = np.random.randint(0, max_people + 1)

        for i in range(n_objects):
            room.add_object(Object(
                position=[
                    np.random.uniform(1, 9),
                    np.random.uniform(1, 7),
                ],
                width=np.random.uniform(0.3, 2.0),
                height=np.random.uniform(0.3, 2.0),
                label=f"object_{i}",
            ))

        for i in range(n_people):
            room.add_person(Person(
                position=[
                    np.random.uniform(1, 9),
                    np.random.uniform(1, 7),
                ],
                facing=np.random.uniform(0, 2 * np.pi),
                label=f"person_{i}",
            ))

        return EchoEnv(room=room, max_steps=MAX_STEPS)

    return _init

class EchoTensorBoardCallback(BaseCallback):
    """
    Logs custom metrics to TensorBoard at each step.
    Run tensorboard --logdir runs/ to watch live.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_coverages = []
        self.current_episode_reward = 0.0

    def _on_step(self) -> bool:
        # accumulate reward
        self.current_episode_reward += float(self.locals["rewards"][0])

        # pull info dict from the env
        info = self.locals["infos"][0]

        # log per-step metrics
        self.logger.record("echo/map_coverage", info.get("map_coverage", 0))
        self.logger.record("echo/f_start", info.get("f_start", 0))
        self.logger.record("echo/f_end", info.get("f_end", 0))
        self.logger.record("echo/step", info.get("step", 0))

        # log episode reward on episode end
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


def parse_checkpoint(path: str) -> tuple[int, int | None]:
    """Return (phase, steps) parsed from a checkpoint filename."""
    name = os.path.basename(path)
    step_match = CHECKPOINT_STEP_PATTERN.match(name)
    if step_match:
        return int(step_match.group(1)), int(step_match.group(2))

    model_match = CHECKPOINT_MODEL_PATTERN.match(name)
    if model_match:
        return int(model_match.group(1)), None

    raise ValueError(f"unrecognized checkpoint filename: {name}")


def find_latest_checkpoint(checkpoint_dir: str = CHECKPOINT_DIR) -> str | None:
    """Find the newest periodic checkpoint in the checkpoint directory."""
    if not os.path.isdir(checkpoint_dir):
        return None

    latest_path = None
    latest_steps = -1

    for name in os.listdir(checkpoint_dir):
        match = CHECKPOINT_STEP_PATTERN.match(name)
        if not match:
            continue
        steps = int(match.group(2))
        if steps > latest_steps:
            latest_steps = steps
            latest_path = os.path.join(checkpoint_dir, name)

    return latest_path


def resolve_resume_checkpoint(resume_from: str | None) -> str | None:
    if resume_from == "":
        return None
    if resume_from:
        path = resume_from
    elif os.path.exists(DEFAULT_RESUME_CHECKPOINT):
        path = DEFAULT_RESUME_CHECKPOINT
    else:
        path = find_latest_checkpoint()

    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return path


def load_or_create_model(vec_env, current_phase: int, checkpoint_path: str | None):
    model_path = f"{CHECKPOINT_DIR}phase_{current_phase}_model"

    if checkpoint_path:
        print(f"loading checkpoint: {checkpoint_path}")
        return PPO.load(checkpoint_path, env=vec_env)

    if os.path.exists(f"{model_path}.zip"):
        print(f"loading phase {current_phase} weights...")
        return PPO.load(model_path, env=vec_env)

    if current_phase > 1 and os.path.exists(f"{CHECKPOINT_DIR}phase_{current_phase - 1}_model.zip"):
        print(f"loading phase {current_phase - 1} weights...")
        return PPO.load(f"{CHECKPOINT_DIR}phase_{current_phase - 1}_model", env=vec_env)

    print(f"starting fresh model for phase {current_phase}")
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


def train(resume_from: str | None = DEFAULT_RESUME_CHECKPOINT):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    resume_checkpoint = resolve_resume_checkpoint(resume_from)
    current_phase = 1
    pending_checkpoint = None

    if resume_checkpoint:
        current_phase, _ = parse_checkpoint(resume_checkpoint)
        pending_checkpoint = resume_checkpoint
        print(f"resuming from phase {current_phase}: {resume_checkpoint}")

    total_steps_trained = 0

    while current_phase <= 4:
        print(f"\n--- phase {current_phase} ---")

        vec_env = make_vec_env(
            make_env(phase=current_phase),
            n_envs=N_ENVS,
        )

        model = load_or_create_model(vec_env, current_phase, pending_checkpoint)
        pending_checkpoint = None
        model.set_logger(configure(LOG_DIR, ["tensorboard"]))

        checkpoint_cb = CheckpointCallback(
            save_freq=CHECKPOINT_FREQ,
            save_path=CHECKPOINT_DIR,
            name_prefix=f"phase_{current_phase}",
        )

        eval_env = make_vec_env(make_env(phase=current_phase), n_envs=1)
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=BEST_MODEL_PATH,
            log_path=LOG_DIR,
            eval_freq=EVAL_FREQ,
            n_eval_episodes=EVAL_EPISODES,
            deterministic=True,
        )

        tensorboard_cb = EchoTensorBoardCallback()

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_cb, eval_cb, tensorboard_cb],
            reset_num_timesteps=False,
        )

        total_steps_trained += TOTAL_TIMESTEPS

        model_path = f"{CHECKPOINT_DIR}phase_{current_phase}_model"
        model.save(model_path)
        print(f"phase {current_phase} saved to {model_path}")

        if len(tensorboard_cb.episode_rewards) >= EPISODE_MEAN_WINDOW:
            mean_reward = np.mean(tensorboard_cb.episode_rewards[-EPISODE_MEAN_WINDOW:])
            threshold = PHASE_THRESHOLDS[current_phase - 1] if current_phase <= len(PHASE_THRESHOLDS) else 0
            print(f"mean reward: {mean_reward:.3f} | threshold: {threshold:.3f}")
            if mean_reward >= threshold or current_phase == 4:
                current_phase += 1
            else:
                print(f"threshold not met, retraining phase {current_phase}")
        else:
            current_phase += 1

        vec_env.close()
        eval_env.close()

    print(f"\ntraining complete. total steps: {total_steps_trained}")


if __name__ == "__main__":
    train()
    