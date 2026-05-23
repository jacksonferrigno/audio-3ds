import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from stable_baselines3 import PPO
from tqdm import tqdm

from src.agent.cnn.config import (
    CACHE_DIR,
    MAX_OBJECTS,
    MAX_PEOPLE,
    STEPS_PER_SAMPLE,
)
from src.agent.cnn.rooms import make_random_room
from src.agent.cnn.rollout import generate_occupancy_map_with_agent
from src.agent.cnn.timefmt import fmt_seconds
from src.agent.cnn_model import MAP_COLS, MAP_ROWS, build_label_map

_RL_MODEL: PPO | None = None


def _init_cache_worker(rl_checkpoint: str) -> None:
    global _RL_MODEL
    _RL_MODEL = PPO.load(rl_checkpoint)


def _generate_cache_sample(sample_seed: int, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    if _RL_MODEL is None:
        raise RuntimeError("cache worker RL model not initialized")

    rng = np.random.default_rng(sample_seed)
    room = make_random_room(rng=rng)
    occupancy = generate_occupancy_map_with_agent(
        room,
        _RL_MODEL,
        n_steps=n_steps,
        seed=sample_seed,
    )
    labels = build_label_map(room)
    return occupancy.astype(np.float32), labels.astype(np.int64)


def _split_paths(split: str) -> tuple[str, str, str, str]:
    split_dir = os.path.join(CACHE_DIR, split)
    return (
        split_dir,
        os.path.join(split_dir, "occupancy.npy"),
        os.path.join(split_dir, "labels.npy"),
        os.path.join(split_dir, "meta.json"),
    )


def _cache_meta(
    n_samples: int,
    seed_start: int,
    n_steps: int,
    rl_checkpoint: str,
) -> dict:
    return {
        "n_samples": n_samples,
        "seed_start": seed_start,
        "n_steps": n_steps,
        "rl_checkpoint": os.path.abspath(rl_checkpoint),
        "max_objects": MAX_OBJECTS,
        "max_people": MAX_PEOPLE,
        "map_rows": MAP_ROWS,
        "map_cols": MAP_COLS,
    }


def cache_is_valid(meta_path: str, expected: dict, occ_path: str, lab_path: str) -> bool:
    if not (os.path.exists(meta_path) and os.path.exists(occ_path) and os.path.exists(lab_path)):
        return False

    with open(meta_path, encoding="utf-8") as f:
        saved = json.load(f)

    if saved != expected:
        return False

    occ = np.load(occ_path, mmap_mode="r")
    lab = np.load(lab_path, mmap_mode="r")
    return (
        occ.shape == (expected["n_samples"], MAP_ROWS, MAP_COLS)
        and lab.shape == (expected["n_samples"], MAP_ROWS, MAP_COLS)
    )


def build_split_cache(
    split: str,
    n_samples: int,
    seed_start: int,
    rl_checkpoint: str,
    n_steps: int = STEPS_PER_SAMPLE,
    gen_workers: int = 1,
    force: bool = False,
) -> tuple[str, str]:
    split_dir, occ_path, lab_path, meta_path = _split_paths(split)
    expected = _cache_meta(n_samples, seed_start, n_steps, rl_checkpoint)

    if not force and cache_is_valid(meta_path, expected, occ_path, lab_path):
        tqdm.write(f"using cached {split} dataset ({n_samples} samples) -> {split_dir}")
        return occ_path, lab_path

    os.makedirs(split_dir, exist_ok=True)
    tqdm.write(
        f"generating {split} cache: {n_samples} rooms x {n_steps} RL steps "
        f"({gen_workers} workers). This runs once, then training is fast."
    )

    occupancy = np.lib.format.open_memmap(
        occ_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_samples, MAP_ROWS, MAP_COLS),
    )
    labels = np.lib.format.open_memmap(
        lab_path,
        mode="w+",
        dtype=np.int64,
        shape=(n_samples, MAP_ROWS, MAP_COLS),
    )

    seeds = [seed_start + i for i in range(n_samples)]
    start = time.time()

    with ProcessPoolExecutor(
        max_workers=gen_workers,
        initializer=_init_cache_worker,
        initargs=(rl_checkpoint,),
    ) as pool:
        futures = {
            pool.submit(_generate_cache_sample, seed, n_steps): idx
            for idx, seed in enumerate(seeds)
        }

        progress = tqdm(total=n_samples, desc=f"generate {split}", unit="room")
        for future in as_completed(futures):
            idx = futures[future]
            occ, lab = future.result()
            occupancy[idx] = occ
            labels[idx] = lab
            progress.update(1)
            elapsed = time.time() - start
            done = progress.n
            if done:
                eta = elapsed / done * (n_samples - done)
                progress.set_postfix(elapsed=fmt_seconds(elapsed), eta=fmt_seconds(eta))
        progress.close()

    occupancy.flush()
    labels.flush()

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(expected, f, indent=2)

    tqdm.write(f"{split} cache ready in {fmt_seconds(time.time() - start)} -> {split_dir}")
    return occ_path, lab_path


def ensure_dataset_cache(
    n_train: int,
    n_val: int,
    rl_checkpoint: str,
    n_steps: int = STEPS_PER_SAMPLE,
    gen_workers: int = 1,
    force: bool = False,
) -> tuple[tuple[str, str], tuple[str, str]]:
    train_paths = build_split_cache(
        split="train",
        n_samples=n_train,
        seed_start=0,
        rl_checkpoint=rl_checkpoint,
        n_steps=n_steps,
        gen_workers=gen_workers,
        force=force,
    )
    val_paths = build_split_cache(
        split="val",
        n_samples=n_val,
        seed_start=100_000,
        rl_checkpoint=rl_checkpoint,
        n_steps=n_steps,
        gen_workers=gen_workers,
        force=force,
    )
    return train_paths, val_paths
