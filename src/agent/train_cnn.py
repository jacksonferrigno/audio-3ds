import os
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.agent.cnn_model import (
    CLASS_NAMES,
    MAP_COLS,
    MAP_ROWS,
    OccupancyMapCNN,
    WeightedCrossEntropyLoss,
    build_label_map,
    count_parameters,
)
from src.env.echo_env import EchoEnv
from src.sim.room import Object, Person, Room

# --- config ---
N_TRAIN_SAMPLES = 10_000
N_VAL_SAMPLES = 500
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-3
STEPS_PER_SAMPLE = 50
MAX_OBJECTS = 8
MAX_PEOPLE = 4
NUM_WORKERS = 0

DEFAULT_RL_CHECKPOINT = "checkpoints/best_model/best_model.zip"
CHECKPOINT_DIR = "checkpoints/cnn"
BEST_MODEL_PATH = f"{CHECKPOINT_DIR}/best_model.pt"
LATEST_MODEL_PATH = f"{CHECKPOINT_DIR}/latest_model.pt"


def make_random_room(
    max_objects: int = MAX_OBJECTS,
    max_people: int = MAX_PEOPLE,
    width: float = 10.0,
    height: float = 8.0,
    rng: np.random.Generator | None = None,
) -> Room:
    rng = rng or np.random.default_rng()
    room = Room(width=width, height=height)

    n_objects = int(rng.integers(0, max_objects + 1))
    n_people = int(rng.integers(0, max_people + 1))

    for i in range(n_objects):
        room.add_object(
            Object(
                position=[
                    float(rng.uniform(1, width - 1)),
                    float(rng.uniform(1, height - 1)),
                ],
                width=float(rng.uniform(0.3, 2.0)),
                height=float(rng.uniform(0.3, 2.0)),
                label=f"object_{i}",
            )
        )

    for i in range(n_people):
        room.add_person(
            Person(
                position=[
                    float(rng.uniform(1, width - 1)),
                    float(rng.uniform(1, height - 1)),
                ],
                facing=float(rng.uniform(0, 2 * np.pi)),
                label=f"person_{i}",
            )
        )

    return room


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
) -> np.ndarray:
    """Run the trained RL agent in a room and return its accumulated occupancy map."""
    env = EchoEnv(room=room, max_steps=n_steps + 5)
    obs, _ = env.reset(seed=seed)

    for _ in range(n_steps):
        action, _ = rl_model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    return env.occupancy_map.copy()


def augment_sample(
    occupancy: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Random flips, rotations, and noise to improve generalization."""
    occ = occupancy.copy()
    lab = labels.copy()

    if rng.random() < 0.5:
        occ = np.flip(occ, axis=1)
        lab = np.flip(lab, axis=1)

    if rng.random() < 0.5:
        occ = np.flip(occ, axis=0)
        lab = np.flip(lab, axis=0)

    # 180° only — 90/270 swap (80, 100) → (100, 80) on our non-square grid
    if rng.random() < 0.5:
        occ = np.rot90(occ, k=2)
        lab = np.rot90(lab, k=2)

    noise = rng.normal(0.0, 0.02, size=occ.shape).astype(np.float32)
    occ = np.clip(occ + noise, 0.0, 1.0)

    return occ.astype(np.float32), lab


class EchoMapDataset(Dataset):
    """Dataset of RL-agent-built occupancy maps paired with geometric labels."""

    def __init__(
        self,
        size: int,
        rl_model: PPO,
        n_steps: int = STEPS_PER_SAMPLE,
        seed: int = 0,
        augment: bool = False,
    ):
        self.size = size
        self.rl_model = rl_model
        self.n_steps = n_steps
        self.seed = seed
        self.augment = augment

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_seed = self.seed + index
        rng = np.random.default_rng(sample_seed)

        room = make_random_room(
            max_objects=MAX_OBJECTS,
            max_people=MAX_PEOPLE,
            rng=rng,
        )
        occupancy = generate_occupancy_map_with_agent(
            room,
            self.rl_model,
            n_steps=self.n_steps,
            seed=sample_seed,
        )
        labels = build_label_map(room)

        if self.augment:
            occupancy, labels = augment_sample(occupancy, labels, rng)

        if occupancy.shape != (MAP_ROWS, MAP_COLS):
            raise ValueError(f"unexpected occupancy shape: {occupancy.shape}")
        if labels.shape != (MAP_ROWS, MAP_COLS):
            raise ValueError(f"unexpected label shape: {labels.shape}")

        x = torch.from_numpy(occupancy).unsqueeze(0)
        y = torch.from_numpy(labels.astype(np.int64))
        return x, y


def pixel_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == targets).float().mean().item())


def class_accuracies(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    preds = logits.argmax(dim=1)
    stats: dict[str, float] = {}

    for class_idx, class_name in enumerate(CLASS_NAMES):
        mask = targets == class_idx
        if int(mask.sum().item()) == 0:
            continue
        stats[class_name] = float((preds[mask] == targets[mask]).float().mean().item())

    return stats


@torch.no_grad()
def evaluate(
    model: OccupancyMapCNN,
    loader: DataLoader,
    loss_fn: WeightedCrossEntropyLoss,
    device: torch.device,
) -> tuple[float, float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    merged_class_acc: dict[str, list[float]] = {name: [] for name in CLASS_NAMES}

    for x, y in tqdm(loader, desc="val", leave=False):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        total_loss += float(loss_fn(logits, y).item())
        total_acc += pixel_accuracy(logits, y)

        for name, acc in class_accuracies(logits, y).items():
            merged_class_acc[name].append(acc)

    n_batches = max(len(loader), 1)
    mean_class_acc = {
        name: float(np.mean(values))
        for name, values in merged_class_acc.items()
        if values
    }
    return total_loss / n_batches, total_acc / n_batches, mean_class_acc


def train_cnn(
    n_train: int = N_TRAIN_SAMPLES,
    n_val: int = N_VAL_SAMPLES,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    rl_checkpoint: str = DEFAULT_RL_CHECKPOINT,
    device: str | None = None,
) -> OccupancyMapCNN:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"loading RL agent from {rl_checkpoint}")
    rl_model = load_rl_agent(rl_checkpoint)

    train_ds = EchoMapDataset(
        size=n_train,
        rl_model=rl_model,
        seed=0,
        augment=True,
    )
    val_ds = EchoMapDataset(
        size=n_val,
        rl_model=rl_model,
        seed=100_000,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = OccupancyMapCNN().to(torch_device)
    loss_fn = WeightedCrossEntropyLoss().to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"device: {torch_device}")
    print(f"parameters: {count_parameters(model):,}")
    print(f"train samples: {n_train} | val samples: {n_val}")
    print(f"maps generated by RL agent ({STEPS_PER_SAMPLE} steps/room)")

    best_val_loss = float("inf")
    start = time.time()

    epoch_bar = tqdm(range(1, epochs + 1), desc="epochs", unit="epoch")

    for epoch in epoch_bar:
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        batch_bar = tqdm(
            train_loader,
            desc=f"train {epoch}/{epochs}",
            leave=False,
            unit="batch",
        )
        for x, y in batch_bar:
            x = x.to(torch_device)
            y = y.to(torch_device)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            batch_loss = float(loss.item())
            batch_acc = pixel_accuracy(logits, y)
            train_loss += batch_loss
            train_acc += batch_acc
            batch_bar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_acc:.3f}")

        n_train_batches = max(len(train_loader), 1)
        train_loss /= n_train_batches
        train_acc /= n_train_batches

        val_loss, val_acc, val_class_acc = evaluate(
            model, val_loader, loss_fn, torch_device
        )

        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_acc:.3f}",
        )
        if val_class_acc:
            top_classes = ", ".join(
                f"{name} {acc:.2f}"
                for name, acc in sorted(val_class_acc.items(), key=lambda item: -item[1])[:3]
            )
            tqdm.write(f"  val class acc: {top_classes}")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
            LATEST_MODEL_PATH,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            tqdm.write(f"  saved best model -> {BEST_MODEL_PATH}")

    elapsed = time.time() - start
    print(f"\ntraining complete in {elapsed:.1f}s")
    print(f"best val loss: {best_val_loss:.4f}")
    return model


if __name__ == "__main__":
    train_cnn()
