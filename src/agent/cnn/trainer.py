import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.agent.cnn.cache import ensure_dataset_cache
from src.agent.cnn.config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CACHE_DIR,
    CHECKPOINT_DIR,
    DEFAULT_GEN_WORKERS,
    DEFAULT_RL_CHECKPOINT,
    DATALOADER_WORKERS,
    EPOCHS,
    LATEST_MODEL_PATH,
    LEARNING_RATE,
    N_TRAIN_SAMPLES,
    N_VAL_SAMPLES,
)
from src.agent.cnn.dataset import CachedEchoMapDataset
from src.agent.cnn.metrics import evaluate, pixel_accuracy
from src.agent.cnn.timefmt import fmt_seconds
from src.agent.cnn_model import OccupancyMapCNN, WeightedCrossEntropyLoss, count_parameters


def train_cnn(
    n_train: int = N_TRAIN_SAMPLES,
    n_val: int = N_VAL_SAMPLES,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    rl_checkpoint: str = DEFAULT_RL_CHECKPOINT,
    gen_workers: int | None = None,
    regenerate_cache: bool = False,
    device: str | None = None,
) -> OccupancyMapCNN:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    if gen_workers is None:
        gen_workers = DEFAULT_GEN_WORKERS

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    (train_occ, train_lab), (val_occ, val_lab) = ensure_dataset_cache(
        n_train=n_train,
        n_val=n_val,
        rl_checkpoint=rl_checkpoint,
        gen_workers=gen_workers,
        force=regenerate_cache,
    )

    train_loader = DataLoader(
        CachedEchoMapDataset(train_occ, train_lab, augment=True, seed=0),
        batch_size=batch_size,
        shuffle=True,
        num_workers=DATALOADER_WORKERS,
    )
    val_loader = DataLoader(
        CachedEchoMapDataset(val_occ, val_lab, augment=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=DATALOADER_WORKERS,
    )

    model = OccupancyMapCNN().to(torch_device)
    loss_fn = WeightedCrossEntropyLoss().to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"device: {torch_device}")
    print(f"parameters: {count_parameters(model):,}")
    print(f"train samples: {n_train} | val samples: {n_val}")
    print(f"cache: {CACHE_DIR}")

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

    print(f"\nCNN training complete in {fmt_seconds(time.time() - start)}")
    print(f"best val loss: {best_val_loss:.4f}")
    return model
