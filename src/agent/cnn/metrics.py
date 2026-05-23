import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.agent.cnn_model import CLASS_NAMES, OccupancyMapCNN, WeightedCrossEntropyLoss


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
        name: float(sum(values) / len(values))
        for name, values in merged_class_acc.items()
        if values
    }
    return total_loss / n_batches, total_acc / n_batches, mean_class_acc
