import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sim.room import Room

MAP_ROWS = 80
MAP_COLS = 100
MAP_RESOLUTION = 0.1
NUM_CLASSES = 6

# semantic class indices
EMPTY = 0
WALL = 1
FURNITURE = 2
PERSON_TORSO = 3
PERSON_HEAD = 4
UNKNOWN = 5

CLASS_NAMES = ("empty", "wall", "furniture", "person_torso", "person_head", "unknown")

# upweight rare classes — most cells are empty
DEFAULT_CLASS_WEIGHTS = torch.tensor(
    [0.5, 2.0, 3.0, 8.0, 12.0, 1.0],
    dtype=torch.float32,
)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class OccupancyMapCNN(nn.Module):
    """
    Fully convolutional semantic segmenter for echo occupancy maps.

    Input:  (B, 1, 80, 100) occupancy grid
    Output: (B, 6, 80, 100) per-cell class logits
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(1, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
        )
        self.decoder = nn.Sequential(
            ConvBlock(64, 32),
            ConvBlock(32, 16),
        )
        self.head = nn.Conv2d(16, NUM_CLASSES, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return self.head(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return hard class labels with shape (B, H, W)."""
        logits = self.forward(x)
        return logits.argmax(dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities with shape (B, 6, H, W)."""
        return F.softmax(self.forward(x), dim=1)


class WeightedCrossEntropyLoss(nn.Module):
    """Per-cell cross entropy with class weights for imbalance."""

    def __init__(self, class_weights: torch.Tensor | None = None):
        super().__init__()
        weights = class_weights if class_weights is not None else DEFAULT_CLASS_WEIGHTS
        self.register_buffer("class_weights", weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.class_weights)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def occupancy_to_tensor(occupancy_map: np.ndarray) -> torch.Tensor:
    """Convert (H, W) or (B, H, W) occupancy map to model input tensor."""
    arr = np.asarray(occupancy_map, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis, np.newaxis, ...]
    elif arr.ndim == 3:
        arr = arr[:, np.newaxis, ...]
    else:
        raise ValueError(f"expected shape (H, W) or (B, H, W), got {arr.shape}")
    return torch.from_numpy(arr)


def _fill_rect(
    labels: np.ndarray,
    cx: float,
    cy: float,
    half_w: float,
    half_h: float,
    label: int,
    resolution: float = MAP_RESOLUTION,
) -> None:
    x0, x1 = cx - half_w, cx + half_w
    y0, y1 = cy - half_h, cy + half_h

    col_start = max(int(x0 / resolution), 0)
    col_end = min(int(np.ceil(x1 / resolution)), labels.shape[1])
    row_start = max(int(y0 / resolution), 0)
    row_end = min(int(np.ceil(y1 / resolution)), labels.shape[0])

    if col_start >= col_end or row_start >= row_end:
        return

    labels[row_start:row_end, col_start:col_end] = label


def build_label_map(
    room: Room,
    resolution: float = MAP_RESOLUTION,
) -> np.ndarray:
    """
    Rasterize room geometry into per-cell semantic labels.

    Returns int array with shape (rows, cols) == (80, 100) for the default room.
    """
    cols = int(room.width / resolution)
    rows = int(room.height / resolution)
    labels = np.full((rows, cols), EMPTY, dtype=np.int64)

    labels[0, :] = WALL
    labels[-1, :] = WALL
    labels[:, 0] = WALL
    labels[:, -1] = WALL

    for obj in room.objects:
        _fill_rect(
            labels,
            obj.position[0],
            obj.position[1],
            obj.width / 2,
            obj.height / 2,
            FURNITURE,
            resolution,
        )

    for person in room.people:
        _fill_rect(
            labels,
            person.position[0],
            person.position[1],
            person.body_width / 2,
            person.body_height / 2,
            PERSON_TORSO,
            resolution,
        )
        head_x, head_y = person.head_position
        _fill_rect(
            labels,
            head_x,
            head_y,
            person.head_radius,
            person.head_radius,
            PERSON_HEAD,
            resolution,
        )

    return labels


if __name__ == "__main__":
    from src.sim.room import make_default_room

    model = OccupancyMapCNN()
    loss_fn = WeightedCrossEntropyLoss()
    print(f"parameters: {count_parameters(model):,}")

    batch = torch.randn(2, 1, MAP_ROWS, MAP_COLS)
    logits = model(batch)
    assert logits.shape == (2, NUM_CLASSES, MAP_ROWS, MAP_COLS)

    room = make_default_room()
    labels = build_label_map(room)
    assert labels.shape == (MAP_ROWS, MAP_COLS)

    targets = torch.from_numpy(labels).unsqueeze(0).expand(2, -1, -1)
    loss = loss_fn(logits, targets)
    preds = model.predict(batch)
    print(f"loss: {loss.item():.4f}")
    print(f"pred shape: {tuple(preds.shape)}")
    print(f"label counts: { {CLASS_NAMES[i]: int((labels == i).sum()) for i in range(NUM_CLASSES)} }")
