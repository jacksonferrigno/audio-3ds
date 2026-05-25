import numpy as np
import torch
from torch.utils.data import Dataset

from src.agent.cnn.augment import augment_sample


class CachedEchoMapDataset(Dataset):
    """Reads pre-generated RL occupancy maps from disk."""

    def __init__(
        self,
        occupancy_path: str,
        labels_path: str,
        augment: bool = False,
        seed: int = 0,
    ):
        self.occupancy = np.load(occupancy_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.occupancy)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        occupancy = np.array(self.occupancy[index], dtype=np.float32)
        labels = np.array(self.labels[index], copy=True)

        if self.augment:
            rng = np.random.default_rng(self.seed + index)
            occupancy, labels = augment_sample(occupancy, labels, rng)

        x = torch.from_numpy(np.ascontiguousarray(occupancy)).unsqueeze(0)
        y = torch.from_numpy(np.ascontiguousarray(labels))
        return x, y
