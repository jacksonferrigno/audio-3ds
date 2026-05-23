import numpy as np


def augment_sample(
    occupancy: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
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
