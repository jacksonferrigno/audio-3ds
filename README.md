# audio-3ds

*models are currently training* 

## Fun physics about this project

Sounds propagtes as a pressure wave - compressed air molecules pushing into neighboring molecultes, radiating outwards in all directions from the source. When this wave hits a surface, it tells us about the object. Hard flat surfaces reflect cleanly. Soft material absorbs more.

Rather than simulating every air molecule, we approx this pressure as a bundle of 360 rays emitted simultaneously in all directions -- pulse. Each ray travels in a straight line, bouncing off surfaces and losing energy at each bounce based on material absorption. either a ray comes back or it doesnt. 

## RL

---

### Action Space

The agent's job is to find what it hasn't seen yet. At each step it controls four continuous values, all normalized 0-1 and scaled to real physical values internally:


| Action      | Range         | What it does             |
| ----------- | ------------- | ------------------------ |
| `f_start`   | 500–4000 Hz   | Chirp start frequency    |
| `f_end`     | 4000–16000 Hz | Chirp end frequency      |
| `direction` | 0–2π          | Where to point the sweep |
| `n_rays`    | 30–360        | How wide to fan out      |


Cranking up the frequency range means looking harder for detail — high frequency chirps resolve finer geometry at shorter range. Early in an episode the agent tends toward wide low-frequency sweeps to build a rough map. Later it narrows in on fuzzy areas with high frequency focused pulses. That transition is learned, not programmed.

### Observation Space

The agent sees two things concatenated into a single vector each step:

**Frequency features** — a 64-dimensional compressed fingerprint of this chirp's returning echoes. The power spectrum of what came back from this specific pulse, bucketed into 64 frequency bins and normalized.

**Occupancy map** — the agent's accumulated belief about the room flattened into a 1D vector. Every cell is a probability between 0 and 1 — how likely is something physically present here. This is everything the agent thinks it knows so far, built up from all previous steps in the episode.

```
observation = [features (64) | occupancy_map flattened (8000)] = 8064-dim vector
```

---

### Reward Function

The reward signal is designed around one core idea: be efficient, explore intelligently, don't waste chirps.

Four components combine each step:

**1. Information gain** `+2.0 weight`
How much did the occupancy map actually change this step. Measured as the fraction of cells that shifted meaningfully (delta > 0.05) between the previous map and the current one. This is the dominant signal — the agent gets rewarded for learning things it didn't already know.

**2. Redundancy penalty** `-0.3 weight`
Cells that are already certain (close to 0 or 1) contribute to an average certainty score. Firing into areas the agent already understands well gets penalized. This pushes exploration toward uncertain regions.

**3. Coverage bonus** `+0.5 weight`
Steady reward for the fraction of the room that has been mapped above the noise floor. Encourages the agent to keep expanding its map rather than refining one small area.

**4. Step penalty** `-0.01 per step`
A small constant penalty every step. The agent should map the room as fast as possible — not wander.

```
reward = (information_gain × 2.0) 
       - (avg_certainty × 0.3) 
       + (coverage × 0.5) 
       - 0.01
```

A random agent scores around `-0.26` per step. A trained agent should push toward `+0.5` per step by learning to prioritize unexplored regions with targeted high-frequency chirps.

---

## CNN

The RL occupancy map is raw signal — a fine-grained accumulation of what echoes returned, cell by cell. It tells you *something* is probably here, but not *what* it is. The CNN is the translation layer: it reads that map and produces a semantic visual the viewer can actually understand.

For people watching the software run, this is what turns the agent's internal belief into walls, furniture, and people on screen.

### Input

The occupancy map the RL agent built over an episode — shape `(1, 80, 100)`.

Single channel, same 10cm grid as the RL env. Each cell is a probability between 0 and 1 — how confident the agent is that something is physically present there. This is not ground truth geometry. It's the textured, probe-patterned map the trained agent actually produces.

### Output

Per-cell semantic labels — shape `(6, 80, 100)`. One channel per class:


| Class | Label          | What it means              |
| ----- | -------------- | -------------------------- |
| 0     | `empty`        | Open floor space           |
| 1     | `wall`         | Room boundary              |
| 2     | `furniture`    | Objects in the room        |
| 3     | `person_torso` | Body                       |
| 4     | `person_head`  | Head                       |
| 5     | `unknown`      | Uncertain / low confidence |


The CNN assigns every cell a class. `unknown` catches areas the map hasn't resolved cleanly yet.

### Architecture

Fully convolutional — no pooling. Input and output stay the same spatial resolution because this is per-cell labeling, not classification.

```
Input (1, 80, 100)
→ Conv2d(1→16,  kernel=3) + BatchNorm + ReLU
→ Conv2d(16→32, kernel=3) + BatchNorm + ReLU
→ Conv2d(32→64, kernel=3) + BatchNorm + ReLU
→ Conv2d(64→32, kernel=3) + BatchNorm + ReLU   # decoder
→ Conv2d(32→16, kernel=3) + BatchNorm + ReLU
→ Conv2d(16→6,  kernel=1)                       # logits
Output (6, 80, 100)
```

Each `Conv2d` slides a 3×3 window across the grid. A `Conv2d(1→16)` reads the occupancy grid and produces 16 filtered versions of it — each filter learns to detect something different: edges, corners, density blobs, gradients. `BatchNorm` normalizes activations after each conv layer so values don't explode or vanish during training. `ReLU` (`max(0, x)`) kills negatives and adds non-linearity.

~47k parameters total. Trains in minutes on CPU.

### Loss

Weighted cross entropy per cell. Most of the grid is empty, so rare classes get upweighted:

```
empty: 0.5 | wall: 2.0 | furniture: 3.0 | person_torso: 8.0 | person_head: 12.0 | unknown: 1.0
```

Person head and torso are the hardest to detect and the most important — they get the highest weight.

### Training Data

Maps must come from the **trained RL agent**, not random chirps. Random probing produces a different map texture than the agent's learned sweep patterns — frequency ranges, directions, and focus areas the CNN would never see at runtime.

Pipeline:

1. Generate random rooms (objects + people placed randomly)
2. Run the trained RL agent for ~50 steps and collect the occupancy map it built
3. Ground truth labels come from known room geometry (`build_label_map`)
4. Train the CNN on agent-built map ↔ label pairs
5. Augment training maps — random flips, rotations, and noise — so the CNN generalizes layout, not memorizes it

### Run

```bash
# train RL first, then CNN
uv run python main.py
uv run python -m src.agent.train_cnn
```

Checkpoints save to `checkpoints/cnn/best_model.pt`.

---

