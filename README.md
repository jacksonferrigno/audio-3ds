# audio-3ds

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

That's the full README core. Want to add an architecture overview diagram or go straight to writing it out as a full file?