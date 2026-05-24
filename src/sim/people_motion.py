import math

import numpy as np

from src.sim.room import Room


def tick_people(
    room: Room,
    rng: np.random.Generator,
    speed: float = 0.06,
    margin: float = 0.6,
) -> None:
    """Small random drift per step — same person body, slightly new position."""
    for person in room.people:
        dx = float(rng.uniform(-1.0, 1.0) * speed)
        dy = float(rng.uniform(-1.0, 1.0) * speed)

        x = float(person.position[0] + dx)
        y = float(person.position[1] + dy)
        hw = person.body_width / 2 + 0.05
        hh = person.body_height / 2 + 0.05

        x = float(np.clip(x, margin + hw, room.width - margin - hw))
        y = float(np.clip(y, margin + hh, room.height - margin - hh))
        person.position = np.array([x, y], dtype=float)

        if abs(dx) + abs(dy) > 1e-6:
            person.facing = math.atan2(dy, dx)
