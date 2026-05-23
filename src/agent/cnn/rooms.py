import numpy as np

from src.agent.cnn.config import MAX_OBJECTS, MAX_PEOPLE
from src.sim.room import Object, Person, Room


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
