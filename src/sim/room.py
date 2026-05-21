import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Wall:
    """A line segment representing a wall in the room."""
    p1: np.ndarray  # start point [x, y]
    p2: np.ndarray  # end point [x, y]
    absorption: float = 0.1  # energy absorbed on each bounce (0 = perfect reflection, 1 = full absorption)

    def __post_init__(self):
        self.p1 = np.array(self.p1, dtype=float)
        self.p2 = np.array(self.p2, dtype=float)

    @property
    def normal(self) -> np.ndarray:
        """Outward-facing normal vector of the wall."""
        d = self.p2 - self.p1
        n = np.array([-d[1], d[0]])
        return n / np.linalg.norm(n)

    @property
    def length(self) -> float:
        return np.linalg.norm(self.p2 - self.p1)


@dataclass
class Object:
    """A rectangular object inside the room (furniture, person, etc)."""
    position: np.ndarray        # center [x, y]
    width: float
    height: float
    absorption: float = 0.3
    label: Optional[str] = None

    def __post_init__(self):
        self.position = np.array(self.position, dtype=float)

    @property
    def walls(self) -> list[Wall]:
        """Returns the 4 walls of the object as line segments."""
        x, y = self.position
        hw, hh = self.width / 2, self.height / 2
        corners = [
            [x - hw, y - hh],
            [x + hw, y - hh],
            [x + hw, y + hh],
            [x - hw, y + hh],
        ]
        return [
            Wall(corners[i], corners[(i + 1) % 4], absorption=self.absorption)
            for i in range(4)
        ]


@dataclass
class Person:
    """
    A person in the room. Has a position, a facing direction, and a mouth point.
    Treated as a capsule shape for ray bouncing (head + body as two overlapping boxes).
    The mouth point is where speech waves originate.
    """
    position: np.ndarray        # center of body [x, y]
    facing: float = 0.0         # angle in radians, 0 = facing right
    absorption: float = 0.6     # bodies absorb more sound than walls
    label: Optional[str] = None

    # body dimensions (meters)
    body_width: float = 0.45
    body_height: float = 0.3
    head_radius: float = 0.12

    def __post_init__(self):
        self.position = np.array(self.position, dtype=float)

        # speech wave state
        self.is_speaking: bool = False
        self.speech_amplitude: float = 0.0  # 0.0 - 1.0, driven by mic input

    @property
    def mouth_point(self) -> np.ndarray:
        """Point slightly in front of the face — where speech waves emit from."""
        facing_dir = np.array([np.cos(self.facing), np.sin(self.facing)])
        # mouth is at head position + a little forward
        head_offset = facing_dir * (self.body_height / 2 + self.head_radius)
        mouth_offset = facing_dir * 0.05  # a bit in front of the head center
        return self.position + head_offset + mouth_offset

    @property
    def head_position(self) -> np.ndarray:
        """Center of the head."""
        facing_dir = np.array([np.cos(self.facing), np.sin(self.facing)])
        return self.position + facing_dir * (self.body_height / 2 + self.head_radius)

    @property
    def walls(self) -> list[Wall]:
        """
        Approximate the person as two rectangles: torso and head box.
        Good enough for ray bouncing.
        """
        # torso
        x, y = self.position
        hw, hh = self.body_width / 2, self.body_height / 2
        torso_corners = [
            [x - hw, y - hh],
            [x + hw, y - hh],
            [x + hw, y + hh],
            [x - hw, y + hh],
        ]
        torso_walls = [
            Wall(torso_corners[i], torso_corners[(i + 1) % 4], absorption=self.absorption)
            for i in range(4)
        ]

        # head (small square approximation)
        hx, hy = self.head_position
        hr = self.head_radius
        head_corners = [
            [hx - hr, hy - hr],
            [hx + hr, hy - hr],
            [hx + hr, hy + hr],
            [hx - hr, hy + hr],
        ]
        head_walls = [
            Wall(head_corners[i], head_corners[(i + 1) % 4], absorption=self.absorption)
            for i in range(4)
        ]

        return torso_walls + head_walls

    def set_speaking(self, amplitude: float):
        """Called by mic_input to update speech state."""
        self.speech_amplitude = float(np.clip(amplitude, 0.0, 1.0))
        self.is_speaking = self.speech_amplitude > 0.05


@dataclass
class Room:
    """
    A 2D rectangular room with walls, objects, and people inside.
    Origin is bottom-left corner.
    """
    width: float                        # meters
    height: float                       # meters
    wall_absorption: float = 0.05
    objects: list[Object] = field(default_factory=list)
    people: list[Person] = field(default_factory=list)

    def __post_init__(self):
        self._build_walls()

    def _build_walls(self):
        """Build the 4 boundary walls of the room."""
        w, h = self.width, self.height
        self.boundary_walls = [
            Wall([0, 0], [w, 0], absorption=self.wall_absorption),   # bottom
            Wall([w, 0], [w, h], absorption=self.wall_absorption),   # right
            Wall([w, h], [0, h], absorption=self.wall_absorption),   # top
            Wall([0, h], [0, 0], absorption=self.wall_absorption),   # left
        ]

    @property
    def all_walls(self) -> list[Wall]:
        """All walls: room boundary + object walls + people."""
        walls = list(self.boundary_walls)
        for obj in self.objects:
            walls.extend(obj.walls)
        for person in self.people:
            walls.extend(person.walls)
        return walls

    def add_object(self, obj: Object):
        self.objects.append(obj)

    def add_person(self, person: Person):
        self.people.append(person)

    @property
    def speaking_people(self) -> list[Person]:
        return [p for p in self.people if p.is_speaking]

    def is_inside(self, point: np.ndarray) -> bool:
        """Check if a point is inside the room boundary."""
        x, y = point
        return 0 <= x <= self.width and 0 <= y <= self.height

    def random_position(self, margin: float = 0.5) -> np.ndarray:
        """Return a random position inside the room with a margin from walls."""
        x = np.random.uniform(margin, self.width - margin)
        y = np.random.uniform(margin, self.height - margin)
        return np.array([x, y])


def make_default_room() -> Room:
    room = Room(width=10.0, height=8.0)
    room.add_object(Object(position=[3.0, 3.0], width=1.0, height=2.0, label="couch"))
    room.add_person(Person(position=[7.0, 5.0], facing=np.pi, label="person_1"))
    return room