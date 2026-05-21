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
class Room:
    """
    A 2D rectangular room with walls and optional objects inside.
    Origin is bottom-left corner.
    """
    width: float                        # meters
    height: float                       # meters
    wall_absorption: float = 0.05
    objects: list[Object] = field(default_factory=list)

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
        """All walls: room boundary + object walls."""
        walls = list(self.boundary_walls)
        for obj in self.objects:
            walls.extend(obj.walls)
        return walls

    def add_object(self, obj: Object):
        self.objects.append(obj)

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
    """A simple 10x8m room with a couple of objects for testing."""
    room = Room(width=10.0, height=8.0)
    room.add_object(Object(position=[3.0, 3.0], width=1.0, height=2.0, label="couch"))
    room.add_object(Object(position=[7.0, 5.0], width=0.5, height=0.5, label="person"))
    return room