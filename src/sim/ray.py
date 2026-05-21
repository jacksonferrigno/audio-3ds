import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from .room import Room, Wall


SPEED_OF_SOUND = 343.0  # m/s
MIN_ENERGY = 0.01       # stop tracing when energy drops below this


@dataclass
class Hit:
    """Result of a ray intersecting a wall."""
    point: np.ndarray       # where the ray hit
    distance: float         # distance from ray origin to hit
    wall: Wall              # which wall was hit
    time_of_flight: float   # seconds from emission to hit


@dataclass
class EchoEvent:
    """A single echo return registered at the microphone."""
    time_of_flight: float   # total travel time back to mic
    energy: float           # remaining energy at arrival
    total_distance: float   # total path length


@dataclass
class Ray:
    """
    A single ray emitted from a source point at a given angle.
    Bounces around the room collecting echo events.
    """
    origin: np.ndarray
    angle: float            # radians
    energy: float = 1.0
    max_bounces: int = 10

    def __post_init__(self):
        self.origin = np.array(self.origin, dtype=float)
        self.direction = np.array([np.cos(self.angle), np.sin(self.angle)])

    def _intersect_wall(self, origin: np.ndarray, direction: np.ndarray, wall: Wall) -> Optional[float]:
        """
        Ray-segment intersection.
        Returns t (distance along ray) if hit, else None.
        Uses parametric form: origin + t*direction = wall.p1 + u*(wall.p2 - wall.p1)
        """
        d = direction
        v1 = origin - wall.p1
        v2 = wall.p2 - wall.p1
        v3 = np.array([-d[1], d[0]])

        denom = np.dot(v2, v3)
        if abs(denom) < 1e-10:
            return None  # parallel

        t = np.cross(v2, v1) / denom
        u = np.dot(v1, v3) / denom

        if t > 1e-6 and 0.0 <= u <= 1.0:
            return t
        return None

    def _reflect_direction(self, direction: np.ndarray, wall: Wall) -> np.ndarray:
        """Reflect direction vector off a wall using its normal."""
        n = wall.normal
        # ensure normal points against the incoming ray
        if np.dot(n, direction) > 0:
            n = -n
        return direction - 2 * np.dot(direction, n) * n

    def _find_closest_hit(self, origin: np.ndarray, direction: np.ndarray, walls: list[Wall]) -> Optional[Hit]:
        """Find the closest wall hit from current origin."""
        closest_t = np.inf
        closest_wall = None

        for wall in walls:
            t = self._intersect_wall(origin, direction, wall)
            if t is not None and t < closest_t:
                closest_t = t
                closest_wall = wall

        if closest_wall is None:
            return None

        hit_point = origin + closest_t * direction
        tof = closest_t / SPEED_OF_SOUND

        return Hit(
            point=hit_point,
            distance=closest_t,
            wall=closest_wall,
            time_of_flight=tof,
        )

    def cast(self, room: Room, mic_position: np.ndarray) -> list[EchoEvent]:
        """
        Cast the ray through the room, bouncing off walls.
        At each bounce, check if the reflection could reach the mic.
        Returns a list of EchoEvents representing echoes heard at the mic.
        """
        echoes = []
        walls = room.all_walls

        current_origin = self.origin.copy()
        current_direction = self.direction.copy()
        current_energy = self.energy
        total_distance = 0.0

        for _ in range(self.max_bounces):
            if current_energy < MIN_ENERGY:
                break

            hit = self._find_closest_hit(current_origin, current_direction, walls)
            if hit is None:
                break

            total_distance += hit.distance
            current_energy *= (1.0 - hit.wall.absorption)

            # check if this bounce point has line-of-sight to the mic
            echo = self._check_mic_path(
                hit.point,
                mic_position,
                walls,
                current_energy,
                total_distance,
            )
            if echo is not None:
                echoes.append(echo)

            # reflect and continue
            current_direction = self._reflect_direction(current_direction, hit.wall)
            current_origin = hit.point + current_direction * 1e-6  # nudge off wall

        return echoes

    def _check_mic_path(
        self,
        bounce_point: np.ndarray,
        mic_position: np.ndarray,
        walls: list[Wall],
        energy: float,
        distance_so_far: float,
    ) -> Optional[EchoEvent]:
        """
        Check if there's a clear path from bounce_point to the mic.
        If so, return an EchoEvent.
        """
        to_mic = mic_position - bounce_point
        dist_to_mic = np.linalg.norm(to_mic)
        if dist_to_mic < 1e-6:
            return None

        direction = to_mic / dist_to_mic

        for wall in walls:
            t = self._intersect_wall(bounce_point + direction * 1e-6, direction, wall)
            if t is not None and t < dist_to_mic - 1e-4:
                return None  # wall is blocking the path to mic

        total_dist = distance_so_far + dist_to_mic
        tof = total_dist / SPEED_OF_SOUND

        # energy decays with distance (inverse square)
        arrival_energy = energy / (1.0 + dist_to_mic ** 2)

        return EchoEvent(
            time_of_flight=tof,
            energy=arrival_energy,
            total_distance=total_dist,
        )


def cast_sweep(
    room: Room,
    emitter_position: np.ndarray,
    mic_position: np.ndarray,
    n_rays: int = 360,
    max_bounces: int = 5,
) -> list[EchoEvent]:
    """
    Emit a full sweep of rays in all directions and collect all echoes.
    This is the full chirp emission from one timestep.
    """
    all_echoes = []
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)

    for angle in angles:
        ray = Ray(
            origin=emitter_position,
            angle=angle,
            energy=1.0,
            max_bounces=max_bounces,
        )
        echoes = ray.cast(room, mic_position)
        all_echoes.extend(echoes)

    # sort by time of flight
    all_echoes.sort(key=lambda e: e.time_of_flight)
    return all_echoes