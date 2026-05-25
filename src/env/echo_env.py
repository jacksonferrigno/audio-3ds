
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.sim.room import Room, make_default_room
from src.sim.ray import cast_sweep, cast_sweep_first_hits
from src.sim.people_motion import tick_people
from src.sim.acoustic import (
    generate_chirp,
    build_echo_signal,
    extract_features,
    cross_correlate,
    build_occupancy_map,
    EchoObservation,
)

MAP_RESOLUTION = 0.1  # meters per grid cell
DEFAULT_MAX_STEPS = 125

def _action_params(action: np.ndarray) -> tuple[float, float, float, float, int]:
    f_start = float(np.interp(action[0], [0, 1], [500, 4000]))
    f_end = float(np.interp(action[1], [0, 1], [4000, 16000]))
    direction = float(np.interp(action[2], [0, 1], [0, 2 * np.pi]))
    sweep_width = float(np.interp(action[3], [0, 1], [np.pi / 12, np.pi]))
    n_rays = max(30, int(30 + (sweep_width / np.pi) * 150))

    if f_end <= f_start:
        f_end = f_start + 500.0

    return f_start, f_end, direction, sweep_width, n_rays


class EchoEnv(gym.Env):
    def __init__(
        self,
        room: Room = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        n_features: int = 64,
    ):
        super().__init__()
        self.room = room or make_default_room()
        self.max_steps = max_steps
        self.n_features = n_features
        self.moving_people = False

        # map dimensions
        self.map_cols = int(self.room.width / MAP_RESOLUTION)
        self.map_rows = int(self.room.height / MAP_RESOLUTION)
        self.map_size = self.map_cols * self.map_rows

        # action space — 4 continuous values all normalized 0-1
        # [f_start, f_end, direction, sweep_width]
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # observation space — features + flattened occupancy map
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_features + self.map_size,),
            dtype=np.float32,
        )

        # internal state
        self.occupancy_map = np.zeros((self.map_rows, self.map_cols), dtype=np.float32)
        self._prev_occupancy_map = self.occupancy_map.copy()
        self.current_step = 0
        self.emitter_pos = self.room.random_position()
        self.last_hit_points: list = []

    def set_moving_people(self, enabled: bool) -> None:
        self.moving_people = enabled

    def reset(
        self,
        seed: int =None,
        options: dict =None,
    ):
        super().reset(seed=seed)
        self.occupancy_map = np.zeros(
            (self.map_rows, self.map_cols), dtype=np.float32
        )
        self._prev_occupancy_map = self.occupancy_map.copy()
        self.current_step = 0

        #place the emitter at a random postion in the room 
        self.emitter_pos= self.room.random_position()

        #fire intial chirp so the agent has something to observe 
        obs = self._get_observation(action=np.array([0.2, 0.8, 0.0, 1.0]))

        return obs.agent_input, {}

    def reset_for_cache(self, seed: int | None = None) -> np.ndarray:
        super().reset(seed=seed)
        self.occupancy_map.fill(0.0)
        self._prev_occupancy_map = self.occupancy_map.copy()
        self.current_step = 0
        self.emitter_pos = self.room.random_position()
        initial = np.array([0.2, 0.8, 0.0, 1.0], dtype=np.float32)
        return self.cache_step(initial)

    def step(self, action: np.ndarray):
        self.current_step += 1

        f_start, f_end, direction, sweep_width, n_rays = _action_params(action)

        if self.moving_people and self.room.people:
            tick_people(self.room, self.np_random)

        # get observation from this action
        obs = self._get_observation(action)

        # calculate reward
        reward = self._calculate_reward(obs)

        # episode ends when max steps reached
        terminated = self.current_step >= self.max_steps
        truncated = False

        info = {
            "step": self.current_step,
            "f_start": f_start,
            "f_end": f_end,
            "direction": direction,
            "sweep_width": sweep_width,
            "n_rays": n_rays,
            "map_coverage": float(self.occupancy_map.mean()),
            "moving_people": self.moving_people,
        }

        return obs.agent_input, reward, terminated, truncated, info

    def cache_step(self, action: np.ndarray) -> np.ndarray:
        """CNN cache generation — map update + RL obs, skip reward/correlation extras."""
        self.current_step += 1
        f_start, f_end, direction, sweep_width, n_rays = _action_params(action)

        emitted_chirp = generate_chirp(f_start, f_end)
        echoes = cast_sweep(
            self.room,
            self.emitter_pos,
            self.emitter_pos,
            direction=direction,
            sweep_width=sweep_width,
            n_rays=n_rays,
        )
        tof_list = [e.time_of_flight for e in echoes]
        self._prev_occupancy_map = self.occupancy_map.copy()
        new_map = build_occupancy_map(
            [(None, tof_list)],
            [self.emitter_pos.copy()],
            self.room.width,
            self.room.height,
        )
        self.occupancy_map = np.clip(self.occupancy_map + new_map * 0.3, 0.0, 1.0)

        received = build_echo_signal(echoes, emitted_chirp)
        features = extract_features(received)
        return np.concatenate(
            [features, self.occupancy_map.flatten()],
        ).astype(np.float32)

    def _get_observation(self, action: np.ndarray) -> EchoObservation:
        f_start, f_end, direction, sweep_width, n_rays = _action_params(action)

        # generate the chirp and cast rays
        emitted_chirp = generate_chirp(f_start, f_end)
        self.last_hit_points = cast_sweep_first_hits(
            self.room,
            self.emitter_pos,
            direction=direction,
            sweep_width=sweep_width,
            n_rays=n_rays,
        )
        echoes = cast_sweep(
            self.room,
            self.emitter_pos,
            self.emitter_pos,  # mic co-located with emitter for now
            direction=direction,
            sweep_width=sweep_width,
            n_rays=n_rays,
        )

        # build signal and extract features
        received = build_echo_signal(echoes, emitted_chirp)
        features = extract_features(received)
        correlated = cross_correlate(emitted_chirp, received)

        # update occupancy map with new echoes
        tof_list = [e.time_of_flight for e in echoes]
        self._prev_occupancy_map = self.occupancy_map.copy()
        new_map = build_occupancy_map(
            [(features, tof_list)],
            [self.emitter_pos.copy()],
            self.room.width,
            self.room.height,
        )
        self.occupancy_map = np.clip(self.occupancy_map + new_map * 0.3, 0.0, 1.0)

        return EchoObservation(
            raw_signal=received,
            correlated=correlated,
            features=features,
            occupancy_map=self.occupancy_map,
            tof_list=tof_list,
            emitter_position=self.emitter_pos.copy(),
            step=self.current_step,
        )
    def _calculate_reward(self, obs: EchoObservation) -> float:
        reward = 0.0

        # 1. information gain — how much did the map change this step
        delta = np.abs(obs.occupancy_map - self._prev_occupancy_map)
        information_gain = float((delta > 0.05).sum()) / self.map_size
        reward += information_gain * 2.0

        # 2. penalize redundancy — firing into already certain areas
        # certain cells are those already close to 0 or 1
        certainty = np.abs(obs.occupancy_map - 0.5) * 2  # 0=uncertain, 1=certain
        avg_certainty = float(certainty.mean())
        reward -= avg_certainty * 0.3

        # 3. coverage bonus — reward for mapping more of the room
        coverage = float((obs.occupancy_map > 0.1).sum()) / self.map_size
        reward += coverage * 0.5

        # 4. small step penalty — encourages efficiency
        # agent should map the room fast not waste steps
        reward -= 0.01

        return float(reward)

    def render(self):
        pass
