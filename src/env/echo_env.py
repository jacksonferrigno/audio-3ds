
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.sim.room import Room, make_default_room
from src.sim.ray import cast_sweep
from src.sim.acoustic import (
    generate_chirp,
    build_echo_signal,
    extract_features,
    cross_correlate,
    build_occupancy_map,
    EchoObservation,
)

MAP_RESOLUTION = 0.1  # meters per grid cell

class EchoEnv(gym.Env):
    def __init__(
        self,
        room: Room = None,
        max_steps: int = 50,
        n_features: int = 64,
    ):
        super().__init__()
        self.room = room or make_default_room()
        self.max_steps = max_steps
        self.n_features = n_features

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


    def step(self, action: np.ndarray):
        self.current_step += 1

        # scale normalized actions to real values
        f_start = float(np.interp(action[0], [0, 1], [500, 4000]))
        f_end = float(np.interp(action[1], [0, 1], [4000, 16000]))
        direction = float(np.interp(action[2], [0, 1], [0, 2 * np.pi]))
        n_rays = int(np.interp(action[3], [0, 1], [30, 360]))

        # ensure f_end > f_start
        if f_end <= f_start:
            f_end = f_start + 500.0

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
            "n_rays": n_rays,
            "map_coverage": float(self.occupancy_map.mean()),
        }

        return obs.agent_input, reward, terminated, truncated, info


    def _get_observation(self, action: np.ndarray) -> EchoObservation:
        # scale action to real values
        f_start = float(np.interp(action[0], [0, 1], [500, 4000]))
        f_end   = float(np.interp(action[1], [0, 1], [4000, 16000]))
        n_rays  = int(np.interp(action[3],   [0, 1], [30, 360]))

        if f_end <= f_start:
            f_end = f_start + 500.0

        # generate the chirp and cast rays
        emitted_chirp = generate_chirp(f_start, f_end)
        echoes = cast_sweep(
            self.room,
            self.emitter_pos,
            self.emitter_pos,  # mic co-located with emitter for now
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
        # plugs into visualizer.py later
        # will show:
        # - dark room with wall outlines
        # - occupancy map as glowing heatmap
        # - agent emitter position
        # - chirp direction as animated beam
        # - speech waves from person mouth points
        pass