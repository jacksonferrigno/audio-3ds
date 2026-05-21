import numpy as np
from scipy.signal import chirp, correlate
from dataclasses import dataclass


SPEED_OF_SOUND = 343.0  # m/s
SAMPLE_RATE = 44100     # Hz, standard audio sample rate


def generate_chirp(
    f_start: float, 
    f_end: float,
    duration: float = 0.01,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    generate a linear frequency sweep (chirp) signal.
    f_start and f end are frequencies in Hz
    duration: how long a chirp is in seconds 
    """
    t = np.linspace(0,duration, int(sample_rate *duration), endpoint=False)
    signal = chirp(t, f0=f_start, f1=f_end, t1=duration, method='linear')

    window = np.hanning(len(signal))

    return signal * window

def build_echo_signal(
    echo_events: list,
    chirp: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    duration: float = 0.1,
) -> np.ndarray:
    """
    Place each echo event onto a time axis as an attenuated copy of the chirp.
    Returns a 1D array representing what the microphone hears.
    duration: total recording window in seconds (default 100ms)
    rl agent will read this like a bat 🦇
    """ 
    n_samples = int(sample_rate * duration)
    signal = np.zeros(n_samples)

    for echo in echo_events:
        # convert time of flight to sample index
        sample_idx = int(echo.time_of_flight * sample_rate)
        end_idx = sample_idx + len(chirp)

        if end_idx > n_samples:
            continue  # echo arrives outside our recording window

        # place attenuated chirp at arrival time
        signal[sample_idx:end_idx] += chirp * echo.energy

    # add a small amount of gaussian noise (real mics aren't perfect)
    noise = np.random.normal(0, 0.005, n_samples)
    return signal + noise

def extract_features(
    signal: np.ndarray,
    n_bins: int = 64,
) -> np.ndarray:
    """
    Compress a raw echo waveform into a compact feature vector.
    Uses the power spectrum — how much energy at each frequency.
    n_bins: how many frequency buckets to return (default 64)
    """
    # fast fourier transform — converts time domain to frequency domain
    fft = np.fft.rfft(signal)
    power = np.abs(fft) ** 2

    # compress down to n_bins by averaging neighboring buckets
    bin_size = len(power) // n_bins
    features = np.array([
        power[i * bin_size:(i + 1) * bin_size].mean()
        for i in range(n_bins)
    ])

    # normalize to 0-1 range
    max_val = features.max()
    if max_val > 0:
        features /= max_val

    return features.astype(np.float32)

def cross_correlate(
    emitted: np.ndarray,
    received: np.ndarray,
) -> np.ndarray:
    """
    Slide the emitted chirp across the received signal to find
    precise echo arrival times.
    Peaks in the output correspond to wall/object reflections.
    Returns a normalized correlation array the same length as received.
    rl agent needs this to seperate blurred things that are close together 
    """
    correlation = correlate(received, emitted, mode='full')

    # take only the causal half - echoes arrive after emission 
    causal = correlation[len(emitted)-1:]

    #normalize 
    max_val = np.abs(causal).max()

    if max_val>0: # no /0 errors
        causal /= max_val

    return causal.astype(np.float32)

def build_occupancy_map(
    echo_history: list,
    emitter_positions: list,
    room_width: float,
    room_height: float,
    resolution: float = 0.1,
) -> np.ndarray:
    """
    Build a 2D occupancy grid from accumulated echo history.
    echo_history: list of (features, tof) pairs from previous steps
    emitter_positions: where the agent was when each echo was recorded
    resolution: grid cell size in meters (default 10cm)
    returns a 2D array of shape (cols, rows) with values 0-1
    """
    cols = int(room_width / resolution)
    rows = int(room_height / resolution)
    grid = np.zeros((rows, cols), dtype=np.float32)

    for (features, tof_list), emitter_pos in zip(echo_history, emitter_positions):
        for tof in tof_list:
            # convert time of flight to distance
            distance = tof * SPEED_OF_SOUND

            # draw a circle of probability at that distance from emitter
            # something was at this distance — we just don't know what angle
            ex, ey = emitter_pos
            for angle in np.linspace(0, 2 * np.pi, 360):
                x = ex + distance * np.cos(angle)
                y = ey + distance * np.sin(angle)

                col = int(x / resolution)
                row = int(y / resolution)

                if 0 <= row < rows and 0 <= col < cols:
                    grid[row, col] += 0.1

    # normalize
    max_val = grid.max()
    if max_val > 0:
        grid /= max_val

    return grid

@dataclass
class EchoObservation:
    """
    Everything the RL agent receives after emitting one chirp.
    This is the agent's full view of the world at one timestep.
    """
    # raw waveform from build_echo_signal
    raw_signal: np.ndarray

    # sharpened via cross_correlate
    correlated: np.ndarray

    # compressed 64-dim feature vector from extract_features
    features: np.ndarray

    # current occupancy map — the agent's running belief about the room
    occupancy_map: np.ndarray

    # time of flight values from this step's echo events
    tof_list: list[float]

    # where the emitter was when this chirp was fired
    emitter_position: np.ndarray

    # step number — how many chirps have been fired so far
    step: int = 0

    @property
    def agent_input(self) -> np.ndarray:
        """
        What actually gets fed into the RL policy network.
        Concatenates features + flattened occupancy map into one vector.
        """
        return np.concatenate([
            self.features,
            self.occupancy_map.flatten(),
        ]).astype(np.float32)