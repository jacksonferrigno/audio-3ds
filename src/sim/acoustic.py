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



