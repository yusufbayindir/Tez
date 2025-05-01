import numpy as np
import librosa
from scipy.signal import butter, filtfilt, find_peaks, hilbert
from scipy.ndimage import uniform_filter1d

def load_mono(path: str, target_fs: int = 4000) -> tuple[np.ndarray,int]:
    """Load audio as mono and resample to target_fs."""
    sig, _ = librosa.load(path, sr=target_fs, mono=True)
    return sig, target_fs

def bandpass_filter(sig: np.ndarray, fs: int,
                    lowcut: float = 20.0,
                    highcut: float = 400.0,
                    order: int = 4) -> np.ndarray:
    """Butterworth band-pass filter."""
    nyq = fs / 2
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, sig)

def compute_envelope(sig: np.ndarray, fs: int,
                     smooth_ms: float = 20) -> np.ndarray:
    """Hilbert envelope + moving-average smoothing."""
    env = np.abs(hilbert(sig))
    win = max(1, int(fs * smooth_ms/1000))
    return uniform_filter1d(env, size=win, mode='reflect')

def detect_peaks(env: np.ndarray, fs: int,
                 min_dist_sec: float = 0.2,
                 thresh_std: float = 0.3) -> np.ndarray:
    """
    Find peaks at least min_dist_sec apart,
    height ≥ mean(env) + thresh_std*std(env).
    """
    d = int(min_dist_sec * fs)
    mu, sd = env.mean(), env.std()
    height = mu + thresh_std * sd
    peaks, _ = find_peaks(env, distance=d, height=height)
    return peaks

def assign_s1_s2(peaks: np.ndarray,
                 env: np.ndarray,
                 fs: int,
                 init_min_rr_sec: float = 0.4,
                 init_max_rr_sec: float = 1.2,
                 s1s2_min_sec: float = 0.10,
                 s1s2_max_sec: float = 0.40
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dynamic refractory & adaptive S1–S2 assignment.
    - init_min_rr_sec/init_max_rr_sec: starting S1–S1 bounds
    - s1s2_min_sec/s1s2_max_sec: fixed S1→S2 window
    After each S1–S1, update min_rr/max_rr from last 5 RR’s.
    """
    s1, s2 = [], []
    last_s1_time = None
    # refractory bounds in seconds
    min_rr, max_rr = init_min_rr_sec, init_max_rr_sec
    rr_history = []
    state = 'SEEK_S1'

    for p in peaks:
        t = p / fs
        if state == 'SEEK_S1':
            if last_s1_time is None or (t - last_s1_time) >= min_rr:
                # accept S1
                s1.append(p)
                # update RR stats
                if last_s1_time is not None:
                    rr = t - last_s1_time
                    rr_history.append(rr)
                    recent = rr_history[-5:]
                    avg_rr = sum(recent)/len(recent)
                    min_rr = max(0.3, avg_rr * 0.8)
                    max_rr = avg_rr * 1.2
                last_s1_time = t
                state = 'SEEK_S2'

        else:  # SEEK_S2
            dt = t - last_s1_time
            if dt < s1s2_min_sec:
                continue
            if dt <= s1s2_max_sec:
                s2.append(p)
                state = 'SEEK_S1'
            elif dt > s1s2_max_sec and dt < min_rr:
                # missed S2 but still refractory
                continue
            else:
                # treat as new S1
                s1.append(p)
                rr = dt
                rr_history.append(rr)
                recent = rr_history[-5:]
                avg_rr = sum(recent)/len(recent)
                min_rr = max(0.3, avg_rr * 0.8)
                max_rr = avg_rr * 1.2
                last_s1_time = t
                # remain SEEK_S2 for its S2
    return np.array(s1, dtype=int), np.array(s2, dtype=int)

def detect_s1_s2(path: str) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,int]:
    """
    Full pipeline:
      1) load_mono
      2) bandpass_filter
      3) compute_envelope
      4) detect_peaks
      5) assign_s1_s2 (dynamic)
    Returns: sig, env, s1_indices, s2_indices, fs
    """
    sig, fs = load_mono(path)
    bp       = bandpass_filter(sig, fs)
    env      = compute_envelope(bp, fs)
    peaks    = detect_peaks(env, fs)
    s1, s2   = assign_s1_s2(peaks, env, fs)
    return sig, env, s1, s2, fs
