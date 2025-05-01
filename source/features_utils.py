# features_utils.py

import numpy as np
import librosa

def compute_mel_spectrogram(
    path: str,
    sr: int = 4000,
    n_fft: int = 512,
    hop_length: int = 128,
    n_mels: int = 64
) -> np.ndarray:
    sig, _ = librosa.load(path, sr=sr, mono=True)
    S = librosa.feature.melspectrogram(
        y=sig,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )
    return librosa.power_to_db(S, ref=np.max)

def extract_cycle_spectrograms(
    path: str,
    s1_idx: np.ndarray,
    s2_idx: np.ndarray,
    sr: int = 4000,
    n_fft: int = 512,
    hop_length: int = 128,
    n_mels: int = 64
) -> list[np.ndarray]:
    sig, _ = librosa.load(path, sr=sr, mono=True)
    specs = []
    for start, end in zip(s1_idx, s2_idx):
        # skip invalid intervals
        if end <= start:
            continue
        # make sure no two S1 or two S2 in a row for this pair
        # (zip already pairs sequentially, so just check ordering)
        pad = int(0.01 * sr)
        a = max(0, start - pad)
        b = min(len(sig), end + pad)
        seg = sig[a:b]
        S = librosa.feature.melspectrogram(
            y=seg,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        )
        specs.append(librosa.power_to_db(S, ref=np.max))
    return specs
