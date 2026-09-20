from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from core.config import RhythmConfig, get_config

@dataclass(frozen=True)
class OnsetEvent:
    timestamp: float
    strength: float

def _spectral_flux(audio: np.ndarray, sr: int, frame_size: int, hop_size: int) -> tuple[np.ndarray, np.ndarray]:
    n_frames = 1 + (len(audio) - frame_size) // hop_size
    if n_frames <= 1:
        return np.array([]), np.array([])

    window = np.hanning(frame_size)
    prev_mag = None
    flux = np.zeros(n_frames)
    times = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop_size
        frame = audio[start:start + frame_size] * window
        spectrum = np.fft.rfft(frame)
        mag = np.abs(spectrum)
        times[i] = start / sr
        if prev_mag is not None:
            diff = mag - prev_mag
            flux[i] = np.sum(diff[diff > 0])  # half-wave rectified spectral flux
        prev_mag = mag

    return times, flux

def detect_onsets(
    audio: np.ndarray,
    sr: int,
    frame_size: int = 1024,
    hop_size: int = 256,
    config: RhythmConfig | None = None,
) -> list[OnsetEvent]:
    cfg = config or get_config().rhythm
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)

    times, flux = _spectral_flux(audio, sr, frame_size, hop_size)
    if len(flux) < 3:
        return []

    # normalize flux to [0, 1] for a threshold expressed as a fraction
    max_flux = np.max(flux)
    if max_flux <= 0:
        return []
    norm_flux = flux / max_flux

    # adaptive threshold: local moving average + fixed margin, standard
    # approach to avoid over-triggering on a loud passage's sustained energy
    window = 7
    padded = np.pad(norm_flux, (window // 2, window // 2), mode="edge")
    moving_avg = np.array([
        np.mean(padded[i:i + window]) for i in range(len(norm_flux))
    ])
    threshold = moving_avg + cfg.onset_energy_threshold

    onsets: list[OnsetEvent] = []
    last_onset_time = -np.inf
    min_gap_s = cfg.min_inter_onset_ms / 1000.0

    for i in range(1, len(norm_flux) - 1):
        is_local_peak = norm_flux[i] > norm_flux[i - 1] and norm_flux[i] >= norm_flux[i + 1]
        above_threshold = norm_flux[i] > threshold[i]
        if is_local_peak and above_threshold:
            t = times[i]
            if t - last_onset_time >= min_gap_s:
                onsets.append(OnsetEvent(timestamp=float(t), strength=float(norm_flux[i])))
                last_onset_time = t

    return onsets
