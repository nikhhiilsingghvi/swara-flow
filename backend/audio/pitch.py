from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from core.config import AudioConfig, get_config

@dataclass(frozen=True)
class PitchFrame:
    timestamp: float
    frequency_hz: float
    confidence: float
    voiced: bool

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "frequency_hz": self.frequency_hz,
            "confidence": self.confidence,
            "voiced": self.voiced,
        }

class PitchEstimator(Protocol):
    def estimate(self, audio_chunk: np.ndarray, timestamp: float) -> PitchFrame: ...

def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))

def _yin_difference_function(frame: np.ndarray, max_lag: int) -> np.ndarray:
    n = len(frame)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    fft_frame = np.fft.rfft(frame, fft_size)
    acf_full = np.fft.irfft(fft_frame * np.conj(fft_frame), fft_size)[:max_lag]

    energy = np.square(frame)
    cumulative_energy = np.concatenate(([0.0], np.cumsum(energy)))
    total_energy = cumulative_energy[n]

    d = np.empty(max_lag)
    for tau in range(max_lag):
        energy_tau = total_energy - cumulative_energy[tau] if tau < n else 0.0
        e0 = cumulative_energy[n - tau] if tau < n else 0.0
        d[tau] = e0 + energy_tau - 2.0 * acf_full[tau]
    d[0] = 0.0
    return d

def _cumulative_mean_normalized_difference(d: np.ndarray) -> np.ndarray:
    cmnd = np.ones_like(d)
    running_sum = 0.0
    for tau in range(1, len(d)):
        running_sum += d[tau]
        cmnd[tau] = d[tau] * tau / running_sum if running_sum > 0 else 1.0
    return cmnd

def _absolute_threshold(cmnd: np.ndarray, threshold: float, min_lag: int) -> Optional[int]:
    tau = min_lag
    n = len(cmnd)
    while tau < n - 1:
        if cmnd[tau] < threshold:
            while tau + 1 < n and cmnd[tau + 1] < cmnd[tau]:
                tau += 1
            return tau
        tau += 1
    return None

def _parabolic_interpolation(cmnd: np.ndarray, tau: int) -> float:
    if tau <= 0 or tau >= len(cmnd) - 1:
        return float(tau)
    s0, s1, s2 = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
    denom = 2 * s1 - s2 - s0
    if denom == 0:
        return float(tau)
    shift = 0.5 * (s2 - s0) / denom
    return tau + shift

class YinPitchEstimator:
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or get_config().audio

    def estimate(self, audio_chunk: np.ndarray, timestamp: float) -> PitchFrame:
        cfg = self.config
        chunk = np.asarray(audio_chunk, dtype=np.float64)

        if chunk.ndim > 1:
            chunk = chunk.mean(axis=-1)  # downmix to mono if needed

        rms = _rms(chunk)
        if rms < cfg.silence_rms_threshold:
            return PitchFrame(timestamp=timestamp, frequency_hz=0.0, confidence=0.0, voiced=False)

        min_lag = max(1, int(cfg.sample_rate_hz / cfg.fmax_hz))
        max_lag = min(len(chunk) - 1, int(cfg.sample_rate_hz / cfg.fmin_hz))

        if max_lag <= min_lag or len(chunk) < 2 * max_lag:
            return PitchFrame(timestamp=timestamp, frequency_hz=0.0, confidence=0.0, voiced=False)

        d = _yin_difference_function(chunk, max_lag)
        cmnd = _cumulative_mean_normalized_difference(d)
        tau = _absolute_threshold(cmnd, cfg.yin_threshold, min_lag)

        if tau is None:
            tau = int(np.argmin(cmnd[min_lag:max_lag]) + min_lag)
            raw_confidence = max(0.0, 1.0 - float(cmnd[tau]))
            refined_tau = _parabolic_interpolation(cmnd, tau)
            freq = cfg.sample_rate_hz / refined_tau if refined_tau > 0 else 0.0
            confidence = raw_confidence * 0.5 
            voiced = confidence >= cfg.min_voiced_confidence
            return PitchFrame(timestamp=timestamp, frequency_hz=freq if voiced else 0.0,
                               confidence=confidence, voiced=voiced)

        confidence = max(0.0, 1.0 - float(cmnd[tau]))
        refined_tau = _parabolic_interpolation(cmnd, tau)
        freq = cfg.sample_rate_hz / refined_tau if refined_tau > 0 else 0.0
        voiced = confidence >= cfg.min_voiced_confidence and cfg.fmin_hz <= freq <= cfg.fmax_hz

        return PitchFrame(
            timestamp=timestamp,
            frequency_hz=freq if voiced else 0.0,
            confidence=confidence,
            voiced=voiced,
        )

class LibrosaPyinEstimator:
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or get_config().audio
        import librosa
        self._librosa = librosa

    def estimate(self, audio_chunk: np.ndarray, timestamp: float) -> PitchFrame:
        cfg = self.config
        chunk = np.asarray(audio_chunk, dtype=np.float64)
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=-1)

        rms = _rms(chunk)
        if rms < cfg.silence_rms_threshold:
            return PitchFrame(timestamp=timestamp, frequency_hz=0.0, confidence=0.0, voiced=False)

        f0, voiced_flag, voiced_prob = self._librosa.pyin(
            chunk,
            fmin=cfg.fmin_hz,
            fmax=cfg.fmax_hz,
            sr=cfg.sample_rate_hz,
            frame_length=min(cfg.frame_size, len(chunk)),
        )
        valid = ~np.isnan(f0)
        if not np.any(valid):
            return PitchFrame(timestamp=timestamp, frequency_hz=0.0, confidence=0.0, voiced=False)

        idx = np.where(valid)[0][-1]
        freq = float(f0[idx])
        confidence = float(voiced_prob[idx]) if voiced_prob is not None else 0.7
        voiced = bool(voiced_flag[idx]) and confidence >= cfg.min_voiced_confidence
        return PitchFrame(
            timestamp=timestamp,
            frequency_hz=freq if voiced else 0.0,
            confidence=confidence,
            voiced=voiced,
        )

def build_default_pitch_estimator(config: Optional[AudioConfig] = None) -> PitchEstimator:
    try:
        return LibrosaPyinEstimator(config)
    except ImportError:
        return YinPitchEstimator(config)

def estimate_pitch(
    audio_chunk: np.ndarray,
    timestamp: float,
    estimator: Optional[PitchEstimator] = None,
) -> PitchFrame:
    est = estimator or build_default_pitch_estimator()
    return est.estimate(audio_chunk, timestamp)
