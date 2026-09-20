from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val is not None else default


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val is not None else default


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class AudioConfig:
    sample_rate_hz: int = field(default_factory=lambda: _env_int("SF_SAMPLE_RATE_HZ", 44100))
    frame_size: int = field(default_factory=lambda: _env_int("SF_FRAME_SIZE", 2048))
    hop_size: int = field(default_factory=lambda: _env_int("SF_HOP_SIZE", 512))
    fmin_hz: float = field(default_factory=lambda: _env_float("SF_FMIN_HZ", 70.0))
    fmax_hz: float = field(default_factory=lambda: _env_float("SF_FMAX_HZ", 1000.0))
    yin_threshold: float = field(default_factory=lambda: _env_float("SF_YIN_THRESHOLD", 0.12))
    silence_rms_threshold: float = field(default_factory=lambda: _env_float("SF_SILENCE_RMS", 0.01))
    min_voiced_confidence: float = field(default_factory=lambda: _env_float("SF_MIN_VOICED_CONF", 0.5))

@dataclass(frozen=True)
class SwaraConfig:
    max_snap_distance_cents: float = field(
        default_factory=lambda: _env_float("SF_MAX_SNAP_CENTS", 60.0)
    )
    stability_window_cents: float = field(
        default_factory=lambda: _env_float("SF_STABILITY_WINDOW_CENTS", 35.0)
    )
    stability_min_ms: float = field(
        default_factory=lambda: _env_float("SF_STABILITY_MIN_MS", 150.0)
    )

    octave_cents: float = 1200.0  

@dataclass(frozen=True)
class RhythmConfig:
    onset_energy_threshold: float = field(
        default_factory=lambda: _env_float("SF_ONSET_ENERGY_THRESHOLD", 0.15)
    )
    min_inter_onset_ms: float = field(
        default_factory=lambda: _env_float("SF_MIN_INTER_ONSET_MS", 120.0)
    )
    tempo_estimation_window_s: float = field(
        default_factory=lambda: _env_float("SF_TEMPO_WINDOW_S", 8.0)
    )

@dataclass(frozen=True)
class VisionConfig:
    target_fps: int = field(default_factory=lambda: _env_int("SF_TARGET_FPS", 24))
    process_every_nth_frame: int = field(
        default_factory=lambda: _env_int("SF_PROCESS_EVERY_NTH_FRAME", 2)
    )
    max_frame_width: int = field(default_factory=lambda: _env_int("SF_MAX_FRAME_WIDTH", 640))
    face_detector_confidence: float = field(
        default_factory=lambda: _env_float("SF_FACE_DETECTOR_CONF", 0.5)
    )
    motion_history_frames: int = field(
        default_factory=lambda: _env_int("SF_MOTION_HISTORY_FRAMES", 15)
    )

@dataclass(frozen=True)
class SyncConfig:
    tolerance_ms: float = field(default_factory=lambda: _env_float("SF_SYNC_TOLERANCE_MS", 40.0))

@dataclass(frozen=True)
class PathsConfig:
    data_dir: Path = field(default_factory=lambda: Path(_env_str("SF_DATA_DIR", "./data")).resolve())
    configs_dir: Path = field(default_factory=lambda: Path(_env_str("SF_CONFIGS_DIR", "./configs")).resolve())

    @property
    def raga_dir(self) -> Path:
        return self.configs_dir / "raga"

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

@dataclass(frozen=True)
class LLMConfig:
    provider: str = field(default_factory=lambda: _env_str("SF_LLM_PROVIDER", "none"))
    lmstudio_base_url: str = field(
        default_factory=lambda: _env_str("SF_LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    )
    ollama_base_url: str = field(
        default_factory=lambda: _env_str("SF_OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_model: str = field(default_factory=lambda: _env_str("SF_OLLAMA_MODEL", "llama3.1"))
    openai_compat_base_url: str = field(
        default_factory=lambda: _env_str("SF_OPENAI_COMPAT_BASE_URL", "")
    )
    openai_compat_api_key: str = field(
        default_factory=lambda: _env_str("SF_OPENAI_COMPAT_API_KEY", "")
    )
    openai_compat_model: str = field(default_factory=lambda: _env_str("SF_OPENAI_COMPAT_MODEL", ""))
    request_timeout_s: float = field(default_factory=lambda: _env_float("SF_LLM_TIMEOUT_S", 15.0))

@dataclass(frozen=True)
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    swara: SwaraConfig = field(default_factory=SwaraConfig)
    rhythm: RhythmConfig = field(default_factory=RhythmConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    telemetry_enabled: bool = field(default_factory=lambda: _env_bool("SF_TELEMETRY_ENABLED", False))

_config: Config | None = None

def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config

def reset_config_for_tests() -> None:
    global _config
    _config = None
