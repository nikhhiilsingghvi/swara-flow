from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .onset import OnsetEvent

@dataclass(frozen=True)
class TempoEstimate:
    bpm: Optional[float]
    confidence: float
    tempo_drift: float

def estimate_tempo(onsets: list[OnsetEvent]) -> TempoEstimate:
    if len(onsets) < 3:
        return TempoEstimate(bpm=None, confidence=0.0, tempo_drift=0.0)

    times = np.array([o.timestamp for o in onsets])
    iois = np.diff(times)
    iois = iois[iois > 0]
    if len(iois) < 2:
        return TempoEstimate(bpm=None, confidence=0.0, tempo_drift=0.0)

    median_ioi = float(np.median(iois))
    bpm = 60.0 / median_ioi if median_ioi > 0 else None

    std_ioi = float(np.std(iois))
    cv = std_ioi / median_ioi if median_ioi > 0 else 1.0
    confidence = max(0.0, min(1.0, 1.0 - cv))

    half = len(iois) // 2
    if half >= 1:
        first_half_median = float(np.median(iois[:half]))
        second_half_median = float(np.median(iois[half:]))
        if first_half_median > 0 and second_half_median > 0:
            bpm_first = 60.0 / first_half_median
            bpm_second = 60.0 / second_half_median
            drift = (bpm_second - bpm_first) / bpm_first
        else:
            drift = 0.0
    else:
        drift = 0.0

    return TempoEstimate(bpm=bpm, confidence=confidence, tempo_drift=drift)

@dataclass(frozen=True)
class Vibhag:
    index: int
    matras: list[int]
    is_khali: bool
    is_sam: bool

@dataclass(frozen=True)
class TaalDefinition:
    name: str
    total_matras: int
    vibhag_pattern: list[int]
    khali_vibhag_indices: list[int]
    sam_matra: int = 1

    def vibhags(self) -> list[Vibhag]:
        result = []
        matra_cursor = 1
        for idx, size in enumerate(self.vibhag_pattern):
            matras = list(range(matra_cursor, matra_cursor + size))
            result.append(
                Vibhag(
                    index=idx,
                    matras=matras,
                    is_khali=idx in self.khali_vibhag_indices,
                    is_sam=(self.sam_matra in matras),
                )
            )
            matra_cursor += size
        return result

    def matra_of_beat_index(self, beat_index: int) -> int:
        return (beat_index % self.total_matras) + 1

TEENTAAL = TaalDefinition(
    name="Teentaal",
    total_matras=16,
    vibhag_pattern=[4, 4, 4, 4],
    khali_vibhag_indices=[2],
    sam_matra=1,
)

TAAL_LIBRARY: dict[str, TaalDefinition] = {
    "teentaal": TEENTAAL,
}

def get_taal(name: str) -> TaalDefinition:
    key = name.strip().lower()
    if key not in TAAL_LIBRARY:
        raise KeyError(f"Unknown taal '{name}'. Available: {sorted(TAAL_LIBRARY)}")
    return TAAL_LIBRARY[key]

@dataclass(frozen=True)
class RhythmicAlignment:
    matched_matras: list[int]
    mean_alignment_error_ms: float
    is_reliable: bool

def align_onsets_to_taal(
    onsets: list[OnsetEvent],
    taal: TaalDefinition,
    bpm: float,
    cycle_start_time: float,
) -> RhythmicAlignment:
    if bpm is None or bpm <= 0 or len(onsets) == 0:
        return RhythmicAlignment(matched_matras=[], mean_alignment_error_ms=0.0, is_reliable=False)

    matra_duration_s = 60.0 / bpm
    matched = []
    errors = []
    for onset in onsets:
        elapsed = onset.timestamp - cycle_start_time
        beat_index = round(elapsed / matra_duration_s)
        expected_time = cycle_start_time + beat_index * matra_duration_s
        error_ms = abs(onset.timestamp - expected_time) * 1000.0
        matra = taal.matra_of_beat_index(beat_index)
        matched.append(matra)
        errors.append(error_ms)

    return RhythmicAlignment(
        matched_matras=matched,
        mean_alignment_error_ms=float(np.mean(errors)) if errors else 0.0,
        is_reliable=len(onsets) >= 4,
    )
