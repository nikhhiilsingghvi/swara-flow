from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PitchPoint:
    timestamp: float
    cents_from_sa: float
    voiced: bool

@dataclass(frozen=True)
class StabilityReport:
    mean_absolute_jitter_cents: float   
    stddev_cents: float                 
    percent_voiced: float               
    is_stable: bool                     

def compute_stability(points: list[PitchPoint], jitter_threshold_cents: float = 25.0) -> StabilityReport:
    if not points:
        return StabilityReport(0.0, 0.0, 0.0, False)

    voiced_points = [p for p in points if p.voiced]
    percent_voiced = 100.0 * len(voiced_points) / len(points)

    if len(voiced_points) < 2:
        return StabilityReport(0.0, 0.0, percent_voiced, False)

    cents = np.array([p.cents_from_sa for p in voiced_points])
    diffs = np.abs(np.diff(cents))
    mean_jitter = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
    stddev = float(np.std(cents))

    is_stable = mean_jitter <= jitter_threshold_cents

    return StabilityReport(
        mean_absolute_jitter_cents=mean_jitter,
        stddev_cents=stddev,
        percent_voiced=percent_voiced,
        is_stable=is_stable,
    )

@dataclass(frozen=True)
class IntonationAccuracy:
    mean_absolute_deviation_cents: float
    percent_within_tolerance: float   
    worst_deviation_cents: float
    n_samples: int

def compute_intonation_accuracy(
    deviations_cents: list[float],
    tolerance_cents: float = 20.0,
) -> IntonationAccuracy:
    if not deviations_cents:
        return IntonationAccuracy(0.0, 0.0, 0.0, 0)

    arr = np.abs(np.array(deviations_cents))
    mean_abs = float(np.mean(arr))
    within = float(np.mean(arr <= tolerance_cents)) * 100.0
    worst = float(np.max(arr))

    return IntonationAccuracy(
        mean_absolute_deviation_cents=mean_abs,
        percent_within_tolerance=within,
        worst_deviation_cents=worst,
        n_samples=len(deviations_cents),
    )
