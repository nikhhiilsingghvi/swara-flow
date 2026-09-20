from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np

from .intonation import PitchPoint

SegmentKind = Literal["stable", "glide", "jump", "unstable", "silence"]


@dataclass(frozen=True)
class TrajectorySegment:
    kind: SegmentKind
    start_time: float
    end_time: float
    start_cents: float
    end_cents: float
    mean_cents: float | None = None 

    @property
    def duration_s(self) -> float:
        return self.end_time - self.start_time

    @property
    def cents_change(self) -> float:
        return self.end_cents - self.start_cents


def segment_trajectory(
    points: list[PitchPoint],
    stability_window_cents: float = 35.0,
    stability_min_ms: float = 150.0,
    jump_threshold_cents: float = 150.0,
    jump_max_duration_ms: float = 60.0,
) -> list[TrajectorySegment]:
    
    if not points:
        return []

    segments: list[TrajectorySegment] = []
    i = 0
    n = len(points)

    while i < n:
        if not points[i].voiced:
            j = i
            while j < n and not points[j].voiced:
                j += 1
            segments.append(TrajectorySegment(
                kind="silence",
                start_time=points[i].timestamp,
                end_time=points[j - 1].timestamp,
                start_cents=points[i].cents_from_sa,
                end_cents=points[j - 1].cents_from_sa,
            ))
            i = j
            continue

        j = i
        while j < n and points[j].voiced:
            j += 1
        voiced_run = points[i:j]
        segments.extend(_segment_voiced_run(voiced_run, stability_window_cents,
                                             stability_min_ms, jump_threshold_cents,
                                             jump_max_duration_ms))
        i = j

    return segments


def _segment_voiced_run(
    run: list[PitchPoint],
    stability_window_cents: float,
    stability_min_ms: float,
    jump_threshold_cents: float,
    jump_max_duration_ms: float,
) -> list[TrajectorySegment]:
    if len(run) < 2:
        p = run[0]
        return [TrajectorySegment("unstable", p.timestamp, p.timestamp, p.cents_from_sa, p.cents_from_sa)]

    n = len(run)
    plateaus: list[tuple[int, int]] = []

    i = 0
    while i < n:
        j = i
        anchor = run[i].cents_from_sa
        while j + 1 < n and abs(run[j + 1].cents_from_sa - anchor) <= stability_window_cents:
            j += 1
        duration_ms = (run[j].timestamp - run[i].timestamp) * 1000.0
        if duration_ms >= stability_min_ms:
            plateaus.append((i, j))
            i = j + 1
        else:
            i += 1

    if not plateaus:
        return [_classify_transition(run, 0, n - 1, jump_threshold_cents, jump_max_duration_ms)]

    segments: list[TrajectorySegment] = []
    cursor = 0
    for (ps, pe) in plateaus:
        if ps > cursor:
            segments.append(_classify_transition(run, cursor, ps, jump_threshold_cents, jump_max_duration_ms))
        seg_points = run[ps:pe + 1]
        mean_cents = float(np.mean([p.cents_from_sa for p in seg_points]))
        segments.append(TrajectorySegment(
            kind="stable",
            start_time=run[ps].timestamp,
            end_time=run[pe].timestamp,
            start_cents=run[ps].cents_from_sa,
            end_cents=run[pe].cents_from_sa,
            mean_cents=mean_cents,
        ))
        cursor = pe
    if cursor < n - 1:
        segments.append(_classify_transition(run, cursor, n - 1, jump_threshold_cents, jump_max_duration_ms))

    return segments


def _classify_transition(
    run: list[PitchPoint],
    start_idx: int,
    end_idx: int,
    jump_threshold_cents: float,
    jump_max_duration_ms: float,
) -> TrajectorySegment:
    start_p, end_p = run[start_idx], run[end_idx]
    total_change = end_p.cents_from_sa - start_p.cents_from_sa

    is_jump = False
    for k in range(start_idx, end_idx):
        step_change = abs(run[k + 1].cents_from_sa - run[k].cents_from_sa)
        step_duration_ms = (run[k + 1].timestamp - run[k].timestamp) * 1000.0
        if step_change >= jump_threshold_cents and step_duration_ms <= jump_max_duration_ms:
            is_jump = True
            break

    if is_jump:
        kind: SegmentKind = "jump"
    elif abs(total_change) > 5.0 and _is_monotonic(run[start_idx:end_idx + 1]):
        kind = "glide"
    else:
        kind = "unstable"

    return TrajectorySegment(
        kind=kind,
        start_time=start_p.timestamp,
        end_time=end_p.timestamp,
        start_cents=start_p.cents_from_sa,
        end_cents=end_p.cents_from_sa,
    )


def _is_monotonic(points: list[PitchPoint], tolerance_cents: float = 15.0) -> bool:
    if len(points) < 2:
        return True
    diffs = np.diff([p.cents_from_sa for p in points])
    if len(diffs) == 0:
        return True
    n_up = int(np.sum(diffs > 0))
    n_down = int(np.sum(diffs < 0))
    dominant = max(n_up, n_down)
    return dominant / max(1, len(diffs)) >= 0.7
