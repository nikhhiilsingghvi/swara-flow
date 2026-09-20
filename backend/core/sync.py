from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

A = TypeVar("A")
B = TypeVar("B")

@dataclass(frozen=True)
class AlignedPair(Generic[A, B]):
    timestamp: float
    audio_event: A
    vision_event: B | None
    offset_ms: float | None

def align(
    audio_events: list[A],
    vision_events: list[B],
    audio_timestamp: Callable[[A], float],
    vision_timestamp: Callable[[B], float],
    tolerance_ms: float,
) -> list[AlignedPair]:

    tolerance_s = tolerance_ms / 1000.0
    result: list[AlignedPair] = []
    v_idx = 0
    n_vision = len(vision_events)

    for a_event in audio_events:
        a_t = audio_timestamp(a_event)

        while (
            v_idx + 1 < n_vision
            and abs(vision_timestamp(vision_events[v_idx + 1]) - a_t)
            <= abs(vision_timestamp(vision_events[v_idx]) - a_t)
        ):
            v_idx += 1

        if n_vision == 0:
            result.append(AlignedPair(a_t, a_event, None, None))
            continue

        v_event = vision_events[v_idx]
        gap_s = vision_timestamp(v_event) - a_t
        gap_ms = gap_s * 1000.0

        if abs(gap_s) <= tolerance_s:
            result.append(AlignedPair(a_t, a_event, v_event, gap_ms))
        else:
            result.append(AlignedPair(a_t, a_event, None, gap_ms))

    return result

@dataclass(frozen=True)
class LatencyBudget:
    capture_latency_ms: float
    processing_latency_ms: float
    websocket_latency_ms: float
    visualization_latency_ms: float

    @property
    def end_to_end_ms(self) -> float:
        return (
            self.capture_latency_ms
            + self.processing_latency_ms
            + self.websocket_latency_ms
            + self.visualization_latency_ms
        )

    def is_within_realtime_budget(self, budget_ms: float = 150.0) -> bool:
        return self.end_to_end_ms <= budget_ms
