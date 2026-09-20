from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

@dataclass(frozen=True)
class MotionFrame:
    timestamp: float
    mean_motion_magnitude: float     
    motion_std: float                  
    dominant_direction_deg: Optional[float]  
    region_motion: dict[str, float]     

class OpticalFlowMotionAnalyzer:
    def __init__(self, history_frames: int = 15, motion_negligible_threshold: float = 0.15):
        self._prev_gray: Optional[np.ndarray] = None
        self._history: deque[MotionFrame] = deque(maxlen=history_frames)
        self._motion_negligible_threshold = motion_negligible_threshold

    def analyze(
        self,
        gray_frame: np.ndarray,
        timestamp: float,
        regions: Optional[dict[str, tuple[int, int, int, int]]] = None,
    ) -> MotionFrame:
        if self._prev_gray is None or self._prev_gray.shape != gray_frame.shape:
            self._prev_gray = gray_frame
            frame = MotionFrame(timestamp, 0.0, 0.0, None, {})
            self._history.append(frame)
            return frame

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray_frame, None,
            pyr_scale=0.5, levels=4, winsize=21, iterations=3,
            poly_n=7, poly_sigma=1.5, flags=0,
        )
        self._prev_gray = gray_frame

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_mag = float(np.mean(magnitude))
        std_mag = float(np.std(magnitude))

        dominant_dir = None
        if mean_mag > self._motion_negligible_threshold:
            mean_dx = float(np.mean(flow[..., 0]))
            mean_dy = float(np.mean(flow[..., 1]))
            dominant_dir = float(np.degrees(np.arctan2(mean_dy, mean_dx)))

        region_motion = {}
        if regions:
            for name, (x, y, w, h) in regions.items():
                x0, y0 = max(0, x), max(0, y)
                x1, y1 = min(flow.shape[1], x + w), min(flow.shape[0], y + h)
                if x1 > x0 and y1 > y0:
                    region_mag = magnitude[y0:y1, x0:x1]
                    region_motion[name] = float(np.mean(region_mag))
                else:
                    region_motion[name] = 0.0

        frame = MotionFrame(
            timestamp=timestamp,
            mean_motion_magnitude=mean_mag,
            motion_std=std_mag,
            dominant_direction_deg=dominant_dir,
            region_motion=region_motion,
        )
        self._history.append(frame)
        return frame

    def recent_motion_trend(self) -> float:
        if not self._history:
            return 0.0
        return float(np.mean([f.mean_motion_magnitude for f in self._history]))

    def reset(self) -> None:
        self._prev_gray = None
        self._history.clear()
