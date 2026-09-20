from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.config import VisionConfig, get_config

@dataclass(frozen=True)
class HandFrame:
    available: bool
    unavailable_reason: Optional[str]
    left_wrist: Optional[tuple[float, float]]   
    right_wrist: Optional[tuple[float, float]]
    num_hands_detected: int

class MediaPipeHandDetector:
    _WRIST_IDX = 0 

    def __init__(self, model_path: str, config: Optional[VisionConfig] = None):
        self.config = config or get_config().vision
        if not os.path.exists(model_path):
            raise ModelUnavailableError(
                f"MediaPipe hand landmark model not found at {model_path}. "
                f"Run scripts/download_models.* once (requires network)."
            )
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=self.config.face_detector_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._mp = mp

    def detect(self, rgb_frame: np.ndarray) -> HandFrame:
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return HandFrame(available=True, unavailable_reason=None,
                              left_wrist=None, right_wrist=None, num_hands_detected=0)

        left_wrist, right_wrist = None, None
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            wrist_lm = landmarks[self._WRIST_IDX]
            label = handedness[0].category_name 
            if label == "Left":
                left_wrist = (wrist_lm.x, wrist_lm.y)
            else:
                right_wrist = (wrist_lm.x, wrist_lm.y)

        return HandFrame(
            available=True, unavailable_reason=None,
            left_wrist=left_wrist, right_wrist=right_wrist,
            num_hands_detected=len(result.hand_landmarks),
        )

class ModelUnavailableError(RuntimeError):
    pass

def build_hand_detector(model_path: Optional[str], config: Optional[VisionConfig] = None):
    if model_path:
        try:
            return MediaPipeHandDetector(model_path, config)
        except ModelUnavailableError as e:
            return _UnavailableHandDetector(str(e))
    return _UnavailableHandDetector(
        "No hand-tracking model configured. Set SF_HAND_MODEL_PATH and run "
        "scripts/download_models.* to enable hand tracking."
    )

class _UnavailableHandDetector:
    def __init__(self, reason: str):
        self._reason = reason

    def detect(self, rgb_frame: np.ndarray) -> HandFrame:
        return HandFrame(available=False, unavailable_reason=self._reason,
                          left_wrist=None, right_wrist=None, num_hands_detected=0)

def wrist_velocity(prev: Optional[tuple[float, float]], curr: Optional[tuple[float, float]],
                    dt_s: float, frame_diagonal_px: float = 1.0) -> Optional[float]:
    if prev is None or curr is None or dt_s <= 0:
        return None
    dx, dy = curr[0] - prev[0], curr[1] - prev[1]
    dist = float(np.hypot(dx, dy))
    return dist / dt_s
