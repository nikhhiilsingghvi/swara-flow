from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.config import VisionConfig, get_config

@dataclass(frozen=True)
class PoseFrame:
    available: bool
    unavailable_reason: Optional[str]
    left_shoulder: Optional[tuple[float, float]]    
    right_shoulder: Optional[tuple[float, float]]
    nose: Optional[tuple[float, float]]
    shoulder_tilt_deg: Optional[float]                
    torso_lean_deg: Optional[float]                    
    posture_stability: Optional[float]                  

class MediaPipePoseDetector:
    def __init__(self, model_path: str, config: Optional[VisionConfig] = None):
        self.config = config or get_config().vision
        if not os.path.exists(model_path):
            raise ModelUnavailableError(
                f"MediaPipe pose landmark model not found at {model_path}. "
                f"Run scripts/download_models.* once (requires network)."
            )
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._mp = mp

    _LEFT_SHOULDER, _RIGHT_SHOULDER, _NOSE = 11, 12, 0
    _LEFT_HIP, _RIGHT_HIP = 23, 24

    def detect(self, rgb_frame: np.ndarray) -> PoseFrame:
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return PoseFrame(
                available=False, unavailable_reason="no person detected in frame",
                left_shoulder=None, right_shoulder=None, nose=None,
                shoulder_tilt_deg=None, torso_lean_deg=None, posture_stability=None,
            )

        lm = result.pose_landmarks[0]
        ls = (lm[self._LEFT_SHOULDER].x, lm[self._LEFT_SHOULDER].y)
        rs = (lm[self._RIGHT_SHOULDER].x, lm[self._RIGHT_SHOULDER].y)
        nose = (lm[self._NOSE].x, lm[self._NOSE].y)

        dx, dy = rs[0] - ls[0], rs[1] - ls[1]
        shoulder_tilt = float(np.degrees(np.arctan2(dy, dx))) if dx != 0 or dy != 0 else 0.0

        lh = (lm[self._LEFT_HIP].x, lm[self._LEFT_HIP].y)
        rh = (lm[self._RIGHT_HIP].x, lm[self._RIGHT_HIP].y)
        shoulder_mid_x = (ls[0] + rs[0]) / 2
        hip_mid_x = (lh[0] + rh[0]) / 2
        torso_lean = float(np.degrees(np.arctan2(shoulder_mid_x - hip_mid_x, 1.0)))

        return PoseFrame(
            available=True, unavailable_reason=None,
            left_shoulder=ls, right_shoulder=rs, nose=nose,
            shoulder_tilt_deg=shoulder_tilt, torso_lean_deg=torso_lean,
            posture_stability=None,
        )

class ModelUnavailableError(RuntimeError):
    pass


def build_pose_detector(model_path: Optional[str], config: Optional[VisionConfig] = None):
    if model_path:
        try:
            return MediaPipePoseDetector(model_path, config)
        except ModelUnavailableError as e:
            return _UnavailablePoseDetector(str(e))
    return _UnavailablePoseDetector(
        "No pose model configured. Set SF_POSE_MODEL_PATH and run "
        "scripts/download_models.* to enable posture tracking."
    )

class _UnavailablePoseDetector:
    def __init__(self, reason: str):
        self._reason = reason

    def detect(self, rgb_frame: np.ndarray) -> PoseFrame:
        return PoseFrame(
            available=False, unavailable_reason=self._reason,
            left_shoulder=None, right_shoulder=None, nose=None,
            shoulder_tilt_deg=None, torso_lean_deg=None, posture_stability=None,
        )
