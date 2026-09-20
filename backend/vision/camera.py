from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.config import VisionConfig, get_config

@dataclass(frozen=True)
class PreprocessedFrame:
    timestamp: float
    bgr: np.ndarray 
    rgb: np.ndarray         
    gray: np.ndarray       
    original_shape: tuple[int, int] 
    scale_factor: float 

def preprocess_frame(
    frame_bgr: np.ndarray,
    timestamp: float,
    config: VisionConfig | None = None,
) -> PreprocessedFrame:
    cfg = config or get_config().vision
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("preprocess_frame received an empty frame")

    original_h, original_w = frame_bgr.shape[:2]

    if original_w > cfg.max_frame_width:
        scale_factor = cfg.max_frame_width / original_w
        new_w = cfg.max_frame_width
        new_h = int(round(original_h * scale_factor))
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        scale_factor = 1.0
        resized = frame_bgr

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    gray_raw = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray_raw, d=5, sigmaColor=50, sigmaSpace=50)

    return PreprocessedFrame(
        timestamp=timestamp,
        bgr=resized,
        rgb=rgb,
        gray=gray,
        original_shape=(original_h, original_w),
        scale_factor=scale_factor,
    )

def check_lighting_quality(gray_frame: np.ndarray) -> dict:
    mean_brightness = float(np.mean(gray_frame))
    std_brightness = float(np.std(gray_frame))
    too_dark = mean_brightness < 40.0
    too_bright = mean_brightness > 220.0
    low_contrast = std_brightness < 15.0
    return {
        "mean_brightness": mean_brightness,
        "std_brightness": std_brightness,
        "too_dark": too_dark,
        "too_bright": too_bright,
        "low_contrast": low_contrast,
        "is_acceptable": not (too_dark or too_bright or low_contrast),
    }
