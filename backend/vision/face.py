from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from core.config import VisionConfig, get_config

@dataclass(frozen=True)
class FaceDetection:
    found: bool
    bbox: Optional[tuple[int, int, int, int]] 
    center_x: Optional[float]                     
    center_y: Optional[float]
    size_ratio: Optional[float]                   
    approx_roll_deg: Optional[float]                
    multiple_faces_detected: bool
    backend: str                                    

class HaarFaceDetector:
    def __init__(self, config: Optional[VisionConfig] = None):
        self.config = config or get_config().vision
        face_cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt2.xml")
        eye_cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")

        self._face_cascade = cv2.CascadeClassifier(face_cascade_path)
        if self._face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {face_cascade_path}")

        self._eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        
    def detect(self, gray_frame: np.ndarray) -> FaceDetection:
        h, w = gray_frame.shape[:2]
        faces = self._face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(int(w * 0.1), int(h * 0.1)),
        )

        if len(faces) == 0:
            return FaceDetection(
                found=False, bbox=None, center_x=None, center_y=None,
                size_ratio=None, approx_roll_deg=None,
                multiple_faces_detected=False, backend="haar_cascade",
            )

        multiple = len(faces) > 1
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])

        center_x = (fx + fw / 2) / w
        center_y = (fy + fh / 2) / h
        size_ratio = fw / w

        roll_deg = None
        if not self._eye_cascade.empty():
            face_roi = gray_frame[fy:fy + fh, fx:fx + fw]
            eyes = self._eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=8)
            if len(eyes) >= 2:
                eyes_sorted = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
                eyes_sorted = sorted(eyes_sorted, key=lambda e: e[0])  
                (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = eyes_sorted
                c1 = (ex1 + ew1 / 2, ey1 + eh1 / 2)
                c2 = (ex2 + ew2 / 2, ey2 + eh2 / 2)
                dx, dy = c2[0] - c1[0], c2[1] - c1[1]
                if dx != 0:
                    roll_deg = float(np.degrees(np.arctan2(dy, dx)))

        return FaceDetection(
            found=True,
            bbox=(int(fx), int(fy), int(fw), int(fh)),
            center_x=float(center_x),
            center_y=float(center_y),
            size_ratio=float(size_ratio),
            approx_roll_deg=roll_deg,
            multiple_faces_detected=multiple,
            backend="haar_cascade",
        )

class MediaPipeFaceDetector:
    def __init__(self, model_path: str, config: Optional[VisionConfig] = None):
        self.config = config or get_config().vision
        if not os.path.exists(model_path):
            raise ModelUnavailableError(
                f"MediaPipe face landmark model not found at {model_path}. "
                f"Run scripts/download_models.* once (requires network) to fetch it, "
                f"or the app will keep using the Haar-cascade fallback."
            )
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=self.config.face_detector_confidence,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._mp = mp

    def detect(self, rgb_frame: np.ndarray) -> FaceDetection:
        h, w = rgb_frame.shape[:2]
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return FaceDetection(
                found=False, bbox=None, center_x=None, center_y=None,
                size_ratio=None, approx_roll_deg=None,
                multiple_faces_detected=False, backend="mediapipe",
            )

        landmarks = result.face_landmarks[0]
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        center_x, center_y = float(np.mean(xs)), float(np.mean(ys))
        size_ratio = float(max(xs) - min(xs))

        roll_deg = None
        if result.facial_transformation_matrixes:
            matrix = np.array(result.facial_transformation_matrixes[0]).reshape(4, 4)
            rot = matrix[:3, :3]
            roll_deg = float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))

        return FaceDetection(
            found=True,
            bbox=(int(min(xs) * w), int(min(ys) * h), int((max(xs) - min(xs)) * w), int((max(ys) - min(ys)) * h)),
            center_x=center_x,
            center_y=center_y,
            size_ratio=size_ratio,
            approx_roll_deg=roll_deg,
            multiple_faces_detected=len(result.face_landmarks) > 1,
            backend="mediapipe",
        )

class ModelUnavailableError(RuntimeError):
    """Raised when a MediaPipe model file hasn't been downloaded yet."""

def build_default_face_detector(model_path: Optional[str] = None, config: Optional[VisionConfig] = None):
    if model_path:
        try:
            return MediaPipeFaceDetector(model_path, config)
        except ModelUnavailableError:
            pass
    return HaarFaceDetector(config)
