from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .face import FaceDetection
from .hands import HandFrame
from .motion import MotionFrame
from .pose import PoseFrame

COLOR_FACE_BOX = (80, 220, 120)     # soft green
COLOR_SHOULDER_LINE = (220, 160, 60)  # muted amber
COLOR_WRIST = (200, 90, 220)          # magenta
COLOR_WARNING_TEXT = (60, 60, 230)     # red
COLOR_INFO_TEXT = (230, 230, 230)      # near-white

def draw_face_overlay(frame_bgr: np.ndarray, face: FaceDetection) -> np.ndarray:
    out = frame_bgr.copy()
    if not face.found or face.bbox is None:
        return out
    x, y, w, h = face.bbox
    cv2.rectangle(out, (x, y), (x + w, y + h), COLOR_FACE_BOX, 2)
    if face.multiple_faces_detected:
        cv2.putText(out, "Multiple faces detected", (x, max(0, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WARNING_TEXT, 1, cv2.LINE_AA)
    return out

def draw_pose_overlay(frame_bgr: np.ndarray, pose: PoseFrame) -> np.ndarray:
    out = frame_bgr.copy()
    if not pose.available or pose.left_shoulder is None or pose.right_shoulder is None:
        return out
    h, w = out.shape[:2]
    lx, ly = int(pose.left_shoulder[0] * w), int(pose.left_shoulder[1] * h)
    rx, ry = int(pose.right_shoulder[0] * w), int(pose.right_shoulder[1] * h)
    cv2.line(out, (lx, ly), (rx, ry), COLOR_SHOULDER_LINE, 3)
    cv2.circle(out, (lx, ly), 5, COLOR_SHOULDER_LINE, -1)
    cv2.circle(out, (rx, ry), 5, COLOR_SHOULDER_LINE, -1)
    return out

def draw_hand_overlay(frame_bgr: np.ndarray, hands: HandFrame) -> np.ndarray:
    out = frame_bgr.copy()
    if not hands.available:
        return out
    h, w = out.shape[:2]
    for wrist in (hands.left_wrist, hands.right_wrist):
        if wrist is not None:
            cx, cy = int(wrist[0] * w), int(wrist[1] * h)
            cv2.circle(out, (cx, cy), 8, COLOR_WRIST, 2)
    return out

def draw_status_text(
    frame_bgr: np.ndarray,
    lines: list[str],
    origin: tuple[int, int] = (10, 20),
    color: tuple[int, int, int] = COLOR_INFO_TEXT,
) -> np.ndarray:
    out = frame_bgr.copy()
    x, y = origin
    for i, line in enumerate(lines):
        cv2.putText(out, line, (x, y + i * 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)
    return out

def draw_unavailable_banner(frame_bgr: np.ndarray, reason: str) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    cv2.putText(out, f"[degraded] {reason}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WARNING_TEXT, 1, cv2.LINE_AA)
    return out

def compose_full_overlay(
    frame_bgr: np.ndarray,
    face: Optional[FaceDetection] = None,
    pose: Optional[PoseFrame] = None,
    hands: Optional[HandFrame] = None,
    motion: Optional[MotionFrame] = None,
    status_lines: Optional[list[str]] = None,
) -> np.ndarray:
    out = frame_bgr
    if face is not None:
        out = draw_face_overlay(out, face)
    if pose is not None:
        out = draw_pose_overlay(out, pose)
        if not pose.available and pose.unavailable_reason:
            out = draw_unavailable_banner(out, pose.unavailable_reason)
    if hands is not None:
        out = draw_hand_overlay(out, hands)
    if status_lines:
        out = draw_status_text(out, status_lines)
    return out
