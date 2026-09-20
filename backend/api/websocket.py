from __future__ import annotations

import base64
import time
from typing import Optional

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from audio.pitch import PitchEstimator, build_default_pitch_estimator
from core.config import get_config
from core.sync import LatencyBudget
from music.raga import Raga
from music.swara_mapping import estimate_swara
from sessions.schemas import RhythmEvent, SwaraEvent, VisionEvent
from sessions.session_manager import SessionManager
from vision.camera import preprocess_frame
from vision.face import HaarFaceDetector
from vision.hands import HandFrame, build_hand_detector
from vision.motion import OpticalFlowMotionAnalyzer
from vision.pose import PoseFrame, build_pose_detector

from .schemas import (
    ClientAudioChunk,
    ClientControlMessage,
    ClientVideoFrame,
    ServerErrorMessage,
    ServerLatencyUpdate,
    ServerPitchUpdate,
    ServerVisionUpdate,
)

class LiveSessionHandler:
    def __init__(self, session_manager: SessionManager, raga: Raga,
                 pose_model_path: Optional[str] = None, hand_model_path: Optional[str] = None):
        self.manager = session_manager
        self.raga = raga
        cfg = get_config()

        self.pitch_estimator: PitchEstimator = build_default_pitch_estimator(cfg.audio)
        self.face_detector = HaarFaceDetector(cfg.vision)
        self.motion_analyzer = OpticalFlowMotionAnalyzer(history_frames=cfg.vision.motion_history_frames)
        self.pose_detector = build_pose_detector(pose_model_path, cfg.vision)
        self.hand_detector = build_hand_detector(hand_model_path, cfg.vision)

        self._last_pose: Optional[PoseFrame] = None
        self._last_hands: Optional[HandFrame] = None

    def handle_audio_chunk(self, msg: ClientAudioChunk) -> ServerPitchUpdate:
        t0 = time.monotonic()
        raw = base64.b64decode(msg.pcm16_base64)
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0

        pitch_frame = self.pitch_estimator.estimate(samples, timestamp=msg.timestamp)

        swara_name = None
        deviation = None
        if pitch_frame.voiced:
            est = estimate_swara(
                pitch_frame.frequency_hz, self.manager.session.metadata.sa_hz,
                max_snap_distance_cents=get_config().swara.max_snap_distance_cents,
                detector_confidence=pitch_frame.confidence,
            )
            swara_name = est.swara
            deviation = est.deviation_cents

            if self.manager.session.metadata.status.value == "active":
                self.manager.add_swara_event(SwaraEvent(
                    timestamp=pitch_frame.timestamp,
                    swara=est.swara,
                    octave=est.octave,
                    frequency_hz=pitch_frame.frequency_hz,
                    target_cents=est.target_cents,
                    detected_cents=est.detected_cents,
                    deviation_cents=est.deviation_cents,
                    confidence=est.confidence,
                ))

        processing_ms = (time.monotonic() - t0) * 1000.0
        return ServerPitchUpdate(
            timestamp=pitch_frame.timestamp,
            frequency_hz=pitch_frame.frequency_hz,
            confidence=pitch_frame.confidence,
            voiced=pitch_frame.voiced,
            swara=swara_name,
            deviation_cents=deviation,
        )

    def handle_video_frame(self, msg: ClientVideoFrame) -> ServerVisionUpdate:
        import cv2

        jpeg_bytes = base64.b64decode(msg.jpeg_base64)
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        warnings: list[str] = []
        if frame_bgr is None:
            return ServerVisionUpdate(
                timestamp=msg.timestamp, face_detected=False, pose_available=False,
                hand_motion=0.0, warnings=["failed to decode video frame"],
            )

        pre = preprocess_frame(frame_bgr, timestamp=msg.timestamp)

        from vision.camera import check_lighting_quality
        lighting = check_lighting_quality(pre.gray)
        if not lighting["is_acceptable"]:
            if lighting["too_dark"]:
                warnings.append("Lighting is too low for reliable face tracking.")
            elif lighting["too_bright"]:
                warnings.append("Frame is overexposed.")
            elif lighting["low_contrast"]:
                warnings.append("Low contrast image -- check camera exposure.")

        face = self.face_detector.detect(pre.gray)
        if face.multiple_faces_detected:
            warnings.append("Multiple faces detected -- only the largest is tracked.")

        pose = self.pose_detector.detect(pre.rgb)
        hands = self.hand_detector.detect(pre.rgb)
        if not pose.available and pose.unavailable_reason:
            warnings.append(f"Posture tracking unavailable: {pose.unavailable_reason}")

        regions = {}
        if face.found and face.bbox:
            regions["head"] = face.bbox
        motion = self.motion_analyzer.analyze(pre.gray, msg.timestamp, regions=regions)

        self._last_pose, self._last_hands = pose, hands

        if self.manager.session.metadata.status.value == "active":
            self.manager.add_vision_event(VisionEvent(
                timestamp=msg.timestamp,
                head_motion=motion.region_motion.get("head", motion.mean_motion_magnitude),
                shoulder_tilt=pose.shoulder_tilt_deg,
                body_motion=motion.mean_motion_magnitude,
                hand_motion=0.0,  
                face_detected=face.found,
                pose_available=pose.available,
            ))

        return ServerVisionUpdate(
            timestamp=msg.timestamp,
            face_detected=face.found,
            pose_available=pose.available,
            hand_motion=motion.region_motion.get("hands", 0.0),
            warnings=warnings,
        )

    def handle_control(self, msg: ClientControlMessage) -> None:
        action_map = {
            "start": self.manager.start,
            "pause": self.manager.pause,
            "resume": self.manager.resume,
            "finish": self.manager.finish,
        }
        action_map[msg.action]()


async def websocket_session_loop(websocket: WebSocket, handler: LiveSessionHandler) -> None:
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_json()
            msg_type = raw.get("type")
            try:
                if msg_type == "audio_chunk":
                    response = handler.handle_audio_chunk(ClientAudioChunk(**raw))
                elif msg_type == "video_frame":
                    response = handler.handle_video_frame(ClientVideoFrame(**raw))
                elif msg_type == "control":
                    handler.handle_control(ClientControlMessage(**raw))
                    continue
                else:
                    await websocket.send_json(
                        ServerErrorMessage(error=f"Unknown message type: {msg_type}").model_dump()
                    )
                    continue
                await websocket.send_json(response.model_dump())
            except Exception as e:
                await websocket.send_json(
                    ServerErrorMessage(error=str(e), recoverable=True).model_dump()
                )
    except WebSocketDisconnect:
        pass
