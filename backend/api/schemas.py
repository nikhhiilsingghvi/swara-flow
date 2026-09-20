from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator

class StartSessionRequest(BaseModel):
    raga_name: str = Field(..., description="e.g. 'Bhupali', 'Yaman', 'Bhairav'")
    sa_hz: float = Field(..., gt=20.0, lt=2000.0, description="User-selected tonic frequency in Hz")
    taal_name: Optional[str] = Field(None, description="e.g. 'Teentaal'")

    @field_validator("raga_name")
    @classmethod
    def raga_name_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("raga_name must not be blank")
        return v.strip()

class SessionResponse(BaseModel):
    session_id: str
    raga_name: str
    sa_hz: float
    taal_name: Optional[str]
    status: str
    created_at: float
    updated_at: float
    total_active_duration_s: float

class SwaraEventOut(BaseModel):
    timestamp: float
    swara: Optional[str]
    octave: int
    frequency_hz: float
    target_cents: Optional[float]
    detected_cents: float
    deviation_cents: Optional[float]
    confidence: float

class VisionEventOut(BaseModel):
    timestamp: float
    head_motion: float
    shoulder_tilt: Optional[float]
    body_motion: float
    hand_motion: float
    face_detected: bool
    pose_available: bool

class DetectedIssueOut(BaseModel):
    issue_type: str
    description: str
    swara: Optional[str]
    mean_deviation_cents: Optional[float]
    confidence: float

class GeneratedExerciseOut(BaseModel):
    title: str
    swara_sequence: list[str]
    rationale: str
    suggested_tempo: str

class FinishedSessionResponse(BaseModel):
    session_id: str
    status: str
    swara_events: list[SwaraEventOut]
    detected_issues: list[DetectedIssueOut]
    generated_exercises: list[GeneratedExerciseOut]
    progress_metrics: dict
    coaching_text: str

class RagaSummary(BaseModel):
    name: str
    thaat: str
    arohana: list[str]
    avarohana: list[str]
    allowed_swaras: list[str]
    time_of_day: str

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    recoverable: bool = True

class ClientAudioChunk(BaseModel):
    type: Literal["audio_chunk"] = "audio_chunk"
    timestamp: float
    sample_rate: int
    pcm16_base64: str

class ClientVideoFrame(BaseModel):
    type: Literal["video_frame"] = "video_frame"
    timestamp: float
    width: int
    height: int
    jpeg_base64: str

class ClientControlMessage(BaseModel):
    type: Literal["control"] = "control"
    action: Literal["start", "pause", "resume", "finish"]

ClientMessage = ClientAudioChunk | ClientVideoFrame | ClientControlMessage

class ServerPitchUpdate(BaseModel):
    type: Literal["pitch_update"] = "pitch_update"
    timestamp: float
    frequency_hz: float
    confidence: float
    voiced: bool
    swara: Optional[str]
    deviation_cents: Optional[float]

class ServerVisionUpdate(BaseModel):
    type: Literal["vision_update"] = "vision_update"
    timestamp: float
    face_detected: bool
    pose_available: bool
    hand_motion: float
    warnings: list[str] = Field(default_factory=list)

class ServerLatencyUpdate(BaseModel):
    type: Literal["latency_update"] = "latency_update"
    capture_latency_ms: float
    processing_latency_ms: float
    websocket_latency_ms: float
    end_to_end_ms: float
    within_realtime_budget: bool

class ServerErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    error: str
    recoverable: bool = True

ServerMessage = ServerPitchUpdate | ServerVisionUpdate | ServerLatencyUpdate | ServerErrorMessage
