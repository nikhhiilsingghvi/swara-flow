from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from core.config import get_config
from music.raga import get_raga_library
from sessions.event_store import EventStore
from sessions.schemas import list_sessions, load_session

from .schemas import (
    DetectedIssueOut,
    FinishedSessionResponse,
    GeneratedExerciseOut,
    RagaSummary,
    SwaraEventOut,
)

router = APIRouter()

@router.get("/ragas", response_model=list[RagaSummary])
def list_ragas() -> list[RagaSummary]:
    library = get_raga_library()
    return [
        RagaSummary(
            name=r.name, thaat=r.thaat, arohana=r.arohana, avarohana=r.avarohana,
            allowed_swaras=r.allowed_swaras, time_of_day=r.time_of_day,
        )
        for r in library.all()
    ]

@router.get("/ragas/{raga_name}", response_model=RagaSummary)
def get_raga(raga_name: str) -> RagaSummary:
    try:
        r = get_raga_library().get(raga_name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return RagaSummary(
        name=r.name, thaat=r.thaat, arohana=r.arohana, avarohana=r.avarohana,
        allowed_swaras=r.allowed_swaras, time_of_day=r.time_of_day,
    )

@router.get("/sessions")
def list_all_sessions() -> list[str]:
    cfg = get_config()
    return list_sessions(cfg.paths.sessions_dir)

@router.get("/sessions/{session_id}", response_model=FinishedSessionResponse)
def get_session(session_id: str) -> FinishedSessionResponse:
    cfg = get_config()
    try:
        session = load_session(session_id, cfg.paths.sessions_dir)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    return FinishedSessionResponse(
        session_id=session.metadata.session_id,
        status=session.metadata.status.value,
        swara_events=[SwaraEventOut(**vars(e)) for e in session.swara_events],
        detected_issues=[
            DetectedIssueOut(
                issue_type=i.issue_type, description=i.description, swara=i.swara,
                mean_deviation_cents=i.mean_deviation_cents, confidence=i.confidence,
            ) for i in session.detected_issues
        ],
        generated_exercises=[
            GeneratedExerciseOut(
                title=e.title, swara_sequence=e.swara_sequence,
                rationale=e.rationale, suggested_tempo=e.suggested_tempo,
            ) for e in session.generated_exercises
        ],
        progress_metrics=session.progress_metrics,
        coaching_text=session.progress_metrics.get("coaching_text", ""),
    )

@router.get("/sessions/{session_id}/replay")
def get_session_replay(session_id: str, start_time: float = 0.0, end_time: float = 1e12) -> list[dict]:
    cfg = get_config()
    events = EventStore.read_range(session_id, cfg.paths.data_dir / "sessions", start_time, end_time)
    if not events:
        all_events = EventStore.read_all(session_id, cfg.paths.data_dir / "sessions")
        if not all_events:
            raise HTTPException(status_code=404, detail=f"No recorded events for session {session_id}")
    return [{"timestamp": e.timestamp, "kind": e.kind, "payload": e.payload} for e in events]

@router.get("/sessions/{session_id}/export.{fmt}")
def export_session(session_id: str, fmt: str) -> StreamingResponse:
    cfg = get_config()
    try:
        session = load_session(session_id, cfg.paths.sessions_dir)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    if fmt == "json":
        content = json.dumps(session.to_dict(), indent=2)
        return StreamingResponse(io.StringIO(content), media_type="application/json")

    if fmt == "csv":
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["timestamp", "swara", "frequency_hz", "target_cents", "detected_cents",
                          "deviation_cents", "confidence"])
        for e in session.swara_events:
            writer.writerow([e.timestamp, e.swara, e.frequency_hz, e.target_cents,
                              e.detected_cents, e.deviation_cents, e.confidence])
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="text/csv")

    raise HTTPException(status_code=400, detail=f"Unsupported export format: {fmt}. Use 'json' or 'csv'.")

@router.get("/demo/{raga_name}")
def get_demo_session(raga_name: str) -> FinishedSessionResponse:
    from demo import generate_demo_session

    try:
        result = generate_demo_session(raga_name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    session = result.session
    return FinishedSessionResponse(
        session_id=session.metadata.session_id,
        status=session.metadata.status.value,
        swara_events=[SwaraEventOut(**vars(e)) for e in session.swara_events],
        detected_issues=[
            DetectedIssueOut(
                issue_type=i.issue_type, description=i.description, swara=i.swara,
                mean_deviation_cents=i.mean_deviation_cents, confidence=i.confidence,
            ) for i in session.detected_issues
        ],
        generated_exercises=[
            GeneratedExerciseOut(
                title=e.title, swara_sequence=e.swara_sequence,
                rationale=e.rationale, suggested_tempo=e.suggested_tempo,
            ) for e in session.generated_exercises
        ],
        progress_metrics=session.progress_metrics,
        coaching_text="",
    )
