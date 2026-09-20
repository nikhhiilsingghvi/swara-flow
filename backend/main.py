from __future__ import annotations

import logging
import os

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router as api_router
from api.websocket import LiveSessionHandler, websocket_session_loop
from core.config import get_config
from core.logging import configure_logging
from music.raga import get_raga_library
from sessions.session_manager import SessionManager

configure_logging()
logger = logging.getLogger("swaraflow")

app = FastAPI(
    title="SwaraFlow API",
    description="Multimodal AI coaching system for Indian classical vocal practice.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("SF_CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

_active_sessions: dict[str, SessionManager] = {}


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "llm_provider": get_config().llm.provider}

@app.websocket("/ws/session/{raga_name}")
async def session_websocket(websocket: WebSocket, raga_name: str, sa_hz: float = 220.0, taal_name: str | None = None):
    try:
        raga = get_raga_library().get(raga_name)
    except KeyError:
        await websocket.close(code=4404, reason=f"Unknown raga: {raga_name}")
        return

    manager = SessionManager(raga_name=raga.name, sa_hz=sa_hz, taal_name=taal_name)
    _active_sessions[manager.session.metadata.session_id] = manager

    cfg = get_config()
    handler = LiveSessionHandler(
        session_manager=manager,
        raga=raga,
        pose_model_path=os.getenv("SF_POSE_MODEL_PATH"),
        hand_model_path=os.getenv("SF_HAND_MODEL_PATH"),
    )

    try:
        await websocket_session_loop(websocket, handler)
    finally:
        # persist whatever was captured, even on an abrupt disconnect,
        # so a dropped connection doesn't silently lose the session
        if manager.session.metadata.status.value in ("active", "paused"):
            try:
                manager.finish()
            except Exception:
                logger.exception("Failed to auto-finish session %s on disconnect", manager.session.metadata.session_id)
        try:
            manager.save(cfg.paths.sessions_dir)
        except Exception:
            logger.exception("Failed to save session %s on disconnect", manager.session.metadata.session_id)
        _active_sessions.pop(manager.session.metadata.session_id, None)
