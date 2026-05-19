"""
FastAPI shell for a FloodDiffusion-style streaming demo.

The heavy models are represented by simulated HF Endpoint clients so the
frontend, session flow, WebSocket motion stream, and budget gate can be tested
locally before real endpoint URLs are wired in.
"""

import asyncio
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
FRAME_RATE = 20
FRAME_INTERVAL = 1.0 / FRAME_RATE
PUBLIC_DAILY_BUDGET_SECONDS = 180

app = FastAPI(title="FloodScene HF Endpoint Simulation")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


class StartSessionRequest(BaseModel):
    text: str = Field(default="walk forward and wave to the camera")
    audio_enabled: bool = True
    video_enabled: bool = True
    endpoint_mode: str = Field(default="hf-endpoint-sim")


class TextUpdateRequest(BaseModel):
    text: str


@dataclass
class SessionState:
    session_id: str
    prompt: str
    transcript: str
    scene_context: str
    audio_enabled: bool
    video_enabled: bool
    endpoint_mode: str
    created_at: float = field(default_factory=time.time)
    running: bool = True
    frame_id: int = 0
    audio_level: float = 0.35
    video_energy: float = 0.25
    status_events: list[dict[str, Any]] = field(default_factory=list)


class SimulatedHFEndpointClients:
    """Drop-in stand-ins for ASR, VLM, TTS, and FloodDiffusion endpoints."""

    async def asr(self, audio_enabled: bool) -> dict[str, Any]:
        await asyncio.sleep(0.15)
        if not audio_enabled:
            return {
                "text": "",
                "confidence": 0.0,
                "source": "disabled",
            }
        return {
            "text": "the speaker asks for a relaxed walk that can change direction",
            "confidence": 0.91,
            "source": "hf-endpoint-sim-asr",
        }

    async def vlm(self, video_enabled: bool) -> dict[str, Any]:
        await asyncio.sleep(0.18)
        if not video_enabled:
            return {
                "context": "no visual context",
                "confidence": 0.0,
                "source": "disabled",
            }
        return {
            "context": "indoor scene, one person framed full body, medium camera motion",
            "confidence": 0.88,
            "source": "hf-endpoint-sim-vlm",
        }

    async def tts(self, text: str) -> dict[str, Any]:
        await asyncio.sleep(0.08)
        return {
            "audio_url": None,
            "duration_sec": min(5.0, max(1.0, len(text) / 32.0)),
            "source": "hf-endpoint-sim-tts",
        }


clients = SimulatedHFEndpointClients()
sessions: dict[str, SessionState] = {}
budget_used_seconds = 0.0
budget_lock = asyncio.Lock()


def _budget_remaining() -> float:
    return max(0.0, PUBLIC_DAILY_BUDGET_SECONDS - budget_used_seconds)


def _event(stage: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {
        "type": "pipeline_event",
        "stage": stage,
        "detail": detail,
        "time": time.time(),
        **extra,
    }


def _compose_prompt(text: str, transcript: str, scene_context: str) -> str:
    pieces = [text.strip()]
    if transcript:
        pieces.append(f"audio intent: {transcript}")
    if scene_context and scene_context != "no visual context":
        pieces.append(f"visual context: {scene_context}")
    return " | ".join(piece for piece in pieces if piece)


def _generate_joints(state: SessionState, t: float) -> list[list[float]]:
    prompt = state.prompt.lower()
    audio_boost = 0.65 + state.audio_level * 0.55
    video_boost = 0.9 + state.video_energy * 0.35
    speed = 1.2 * audio_boost * video_boost

    if "dance" in prompt:
        speed *= 1.55
    elif "slow" in prompt or "relaxed" in prompt:
        speed *= 0.72

    phase = t * speed * math.tau
    step = math.sin(phase)
    counter_step = math.sin(phase + math.pi)
    arm = math.sin(phase + math.pi * 0.7)
    counter_arm = math.sin(phase + math.pi * 1.7)

    if "circle" in prompt:
        radius = 1.25
        root_x = math.cos(t * 0.45) * radius
        root_z = math.sin(t * 0.45) * radius
    else:
        root_x = math.sin(t * 0.35) * 0.55
        root_z = (t * 0.34) % 5.0 - 2.5

    jump = 0.0
    if "jump" in prompt:
        jump = max(0.0, math.sin(phase * 0.5)) * 0.45

    sway = math.sin(phase * 0.5) * 0.04
    root_y = 0.92 + jump
    joints = [[root_x, root_y, root_z] for _ in range(22)]

    # Legs: chains [0, 1, 4, 7, 10] and [0, 2, 5, 8, 11].
    joints[1] = [root_x - 0.13, root_y - 0.08, root_z]
    joints[4] = [root_x - 0.18, root_y - 0.48, root_z + 0.20 * step]
    joints[7] = [root_x - 0.15, root_y - 0.86, root_z - 0.24 * step]
    joints[10] = [root_x - 0.14, root_y - 0.91, root_z - 0.36 * step + 0.12]

    joints[2] = [root_x + 0.13, root_y - 0.08, root_z]
    joints[5] = [root_x + 0.18, root_y - 0.48, root_z + 0.20 * counter_step]
    joints[8] = [root_x + 0.15, root_y - 0.86, root_z - 0.24 * counter_step]
    joints[11] = [root_x + 0.14, root_y - 0.91, root_z - 0.36 * counter_step + 0.12]

    # Spine/head: chain [0, 3, 6, 9, 12, 15].
    joints[3] = [root_x + sway, root_y + 0.24, root_z]
    joints[6] = [root_x + sway * 1.4, root_y + 0.52, root_z]
    joints[9] = [root_x + sway * 1.8, root_y + 0.73, root_z]
    joints[12] = [root_x + sway * 2.0, root_y + 0.94, root_z]
    joints[15] = [root_x + sway * 2.0, root_y + 1.06, root_z]

    wave = 0.0
    if "wave" in prompt or state.audio_level > 0.7:
        wave = max(0.0, math.sin(phase * 1.6)) * 0.38

    # Arms: chains [9, 14, 17, 19, 21] and [9, 13, 16, 18, 20].
    shoulder_y = root_y + 0.68
    joints[14] = [root_x - 0.24, shoulder_y, root_z]
    joints[17] = [root_x - 0.48, shoulder_y - 0.20 + 0.08 * arm, root_z + 0.18 * arm]
    joints[19] = [root_x - 0.66, shoulder_y - 0.42 + 0.16 * arm, root_z + 0.24 * arm]
    joints[21] = [root_x - 0.72, shoulder_y - 0.49 + 0.18 * arm, root_z + 0.28 * arm]

    joints[13] = [root_x + 0.24, shoulder_y, root_z]
    joints[16] = [root_x + 0.48, shoulder_y - 0.18 + wave, root_z + 0.17 * counter_arm]
    joints[18] = [root_x + 0.66, shoulder_y - 0.38 + wave * 1.25, root_z + 0.26 * counter_arm]
    joints[20] = [root_x + 0.72, shoulder_y - 0.44 + wave * 1.45, root_z + 0.31 * counter_arm]

    return [[round(v, 4) for v in joint] for joint in joints]


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (BASE_DIR / "templates" / "index.html").read_text()


@app.get("/api/config")
async def get_config() -> dict[str, Any]:
    return {
        "mode": "hf-endpoint-sim",
        "frame_rate": FRAME_RATE,
        "public_budget_seconds": PUBLIC_DAILY_BUDGET_SECONDS,
        "budget_used_seconds": round(budget_used_seconds, 2),
        "budget_remaining_seconds": round(_budget_remaining(), 2),
        "endpoints": {
            "asr": "simulated-hf-endpoint",
            "vlm": "simulated-hf-endpoint",
            "tts": "simulated-hf-endpoint",
            "motion": "simulated-flooddiffusion-endpoint",
        },
    }


@app.post("/api/session/start")
async def start_session(payload: StartSessionRequest) -> dict[str, Any]:
    if payload.endpoint_mode != "hf-endpoint-sim":
        raise HTTPException(status_code=400, detail="Only the HF Endpoint simulation path is wired in this prototype.")

    if _budget_remaining() <= 0:
        raise HTTPException(
            status_code=429,
            detail="Public demo budget exhausted for this Space.",
        )

    asr_result, vlm_result = await asyncio.gather(
        clients.asr(payload.audio_enabled),
        clients.vlm(payload.video_enabled),
    )
    prompt = _compose_prompt(payload.text, asr_result["text"], vlm_result["context"])
    tts_result = await clients.tts("Simulation ready.")
    session_id = uuid.uuid4().hex[:16]
    state = SessionState(
        session_id=session_id,
        prompt=prompt,
        transcript=asr_result["text"],
        scene_context=vlm_result["context"],
        audio_enabled=payload.audio_enabled,
        video_enabled=payload.video_enabled,
        endpoint_mode="hf-endpoint-sim",
        status_events=[
            _event("ASR", "endpoint transcript ready", result=asr_result),
            _event("VLM", "endpoint scene context ready", result=vlm_result),
            _event("TTS", "endpoint response metadata ready", result=tts_result),
            _event("Motion", "motion endpoint stream waiting for WebSocket"),
        ],
    )
    sessions[session_id] = state
    return {
        "session_id": session_id,
        "prompt": prompt,
        "transcript": state.transcript,
        "scene_context": state.scene_context,
        "events": state.status_events,
        "budget_remaining_seconds": round(_budget_remaining(), 2),
    }


@app.post("/api/session/{session_id}/text")
async def update_text(session_id: str, payload: TextUpdateRequest) -> dict[str, Any]:
    state = sessions.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Unknown session")
    state.prompt = _compose_prompt(payload.text, state.transcript, state.scene_context)
    event = _event("Motion", "prompt updated without resetting the stream", prompt=state.prompt)
    state.status_events.append(event)
    return {"prompt": state.prompt, "event": event}


@app.post("/api/session/{session_id}/reset")
async def reset_session(session_id: str) -> dict[str, Any]:
    state = sessions.get(session_id)
    if state:
        state.running = False
        sessions.pop(session_id, None)
    return {"status": "reset", "session_id": session_id}


@app.websocket("/ws/motion/{session_id}")
async def motion_stream(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    state = sessions.get(session_id)
    if not state:
        await websocket.send_json({"type": "error", "detail": "Unknown session"})
        await websocket.close()
        return

    for event in state.status_events:
        await websocket.send_json(event)

    started = time.time()
    global budget_used_seconds
    try:
        while state.running:
            if _budget_remaining() <= 0:
                await websocket.send_json(
                    {
                        "type": "budget_exhausted",
                        "detail": "Public demo budget exhausted for this Space.",
                    }
                )
                break

            now = time.time()
            elapsed = now - started
            joints = _generate_joints(state, elapsed)
            state.frame_id += 1
            await websocket.send_json(
                {
                    "type": "motion_frame",
                    "session_id": session_id,
                    "frame_id": state.frame_id,
                    "timestamp": now,
                    "joints": joints,
                    "audio_level": round(state.audio_level, 3),
                    "video_energy": round(state.video_energy, 3),
                    "prompt": state.prompt,
                    "budget_remaining_seconds": round(_budget_remaining(), 2),
                }
            )
            async with budget_lock:
                budget_used_seconds += FRAME_INTERVAL
            await asyncio.sleep(FRAME_INTERVAL)
    except WebSocketDisconnect:
        return
    finally:
        state.running = False


@app.websocket("/ws/audio/{session_id}")
async def audio_stream(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    state = sessions.get(session_id)
    if not state:
        await websocket.send_json({"type": "error", "detail": "Unknown session"})
        await websocket.close()
        return

    try:
        while state.running:
            data = await websocket.receive_json()
            level = float(data.get("level", 0.0))
            state.audio_level = max(0.0, min(1.0, level))
            if data.get("sequence", 0) % 8 == 0:
                mood = "energetic" if state.audio_level > 0.65 else "steady"
                state.transcript = f"live ASR hears a {mood} instruction stream"
                await websocket.send_json(
                    {
                        "type": "asr_partial",
                        "text": state.transcript,
                        "level": round(state.audio_level, 3),
                        "source": "hf-endpoint-sim-asr",
                    }
                )
    except WebSocketDisconnect:
        return


@app.websocket("/ws/video/{session_id}")
async def video_stream(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    state = sessions.get(session_id)
    if not state:
        await websocket.send_json({"type": "error", "detail": "Unknown session"})
        await websocket.close()
        return

    try:
        while state.running:
            data = await websocket.receive_json()
            energy = float(data.get("motion_energy", 0.0))
            state.video_energy = max(0.0, min(1.0, energy))
            if data.get("sequence", 0) % 4 == 0:
                scene = "active full-body framing" if state.video_energy > 0.55 else "stable indoor full-body framing"
                state.scene_context = f"{scene}, simulated keyframe analysis"
                await websocket.send_json(
                    {
                        "type": "vlm_context",
                        "context": state.scene_context,
                        "motion_energy": round(state.video_energy, 3),
                        "source": "hf-endpoint-sim-vlm",
                    }
                )
    except WebSocketDisconnect:
        return
