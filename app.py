"""FastAPI app for FloodDiffusion-style realtime motion streaming."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import CreateRealtimeSessionRequest, InputTextRequest
from integrations.kimodo_g1_client import (
    GeneratedG1Frame,
    GeneratedG1Segment,
    KimodoG1Client,
    KimodoG1Error,
)
from renderers import DEFAULT_RENDERER, RenderInput, build_renderer_registry
from sessions import MotionSession, MotionSessionManager


BASE_DIR = Path(__file__).resolve().parent
FRAME_RATE = 20
PUBLIC_DAILY_BUDGET_SECONDS = 180
G1_ASSET_ROOT = BASE_DIR / ".g1_cache" / "g1skel34"

app = FastAPI(title="FloodScene Realtime Motion API")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/g1_assets", StaticFiles(directory=G1_ASSET_ROOT, check_dir=False), name="g1_assets")

renderers = build_renderer_registry(BASE_DIR)
session_manager = MotionSessionManager(renderers, DEFAULT_RENDERER)
g1_runtime = renderers["g1"]
smplx_runtime = renderers["smplx"]
kimodo_g1_clients = KimodoG1Client.from_env_pool()
kimodo_g1_client = kimodo_g1_clients[0] if kimodo_g1_clients else None

budget_used_seconds = 0.0
budget_lock = asyncio.Lock()


def _budget_remaining() -> float:
    return max(0.0, PUBLIC_DAILY_BUDGET_SECONDS - budget_used_seconds)


def _kimodo_client_for_index(worker_index: int | None) -> KimodoG1Client | None:
    if not kimodo_g1_clients:
        return None
    if worker_index is None:
        return kimodo_g1_clients[0]
    return kimodo_g1_clients[worker_index % len(kimodo_g1_clients)]


def _kimodo_client_for_session(session: MotionSession) -> KimodoG1Client | None:
    return _kimodo_client_for_index(session.kimodo_worker_index)


def _motion_format(renderer_name: str) -> dict[str, Any]:
    if renderer_name == "smplx":
        return {
            "type": "motion.format",
            "renderer": "smplx",
            "format": "smplx.v1",
            "header_float32": 9,
            "payload": ["vertices_10475x3", "joints_22x3"],
        }
    return {
        "type": "motion.format",
        "renderer": "g1",
        "format": "g1.v1",
        "header_float32": 9,
        "payload": ["joints_34x3", "global_rotations_34x9"],
    }


def _session_payload(session: MotionSession) -> dict[str, Any]:
    return {
        "session_id": session.session_id,
        "renderer": session.renderer_name,
        "input_mode": session.input_mode,
        "status": "created" if session.running else "closed",
        "initial_text": session.current_text,
        "schedule": [
            {"text": cue.text, "start": cue.start, "end": cue.end}
            for cue in session.schedule
        ],
        "frame_rate": session.frame_rate,
        "seed": session.seed,
        "kimodo_worker_index": session.kimodo_worker_index,
        "websocket_url": f"/api/realtime/sessions/{session.session_id}",
    }


def _create_session(payload: CreateRealtimeSessionRequest) -> MotionSession:
    if _budget_remaining() <= 0:
        raise HTTPException(status_code=429, detail="Public demo budget exhausted for this Space.")

    selected_client = _kimodo_client_for_index(payload.kimodo_worker_index)
    try:
        session = session_manager.create(
            renderer_name=payload.renderer,
            input_mode=payload.input_mode,
            initial_text=payload.initial_text,
            schedule=payload.schedule,
            frame_rate=payload.frame_rate,
            seed=payload.seed if payload.seed is not None else (selected_client.seed if selected_client else None),
            kimodo_worker_index=payload.kimodo_worker_index,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    renderer = renderers[session.renderer_name]
    if not renderer.available:
        session_manager.close(session.session_id)
        raise HTTPException(
            status_code=503,
            detail=f"Renderer {session.renderer_name} unavailable: {renderer.load_error}",
        )
    return session


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (BASE_DIR / "templates" / "index.html").read_text()


@app.get("/multi", response_class=HTMLResponse)
async def multi_grid() -> str:
    return (BASE_DIR / "templates" / "multi.html").read_text()


@app.get("/api/config")
async def get_config() -> dict[str, Any]:
    return {
        "mode": "realtime-motion",
        "renderer": DEFAULT_RENDERER,
        "visualization": "unitree-g1-stl",
        "g1_available": g1_runtime.available,
        "g1_error": g1_runtime.load_error,
        "g1_asset_root": str(G1_ASSET_ROOT),
        "kimodo_g1_offline_enabled": bool(kimodo_g1_clients),
        "kimodo_g1_worker_count": len(kimodo_g1_clients),
        "kimodo_g1_api_urls": [client.base_url for client in kimodo_g1_clients],
        "smplx_available": smplx_runtime.available,
        "smplx_error": smplx_runtime.load_error,
        "smplx_gender": "neutral",
        "smplx_beta_mode": "all_zero",
        "frame_rate": FRAME_RATE,
        "kimodo_g1_default_seed": kimodo_g1_client.seed if kimodo_g1_client else None,
        "public_budget_seconds": PUBLIC_DAILY_BUDGET_SECONDS,
        "budget_used_seconds": round(budget_used_seconds, 2),
        "budget_remaining_seconds": round(_budget_remaining(), 2),
        "realtime": {
            "session_endpoint": "/api/realtime/sessions",
            "websocket_endpoint": "/api/realtime/sessions/{session_id}",
            "input_modes": ["online", "offline"],
        },
    }


@app.get("/api/g1/topology")
async def get_g1_topology() -> dict[str, Any]:
    try:
        return g1_runtime.topology()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Unitree G1 assets unavailable: {exc}") from exc


@app.get("/api/smplx/topology")
async def get_smplx_topology() -> dict[str, Any]:
    try:
        return smplx_runtime.topology()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"SMPL-X model unavailable: {exc}") from exc


@app.post("/api/realtime/sessions")
async def create_realtime_session(payload: CreateRealtimeSessionRequest) -> dict[str, Any]:
    return _session_payload(_create_session(payload))


@app.post("/api/realtime/sessions/{session_id}/input_text")
async def input_text(session_id: str, payload: InputTextRequest) -> dict[str, Any]:
    session = session_manager.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Unknown realtime session.")
    try:
        session.set_online_text(payload.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "type": "input_text.committed",
        "session_id": session_id,
        "text": session.current_text,
    }


@app.post("/api/realtime/sessions/{session_id}/close")
async def close_realtime_session(session_id: str) -> dict[str, Any]:
    session = session_manager.close(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Unknown realtime session.")
    return {"type": "session.closed", "session_id": session_id}


async def _receive_realtime_events(
    websocket: WebSocket,
    session: MotionSession,
    stream_started_at: float,
    send_json: Callable[[dict[str, Any]], Awaitable[None]],
) -> None:
    while session.running:
        try:
            event = await websocket.receive_json()
        except WebSocketDisconnect:
            session.running = False
            break

        event_type = event.get("type")
        elapsed = max(0.0, time.time() - stream_started_at)

        if event_type == "input_text.append":
            try:
                session.set_online_text(str(event.get("text", "")))
            except ValueError as exc:
                await send_json({"type": "error", "code": "invalid_input_text", "message": str(exc)})
                continue
            await send_json(
                {
                    "type": "input_text.committed",
                    "session_id": session.session_id,
                    "text": session.current_text,
                    "elapsed": round(elapsed, 3),
                }
            )
        elif event_type == "session.pause":
            session.paused = True
            await send_json({"type": "session.paused", "session_id": session.session_id})
        elif event_type == "session.resume":
            session.paused = False
            await send_json({"type": "session.resumed", "session_id": session.session_id})
        elif event_type == "session.close":
            session.running = False
            await send_json({"type": "session.closed", "session_id": session.session_id})
        elif event_type == "session.start":
            session.paused = False
            await send_json({"type": "session.started", "session_id": session.session_id})
        else:
            await send_json(
                {
                    "type": "error",
                    "code": "unknown_event",
                    "message": f"Unknown realtime event: {event_type}",
                }
            )


def _uses_kimodo_g1_offline(session: MotionSession) -> bool:
    return (
        _kimodo_client_for_session(session) is not None
        and session.renderer_name == "g1"
        and session.input_mode == "offline"
    )


async def _stream_kimodo_g1_offline(
    session: MotionSession,
    renderer: Any,
    send_json: Callable[[dict[str, Any]], Awaitable[None]],
    send_bytes: Callable[[bytes], Awaitable[None]],
) -> None:
    kimodo_client = _kimodo_client_for_session(session)
    assert kimodo_client is not None
    if len(session.schedule) > 1:
        await _stream_kimodo_g1_offline_sequence(session, renderer, send_json, send_bytes)
        return

    started_at = time.time()
    await send_json(
        {
            "type": "motion_generation.started",
            "session_id": session.session_id,
            "renderer": "g1",
            "provider": "kimodo",
            "mode": "offline",
        }
    )

    try:
        motion = await asyncio.to_thread(kimodo_client.generate_offline, session.schedule, session.seed)
    except KimodoG1Error as exc:
        session.running = False
        await send_json(
            {
                "type": "error",
                "code": "kimodo_g1_generation_failed",
                "message": str(exc),
            }
        )
        return
    except Exception as exc:  # noqa: BLE001 - surfaced to websocket client.
        session.running = False
        await send_json(
            {
                "type": "error",
                "code": "kimodo_g1_generation_failed",
                "message": str(exc),
            }
        )
        return

    global budget_used_seconds
    stream_completed = True
    streamed_frames = 0
    streamed_seconds = 0.0
    frame_interval = 1.0 / motion.fps

    await send_json(
        {
            "type": "motion_generation.segment_completed",
            "session_id": session.session_id,
            "renderer": "g1",
            "provider": "kimodo",
            "segment_index": 0,
            "text": session.current_text,
            "frames": motion.num_frames,
            "target_frames": motion.num_frames,
            "generated_frames": motion.num_frames,
            "output_frames": motion.num_frames,
            "next_start_frames": 0,
            "fps": round(motion.fps, 3),
            "generation_seconds": round(motion.generation_seconds, 3),
            "received_seconds": round(motion.wall_seconds, 3),
        }
    )

    for frame_index in range(motion.num_frames):
        if not session.running:
            stream_completed = False
            break
        while session.paused and session.running:
            await asyncio.sleep(0.05)
        if not session.running:
            stream_completed = False
            break
        if _budget_remaining() <= 0:
            await send_json({"type": "budget_exhausted", "detail": "Public demo budget exhausted for this Space."})
            session.running = False
            stream_completed = False
            break

        prompt, cue_index, cue_changed = session.text_for_elapsed(streamed_seconds)
        if cue_changed:
            await send_json(
                {
                    "type": "offline_cue.changed",
                    "session_id": session.session_id,
                    "cue_index": cue_index,
                    "text": prompt,
                    "elapsed": round(streamed_seconds, 3),
                }
            )

        session.frame_id += 1
        await send_bytes(
            renderer.binary_frame_from_arrays(
                joints=motion.posed_joints[frame_index],
                global_rots=motion.global_rot_mats[frame_index],
                root_position=motion.root_positions[frame_index],
                frame_id=session.frame_id,
                audio_level=session.audio_level,
                video_energy=session.video_energy,
                budget_remaining=round(_budget_remaining(), 2),
                buffer_size=min(4, session.frame_id),
                buffer_capacity=4,
            )
        )
        streamed_frames += 1
        streamed_seconds += frame_interval
        async with budget_lock:
            budget_used_seconds += frame_interval
        if frame_index < motion.num_frames - 1:
            await asyncio.sleep(frame_interval)

    if stream_completed:
        # Give the final binary frame one event-loop tick before terminal JSON events.
        # Without this, some websocket clients observe the close before the tail messages.
        await asyncio.sleep(0.05)
        await send_json(
            {
                "type": "motion_generation.completed",
                "session_id": session.session_id,
                "renderer": "g1",
                "provider": "kimodo",
                "mode": "offline",
                "segments": len(session.schedule),
                "frames": streamed_frames,
                "duration": round(streamed_seconds, 3),
                "generation_seconds": round(motion.generation_seconds, 3),
                "wall_seconds": round(time.time() - started_at, 3),
            }
        )
        await send_json(
            {
                "type": "offline_schedule.completed",
                "session_id": session.session_id,
                "elapsed": round(streamed_seconds, 3),
            }
        )
        await asyncio.sleep(0.1)


async def _stream_kimodo_g1_offline_sequence(
    session: MotionSession,
    renderer: Any,
    send_json: Callable[[dict[str, Any]], Awaitable[None]],
    send_bytes: Callable[[bytes], Awaitable[None]],
) -> None:
    kimodo_client = _kimodo_client_for_session(session)
    assert kimodo_client is not None
    started_at = time.time()
    await send_json(
        {
            "type": "motion_generation.started",
            "session_id": session.session_id,
            "renderer": "g1",
            "provider": "kimodo",
            "mode": "sequence",
        }
    )

    queue: asyncio.Queue[Any] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def produce_frames() -> None:
        try:
            for item in kimodo_client.generate_offline_sequence_frames(session.schedule, seed=session.seed):
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception as exc:  # noqa: BLE001 - surfaced to websocket client.
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    producer_task = asyncio.create_task(asyncio.to_thread(produce_frames))

    global budget_used_seconds
    stream_completed = True
    streamed_frames = 0
    streamed_seconds = 0.0
    generation_seconds = 0.0
    segment_count = 0

    async def stream_segment(segment: GeneratedG1Segment) -> None:
        global budget_used_seconds
        nonlocal streamed_frames, streamed_seconds, generation_seconds, segment_count
        segment_count += 1
        generation_seconds += segment.generation_seconds
        await send_json(
            {
                "type": "motion_generation.segment_completed",
                "session_id": session.session_id,
                "renderer": "g1",
                "provider": "kimodo",
                "segment_index": segment.segment_index,
                "text": segment.text,
                "frames": segment.num_frames,
                "target_frames": segment.target_frames,
                "generated_frames": segment.generated_frames,
                "output_frames": segment.output_frames,
                "next_start_frames": segment.next_start_frames,
                "fps": round(segment.fps, 3),
                "generation_seconds": round(segment.generation_seconds, 3),
                "received_seconds": round(segment.received_seconds, 3),
            }
        )

        if segment.segment_index != session.last_cue_index:
            session.current_text = segment.text
            session.last_cue_index = segment.segment_index
            await send_json(
                {
                    "type": "offline_cue.changed",
                    "session_id": session.session_id,
                    "cue_index": segment.segment_index,
                    "text": segment.text,
                    "elapsed": round(streamed_seconds, 3),
                }
            )

    async def stream_frame(frame: GeneratedG1Frame) -> bool:
        global budget_used_seconds
        nonlocal streamed_frames, streamed_seconds
        if not session.running:
            return False
        while session.paused and session.running:
            await asyncio.sleep(0.05)
        if not session.running:
            return False
        if _budget_remaining() <= 0:
            await send_json({"type": "budget_exhausted", "detail": "Public demo budget exhausted for this Space."})
            session.running = False
            return False

        session.frame_id += 1
        await send_bytes(
            renderer.binary_frame_from_arrays(
                joints=frame.posed_joints,
                global_rots=frame.global_rot_mats,
                root_position=frame.root_position,
                frame_id=session.frame_id,
                audio_level=session.audio_level,
                video_energy=session.video_energy,
                budget_remaining=round(_budget_remaining(), 2),
                buffer_size=min(4, max(1, queue.qsize())),
                buffer_capacity=4,
            )
        )
        frame_interval = 1.0 / frame.fps
        streamed_frames += 1
        streamed_seconds += frame_interval
        async with budget_lock:
            budget_used_seconds += frame_interval
        await asyncio.sleep(frame_interval)
        return True

    while session.running:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            if producer_task.done() and queue.empty():
                break
            continue

        if item is None:
            break
        if isinstance(item, KimodoG1Error):
            session.running = False
            stream_completed = False
            await send_json(
                {
                    "type": "error",
                    "code": "kimodo_g1_generation_failed",
                    "message": str(item),
                }
            )
            break
        if isinstance(item, Exception):
            session.running = False
            stream_completed = False
            await send_json(
                {
                    "type": "error",
                    "code": "kimodo_g1_generation_failed",
                    "message": str(item),
                }
            )
            break
        if isinstance(item, GeneratedG1Segment):
            await stream_segment(item)
            continue
        if not isinstance(item, GeneratedG1Frame):
            continue
        if not await stream_frame(item):
            stream_completed = False
            break

    if not producer_task.done() and not session.running:
        producer_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await producer_task

    if stream_completed:
        await asyncio.sleep(0.05)
        await send_json(
            {
                "type": "motion_generation.completed",
                "session_id": session.session_id,
                "renderer": "g1",
                "provider": "kimodo",
                "mode": "sequence",
                "segments": segment_count,
                "frames": streamed_frames,
                "duration": round(streamed_seconds, 3),
                "generation_seconds": round(generation_seconds, 3),
                "wall_seconds": round(time.time() - started_at, 3),
            }
        )
        await send_json(
            {
                "type": "offline_schedule.completed",
                "session_id": session.session_id,
                "elapsed": round(streamed_seconds, 3),
            }
        )
        await asyncio.sleep(0.1)


@app.websocket("/api/realtime/sessions/{session_id}")
async def realtime_session_stream(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    session = session_manager.get(session_id)
    if not session:
        await websocket.send_json({"type": "error", "code": "unknown_session", "message": "Unknown realtime session."})
        await websocket.close()
        return

    renderer = renderers[session.renderer_name]
    send_lock = asyncio.Lock()

    async def send_json(payload: dict[str, Any]) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def send_bytes(payload: bytes) -> None:
        async with send_lock:
            await websocket.send_bytes(payload)

    stream_started_at = time.time()
    frame_interval = 1.0 / session.frame_rate
    control_task: asyncio.Task | None = None

    global budget_used_seconds
    try:
        await send_json({"type": "session.created", **_session_payload(session)})
        await send_json(_motion_format(session.renderer_name))
        await send_json({"type": "session.started", "session_id": session.session_id})

        if _uses_kimodo_g1_offline(session):
            await _stream_kimodo_g1_offline(session, renderer, send_json, send_bytes)
            return

        control_task = asyncio.create_task(
            _receive_realtime_events(websocket, session, stream_started_at, send_json)
        )

        while session.running:
            if session.paused:
                await asyncio.sleep(0.05)
                continue

            if _budget_remaining() <= 0:
                await send_json({"type": "budget_exhausted", "detail": "Public demo budget exhausted for this Space."})
                break

            elapsed = max(0.0, time.time() - stream_started_at)
            if session.offline_finished(elapsed):
                await send_json(
                    {
                        "type": "offline_schedule.completed",
                        "session_id": session.session_id,
                        "elapsed": round(elapsed, 3),
                    }
                )
                break

            prompt, cue_index, cue_changed = session.text_for_elapsed(elapsed)
            if cue_changed:
                await send_json(
                    {
                        "type": "offline_cue.changed",
                        "session_id": session.session_id,
                        "cue_index": cue_index,
                        "text": prompt,
                        "elapsed": round(elapsed, 3),
                    }
                )

            session.frame_id += 1
            await send_bytes(
                renderer.binary_frame(
                    render_input=RenderInput(
                        prompt=prompt,
                        audio_level=session.audio_level,
                        video_energy=session.video_energy,
                    ),
                    t=elapsed,
                    frame_id=session.frame_id,
                    budget_remaining=round(_budget_remaining(), 2),
                    buffer_size=min(4, session.frame_id),
                    buffer_capacity=4,
                )
            )
            async with budget_lock:
                budget_used_seconds += frame_interval
            await asyncio.sleep(frame_interval)
    except WebSocketDisconnect:
        return
    finally:
        session_manager.close(session_id)
        if control_task is not None:
            control_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
                await control_task
