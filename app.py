"""Local demo server: UI, topology, and remote Kimodo stream forwarding."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from renderers import DEFAULT_RENDERER, build_renderer_registry


BASE_DIR = Path(__file__).resolve().parent
FRAME_RATE = 20
PUBLIC_DAILY_BUDGET_SECONDS = 180
G1_ASSET_ROOT = BASE_DIR / ".g1_cache" / "g1skel34"

app = FastAPI(title="Motion Generation Demo")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/g1_assets", StaticFiles(directory=G1_ASSET_ROOT, check_dir=False), name="g1_assets")

renderers = build_renderer_registry(BASE_DIR)
g1_runtime = renderers["g1"]
smplx_runtime = renderers["smplx"]

budget_used_seconds = 0.0
budget_lock = asyncio.Lock()


class KimodoAPIError(RuntimeError):
    pass


@dataclass(frozen=True)
class TextCue:
    text: str
    start: float
    end: float


class KimodoG1Client:
    def __init__(
        self,
        base_url: str,
        *,
        diffusion_steps: int = 20,
        seed: int | None = 11,
        timeout_seconds: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.diffusion_steps = diffusion_steps
        self.seed = seed
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls) -> "KimodoG1Client | None":
        base_url = os.getenv("KIMODO_G1_API_URL", "").strip()
        if not base_url:
            return None
        seed_raw = os.getenv("KIMODO_G1_SEED", "11").strip()
        seed = None if seed_raw.lower() in {"", "none", "null"} else int(seed_raw)
        return cls(
            base_url,
            diffusion_steps=int(os.getenv("KIMODO_G1_DIFFUSION_STEPS", "20")),
            seed=seed,
            timeout_seconds=float(os.getenv("KIMODO_G1_REQUEST_TIMEOUT", "600")),
        )

    def stream_offline_frames(
        self,
        schedule: Iterable[TextCue],
        *,
        seed: int | None,
        diffusion_steps: int | None,
    ) -> Iterable[dict[str, Any]]:
        payload: dict[str, Any] = {
            "segments": [
                {
                    "text": cue.text,
                    "duration": cue.end - cue.start,
                }
                for cue in schedule
            ],
            "diffusion_steps": diffusion_steps or self.diffusion_steps,
        }
        resolved_seed = self.seed if seed is None else seed
        if resolved_seed is not None:
            payload["seed"] = resolved_seed

        request = urllib.request.Request(
            f"{self.base_url}/v1/g1/generate_sequence_frames",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise KimodoAPIError("Remote Kimodo API returned invalid NDJSON.") from exc
                    if not isinstance(event, dict):
                        raise KimodoAPIError("Remote Kimodo API returned a non-object stream event.")
                    if event.get("type") == "error":
                        raise KimodoAPIError(str(event.get("detail", "Remote Kimodo generation failed.")))
                    yield event
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise KimodoAPIError(f"Remote Kimodo API returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise KimodoAPIError(f"Remote Kimodo API is unreachable: {exc.reason}") from exc
        except TimeoutError as exc:
            raise KimodoAPIError("Remote Kimodo API request timed out.") from exc


kimodo_g1_client = KimodoG1Client.from_env()


def _budget_remaining() -> float:
    return max(0.0, PUBLIC_DAILY_BUDGET_SECONDS - budget_used_seconds)


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


def _normalize_schedule(raw_schedule: Any) -> list[TextCue]:
    if not isinstance(raw_schedule, list) or not raw_schedule:
        raise ValueError("Offline request requires a non-empty schedule.")

    raw_cues = sorted(raw_schedule, key=lambda cue: float(cue.get("start", 0)))
    if float(raw_cues[0].get("start", 0)) != 0:
        raise ValueError("The first schedule cue must start at 0.")

    normalized: list[TextCue] = []
    for index, raw in enumerate(raw_cues):
        text = str(raw.get("text", "")).strip()
        start = float(raw.get("start", 0))
        end_value = raw.get("end")
        end = float(end_value) if end_value is not None and end_value != "" else None
        if index < len(raw_cues) - 1:
            end = float(raw_cues[index + 1].get("start", 0))
        if not text:
            raise ValueError("Schedule cue text cannot be empty.")
        if start < 0:
            raise ValueError("Schedule cue start must be non-negative.")
        if end is None or end <= start:
            raise ValueError("Each schedule cue must have a positive end time.")
        if normalized and start <= normalized[-1].start:
            raise ValueError("Schedule cue starts must be strictly increasing.")
        normalized.append(TextCue(text=text, start=start, end=end))
    return normalized


def _request_options(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("config")
    if not isinstance(config, dict):
        config = {}
    return {
        "renderer": str(config.get("renderer", payload.get("renderer", DEFAULT_RENDERER))).lower(),
        "seed": config.get("seed", payload.get("seed")),
    }


def _decode_g1_frame(event: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_data = event.get("data")
    if not isinstance(raw_data, str):
        raise KimodoAPIError("Remote frame event is missing data.")
    values = np.frombuffer(base64.b64decode(raw_data), dtype=np.float32).copy()
    expected = 34 * 3 + 34 * 9 + 3
    if values.shape != (expected,):
        raise KimodoAPIError(f"Unexpected remote frame payload shape: {values.shape}.")
    joint_end = 34 * 3
    rot_end = joint_end + 34 * 9
    joints = values[:joint_end].reshape(34, 3)
    rotations = values[joint_end:rot_end].reshape(34, 3, 3)
    root_position = values[rot_end:].reshape(3)
    if not np.isfinite(joints).all() or not np.isfinite(rotations).all() or not np.isfinite(root_position).all():
        raise KimodoAPIError("Remote frame contains non-finite values.")
    return joints, rotations, root_position


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (BASE_DIR / "templates" / "index.html").read_text()


@app.get("/api/config")
async def get_config() -> dict[str, Any]:
    return {
        "mode": "offline-motion",
        "renderer": DEFAULT_RENDERER,
        "visualization": "unitree-g1-stl",
        "g1_available": g1_runtime.available,
        "g1_error": g1_runtime.load_error,
        "g1_asset_root": str(G1_ASSET_ROOT),
        "kimodo_g1_offline_enabled": kimodo_g1_client is not None,
        "kimodo_g1_api_url": kimodo_g1_client.base_url if kimodo_g1_client else None,
        "smplx_available": smplx_runtime.available,
        "smplx_error": smplx_runtime.load_error,
        "smplx_gender": "neutral",
        "smplx_beta_mode": "all_zero",
        "frame_rate": FRAME_RATE,
        "kimodo_g1_default_seed": kimodo_g1_client.seed if kimodo_g1_client else None,
        "public_budget_seconds": PUBLIC_DAILY_BUDGET_SECONDS,
        "budget_used_seconds": round(budget_used_seconds, 2),
        "budget_remaining_seconds": round(_budget_remaining(), 2),
        "offline": {"websocket_endpoint": "/api/offline"},
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


@app.websocket("/api/offline")
async def offline_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = uuid.uuid4().hex[:16]
    send_lock = asyncio.Lock()

    async def send_json(payload: dict[str, Any]) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def send_bytes(payload: bytes) -> None:
        async with send_lock:
            await websocket.send_bytes(payload)

    try:
        payload = await websocket.receive_json()
    except WebSocketDisconnect:
        return
    except Exception:
        await send_json({"type": "error", "code": "invalid_request", "message": "Expected JSON schedule/config."})
        await websocket.close()
        return

    try:
        if not isinstance(payload, dict):
            raise ValueError("Offline request must be a JSON object.")
        options = _request_options(payload)
        if options["renderer"] != "g1":
            raise ValueError("Offline generation currently supports renderer=g1.")
        schedule = _normalize_schedule(payload.get("schedule"))
        seed = None if options["seed"] is None else int(options["seed"])
        if kimodo_g1_client is None:
            raise ValueError("Remote Kimodo G1 API is not configured.")
    except Exception as exc:
        await send_json({"type": "error", "code": "invalid_request", "message": str(exc)})
        await websocket.close()
        return

    await send_json(
        {
            "type": "session.created",
            "session_id": session_id,
            "renderer": "g1",
            "input_mode": "offline",
            "seed": seed,
        }
    )
    await send_json(_motion_format("g1"))
    await send_json({"type": "session.started", "session_id": session_id})
    await send_json(
        {
            "type": "motion_generation.started",
            "session_id": session_id,
            "renderer": "g1",
            "provider": "kimodo",
            "mode": "offline",
        }
    )

    queue: asyncio.Queue[Any] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    started_at = time.time()

    def produce() -> None:
        try:
            for item in kimodo_g1_client.stream_offline_frames(
                schedule,
                seed=seed,
                diffusion_steps=None,
            ):
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    producer_task = asyncio.create_task(asyncio.to_thread(produce))

    global budget_used_seconds
    renderer = renderers["g1"]
    frame_id = 0
    streamed_frames = 0
    streamed_seconds = 0.0
    generation_seconds = 0.0
    segment_count = 0
    stream_completed = True

    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            if not isinstance(item, dict):
                continue

            event_type = item.get("type")
            if event_type == "segment.completed":
                segment_count += 1
                generation_seconds += float(item.get("generation_seconds", 0.0))
                await send_json(
                    {
                        "type": "motion_generation.segment_completed",
                        "session_id": session_id,
                        "renderer": "g1",
                        "provider": "kimodo",
                        "segment_index": int(item.get("index", -1)),
                        "text": str(item.get("text", "")),
                        "frames": int(item.get("output_frames", 0)),
                        "target_frames": int(item.get("target_frames", 0)),
                        "generated_frames": int(item.get("generated_frames", 0)),
                        "output_frames": int(item.get("output_frames", 0)),
                        "next_start_frames": int(item.get("next_start_frames", 0)),
                        "fps": round(float(item.get("fps", 30.0)), 3),
                        "generation_seconds": round(float(item.get("generation_seconds", 0.0)), 3),
                        "received_seconds": round(time.time() - started_at, 3),
                    }
                )
                await send_json(
                    {
                        "type": "offline_cue.changed",
                        "session_id": session_id,
                        "cue_index": int(item.get("index", -1)),
                        "text": str(item.get("text", "")),
                        "elapsed": round(streamed_seconds, 3),
                    }
                )
                continue

            if event_type == "sequence.completed":
                break
            if event_type != "frame":
                continue

            if _budget_remaining() <= 0:
                await send_json({"type": "budget_exhausted", "detail": "Public demo budget exhausted for this Space."})
                stream_completed = False
                break

            joints, rotations, root_position = _decode_g1_frame(item)
            frame_id += 1
            fps = float(item.get("fps", 30.0))
            frame_interval = 1.0 / fps
            await send_bytes(
                renderer.binary_frame_from_arrays(
                    joints=joints,
                    global_rots=rotations,
                    root_position=root_position,
                    frame_id=frame_id,
                    audio_level=0.35,
                    video_energy=0.25,
                    budget_remaining=round(_budget_remaining(), 2),
                    buffer_size=min(4, max(1, queue.qsize())),
                    buffer_capacity=4,
                )
            )
            streamed_frames += 1
            streamed_seconds += frame_interval
            async with budget_lock:
                budget_used_seconds += frame_interval
            await asyncio.sleep(frame_interval)

        if stream_completed:
            await asyncio.sleep(0.05)
            await send_json(
                {
                    "type": "motion_generation.completed",
                    "session_id": session_id,
                    "renderer": "g1",
                    "provider": "kimodo",
                    "mode": "offline",
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
                    "session_id": session_id,
                    "elapsed": round(streamed_seconds, 3),
                }
            )
    except WebSocketDisconnect:
        return
    except Exception as exc:
        with contextlib.suppress(Exception):
            await send_json({"type": "error", "code": "kimodo_g1_generation_failed", "message": str(exc)})
    finally:
        with contextlib.suppress(asyncio.CancelledError):
            await producer_task
