"""Local demo server: UI, topology, and remote Kimodo stream forwarding."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import wave
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from renderers import DEFAULT_RENDERER, build_renderer_registry


BASE_DIR = Path(__file__).resolve().parent
FRAME_RATE = 20
PUBLIC_DAILY_BUDGET_SECONDS = 180
G1_ASSET_ROOT = BASE_DIR / ".g1_cache" / "g1skel34"
MAX_AUDIO_CHUNK_SECONDS = 10.0
MAX_RECORDING_BYTES = 500 * 1024 * 1024
RECORDINGS_DIR = BASE_DIR / "recordings"
DEFAULT_FLOOD_DIFFUSION_OUTPUT_DIR = Path(
    "/mnt/data/cpfs/haiyang/FloodDiffusion-Dev/outputs/"
    "20260526_003412_chunk_dit_201d_concat_mix_l2_l3_random_mini16_beat_test128"
)
DEFAULT_FLOOD_DIFFUSION_CONFIG = Path(
    "/mnt/data/cpfs/haiyang/FloodDiffusion-Dev/configs/201d/"
    "chunk_dit_201d_concat_beat2_l234_units.yaml"
)

RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Motion Generation Demo")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/g1_assets", StaticFiles(directory=G1_ASSET_ROOT, check_dir=False), name="g1_assets")
app.mount("/recordings", StaticFiles(directory=RECORDINGS_DIR, check_dir=False), name="recordings")

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


@dataclass(frozen=True)
class AudioCue:
    audio_bytes: bytes
    mime_type: str
    name: str
    duration: float
    text: str = ""
    text_start: float | None = None
    text_end: float | None = None


@dataclass(frozen=True)
class AudioSegment:
    source_index: int
    chunk_index: int
    audio_start: float
    audio_end: float
    duration: float
    audio_bytes: bytes
    mime_type: str
    name: str
    text: str = ""
    text_start: float | None = None
    text_end: float | None = None


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


class FloodDiffusionAPIError(RuntimeError):
    pass


def _find_latest_checkpoint(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates: list[Path] = []
    for pattern in ("*.ckpt", "*.pt", "*.pth", "*.safetensors"):
        candidates.extend(path for path in root.rglob(pattern) if path.is_file())
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


class FloodDiffusionAudioClient:
    def __init__(
        self,
        base_url: str,
        *,
        endpoint: str = "/v1/smplx/generate_audio_sequence_frames",
        checkpoint_path: Path | None = None,
        config_path: Path | None = None,
        diffusion_steps: int | None = None,
        seed: int | None = 11,
        timeout_seconds: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.diffusion_steps = diffusion_steps
        self.seed = seed
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls) -> "FloodDiffusionAudioClient | None":
        base_url = os.getenv("FLOOD_DIFFUSION_API_URL", "").strip()
        if not base_url:
            return None

        checkpoint_raw = os.getenv("FLOOD_DIFFUSION_CHECKPOINT", "").strip()
        checkpoint_dir = Path(os.getenv("FLOOD_DIFFUSION_CHECKPOINT_DIR", str(DEFAULT_FLOOD_DIFFUSION_OUTPUT_DIR)))
        checkpoint_path = Path(checkpoint_raw) if checkpoint_raw else _find_latest_checkpoint(checkpoint_dir)

        config_raw = os.getenv("FLOOD_DIFFUSION_CONFIG", "").strip()
        config_path = Path(config_raw) if config_raw else DEFAULT_FLOOD_DIFFUSION_CONFIG
        if not config_path.exists():
            config_path = None

        seed_raw = os.getenv("FLOOD_DIFFUSION_SEED", "11").strip()
        seed = None if seed_raw.lower() in {"", "none", "null"} else int(seed_raw)
        steps_raw = os.getenv("FLOOD_DIFFUSION_DIFFUSION_STEPS", "").strip()
        diffusion_steps = int(steps_raw) if steps_raw else None

        return cls(
            base_url,
            endpoint=os.getenv("FLOOD_DIFFUSION_AUDIO_ENDPOINT", "/v1/smplx/generate_audio_sequence_frames"),
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            diffusion_steps=diffusion_steps,
            seed=seed,
            timeout_seconds=float(os.getenv("FLOOD_DIFFUSION_REQUEST_TIMEOUT", "600")),
        )

    def _url(self) -> str:
        if self.endpoint.startswith(("http://", "https://")):
            return self.endpoint
        return f"{self.base_url}/{self.endpoint.lstrip('/')}"

    def _build_payload(
        self,
        segments: Iterable[AudioSegment],
        *,
        seed: int | None,
        diffusion_steps: int | None,
    ) -> dict[str, Any]:
        payload_segments: list[dict[str, Any]] = []
        for segment in segments:
            item: dict[str, Any] = {
                "audio": {
                    "name": segment.name,
                    "mime_type": segment.mime_type,
                    "data": base64.b64encode(segment.audio_bytes).decode("ascii"),
                },
                "duration": segment.duration,
            }
            if segment.text:
                item["text"] = segment.text
                item["text_start"] = segment.text_start if segment.text_start is not None else 0.0
                item["text_end"] = segment.text_end if segment.text_end is not None else segment.duration
            payload_segments.append(item)

        payload: dict[str, Any] = {
            "model": "chunkdit_concat_201d_audio",
            "render_format": "smplx_params",
            "segments": payload_segments,
        }
        if self.checkpoint_path is not None:
            payload["checkpoint"] = str(self.checkpoint_path)
        if self.config_path is not None:
            payload["config"] = str(self.config_path)
        resolved_steps = diffusion_steps if diffusion_steps is not None else self.diffusion_steps
        if resolved_steps is not None:
            payload["diffusion_steps"] = resolved_steps
        resolved_seed = self.seed if seed is None else seed
        if resolved_seed is not None:
            payload["seed"] = resolved_seed
        return payload

    def stream_offline_audio_frames(
        self,
        segments: Iterable[AudioSegment],
        *,
        seed: int | None,
        diffusion_steps: int | None,
    ) -> Iterable[dict[str, Any]]:
        payload = self._build_payload(segments, seed=seed, diffusion_steps=diffusion_steps)
        request = urllib.request.Request(
            self._url(),
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
                        raise FloodDiffusionAPIError("Flood Diffusion API returned invalid NDJSON.") from exc
                    if not isinstance(event, dict):
                        raise FloodDiffusionAPIError("Flood Diffusion API returned a non-object stream event.")
                    if event.get("type") == "error":
                        raise FloodDiffusionAPIError(str(event.get("detail", "Flood Diffusion generation failed.")))
                    yield event
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise FloodDiffusionAPIError(f"Flood Diffusion API returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise FloodDiffusionAPIError(f"Flood Diffusion API is unreachable: {exc.reason}") from exc
        except TimeoutError as exc:
            raise FloodDiffusionAPIError("Flood Diffusion API request timed out.") from exc


flood_audio_client = FloodDiffusionAudioClient.from_env()


def _budget_remaining() -> float:
    return max(0.0, PUBLIC_DAILY_BUDGET_SECONDS - budget_used_seconds)


def _recording_suffix(mime_type: str, filename: str) -> str:
    normalized = mime_type.lower()
    suffix = Path(filename).suffix.lower()
    if "mp4" in normalized or suffix == ".mp4":
        return ".mp4"
    if "webm" in normalized or suffix == ".webm":
        return ".webm"
    return ".bin"


def _recording_url(path: Path) -> str:
    return f"/recordings/{path.name}"


def _transcode_recording_to_mp4(source_path: Path, target_path: Path) -> tuple[bool, str | None]:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return False, "ffmpeg_not_found"
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(source_path),
        "-movflags",
        "+faststart",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        str(target_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=300)
    except subprocess.TimeoutExpired:
        return False, "ffmpeg_timeout"
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip().splitlines()
        return False, detail[-1] if detail else "ffmpeg_failed"
    return target_path.exists() and target_path.stat().st_size > 0, None


def _motion_format(renderer_name: str) -> dict[str, Any]:
    if renderer_name == "smplx":
        return {
            "type": "motion.format",
            "renderer": "smplx",
            "format": "smplx_params.v1",
            "header_float32": 9,
            "payload": ["root_orient_3", "pose_body_21x3", "trans_3", "joints_22x3"],
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


def _strip_data_url(value: str) -> str:
    if value.startswith("data:") and "," in value:
        return value.split(",", 1)[1]
    return value


def _decode_audio_payload(raw_audio: Any) -> tuple[bytes, str, str]:
    if isinstance(raw_audio, str):
        try:
            return base64.b64decode(_strip_data_url(raw_audio), validate=True), "audio/wav", "audio.wav"
        except Exception as exc:
            raise ValueError("Audio must be base64 encoded.") from exc
    if not isinstance(raw_audio, dict):
        raise ValueError("Audio cue requires an audio object.")

    raw_data = raw_audio.get("data")
    if not isinstance(raw_data, str) or not raw_data:
        raise ValueError("Audio cue requires audio.data as base64.")
    try:
        audio_bytes = base64.b64decode(_strip_data_url(raw_data), validate=True)
    except Exception as exc:
        raise ValueError("Audio data must be valid base64.") from exc
    if not audio_bytes:
        raise ValueError("Audio data cannot be empty.")

    mime_type = str(raw_audio.get("mime_type", raw_audio.get("type", "audio/wav")) or "audio/wav")
    name = str(raw_audio.get("name", "audio.wav") or "audio.wav")
    return audio_bytes, mime_type, name


def _is_wav_audio(audio_bytes: bytes, mime_type: str, name: str) -> bool:
    return (
        audio_bytes.startswith(b"RIFF")
        or "wav" in mime_type.lower()
        or name.lower().endswith(".wav")
    )


def _wav_duration(audio_bytes: bytes) -> float:
    with wave.open(io.BytesIO(audio_bytes), "rb") as reader:
        frame_rate = reader.getframerate()
        if frame_rate <= 0:
            raise ValueError("WAV audio has an invalid frame rate.")
        return reader.getnframes() / frame_rate


def _ffprobe_duration(audio_bytes: bytes, name: str) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    suffix = Path(name).suffix or ".audio"
    with tempfile.NamedTemporaryFile(suffix=suffix) as source:
        source.write(audio_bytes)
        source.flush()
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                source.name,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    if result.returncode != 0:
        return None
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        return None
    return duration if duration > 0 else None


def _audio_duration_seconds(audio_bytes: bytes, mime_type: str, name: str) -> float:
    if _is_wav_audio(audio_bytes, mime_type, name):
        try:
            return _wav_duration(audio_bytes)
        except wave.Error as exc:
            raise ValueError("WAV audio is invalid or unsupported.") from exc
    duration = _ffprobe_duration(audio_bytes, name)
    if duration is None:
        raise ValueError("Could not determine audio duration. Use WAV audio or install ffprobe.")
    return duration


def _slice_wav_audio(audio_bytes: bytes, start: float, end: float) -> bytes:
    with wave.open(io.BytesIO(audio_bytes), "rb") as reader:
        params = reader.getparams()
        frame_rate = reader.getframerate()
        start_frame = max(0, int(round(start * frame_rate)))
        end_frame = min(reader.getnframes(), int(round(end * frame_rate)))
        reader.setpos(start_frame)
        frames = reader.readframes(max(0, end_frame - start_frame))

    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setparams(params)
        writer.writeframes(frames)
    return output.getvalue()


def _ffmpeg_slice_to_wav(audio_bytes: bytes, name: str, start: float, duration: float) -> bytes | None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    suffix = Path(name).suffix or ".audio"
    with tempfile.TemporaryDirectory() as temp_dir:
        source_path = Path(temp_dir) / f"source{suffix}"
        output_path = Path(temp_dir) / "chunk.wav"
        source_path.write_bytes(audio_bytes)
        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start:.6f}",
                "-t",
                f"{duration:.6f}",
                "-i",
                str(source_path),
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                str(output_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not output_path.exists():
            return None
        return output_path.read_bytes()


def _slice_audio_segment(cue: AudioCue, start: float, end: float) -> tuple[bytes, str, str]:
    duration = end - start
    if _is_wav_audio(cue.audio_bytes, cue.mime_type, cue.name):
        return _slice_wav_audio(cue.audio_bytes, start, end), "audio/wav", cue.name
    if start <= 1e-6 and cue.duration <= MAX_AUDIO_CHUNK_SECONDS + 1e-6:
        return cue.audio_bytes, cue.mime_type, cue.name
    chunk = _ffmpeg_slice_to_wav(cue.audio_bytes, cue.name, start, duration)
    if chunk is None:
        raise ValueError("Audio over 10s must be WAV or ffmpeg-sliceable.")
    return chunk, "audio/wav", f"{Path(cue.name).stem}_{start:.1f}_{end:.1f}.wav"


def _normalize_audio_schedule(raw_schedule: Any) -> list[AudioCue]:
    if not isinstance(raw_schedule, list) or not raw_schedule:
        raise ValueError("Offline audio request requires a non-empty schedule.")

    cues: list[AudioCue] = []
    for index, raw in enumerate(raw_schedule):
        if not isinstance(raw, dict):
            raise ValueError("Each offline audio cue must be an object.")
        audio_bytes, mime_type, name = _decode_audio_payload(raw.get("audio"))
        duration = _audio_duration_seconds(audio_bytes, mime_type, name)
        if duration <= 0:
            raise ValueError("Audio duration must be positive.")

        text = str(raw.get("text", "") or "").strip()
        text_start: float | None = None
        text_end: float | None = None
        if text:
            text_start = float(raw.get("text_start", 0) or 0)
            raw_text_end = raw.get("text_end", duration)
            text_end = float(raw_text_end if raw_text_end not in {None, ""} else duration)
            if text_start < 0:
                raise ValueError("text_start must be non-negative.")
            if text_end <= text_start:
                raise ValueError("text_end must be after text_start.")
            if text_end > duration + 1e-3:
                raise ValueError("text_end cannot exceed audio duration.")
        elif str(raw.get("text_start", "") or "") not in {"", "0"} or str(raw.get("text_end", "") or ""):
            raise ValueError("text_start/text_end require non-empty text.")

        cues.append(
            AudioCue(
                audio_bytes=audio_bytes,
                mime_type=mime_type,
                name=name or f"audio_{index}.wav",
                duration=duration,
                text=text,
                text_start=text_start,
                text_end=text_end,
            )
        )
    return cues


def _expand_audio_segments(cues: Iterable[AudioCue]) -> list[AudioSegment]:
    segments: list[AudioSegment] = []
    for source_index, cue in enumerate(cues):
        boundaries = {0.0, cue.duration}
        split = MAX_AUDIO_CHUNK_SECONDS
        while split < cue.duration - 1e-6:
            boundaries.add(split)
            split += MAX_AUDIO_CHUNK_SECONDS
        if cue.text and cue.text_start is not None and cue.text_end is not None:
            boundaries.add(max(0.0, min(cue.duration, cue.text_start)))
            boundaries.add(max(0.0, min(cue.duration, cue.text_end)))

        ordered_boundaries = sorted(boundaries)
        for chunk_index, (start, end) in enumerate(zip(ordered_boundaries, ordered_boundaries[1:])):
            if end <= start + 1e-6:
                continue
            audio_bytes, mime_type, name = _slice_audio_segment(cue, start, end)
            text = ""
            text_start: float | None = None
            text_end: float | None = None
            if cue.text and cue.text_start is not None and cue.text_end is not None:
                overlap_start = max(start, cue.text_start)
                overlap_end = min(end, cue.text_end)
                if overlap_end > overlap_start:
                    text = cue.text
                    text_start = overlap_start - start
                    text_end = overlap_end - start
            segments.append(
                AudioSegment(
                    source_index=source_index,
                    chunk_index=chunk_index,
                    audio_start=start,
                    audio_end=end,
                    duration=end - start,
                    audio_bytes=audio_bytes,
                    mime_type=mime_type,
                    name=name,
                    text=text,
                    text_start=text_start,
                    text_end=text_end,
                )
            )
    return segments


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
    values = np.frombuffer(base64.b64decode(raw_data), dtype="<f4").copy()
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


def _smplx_vertex_count() -> int:
    try:
        smplx_runtime.load()
        if smplx_runtime.v_template is not None:
            return int(smplx_runtime.v_template.shape[0])
    except Exception:
        pass
    return 10475


def _decode_smplx_frame(event: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if event.get("format") == "smplx_params.v1":
        return _decode_smplx_param_frame(event)

    vertex_count = _smplx_vertex_count()
    vertex_values = vertex_count * 3
    joint_values = 22 * 3
    payload_values = vertex_values + joint_values

    if "vertices" in event and "joints" in event:
        vertices = np.asarray(event["vertices"], dtype=np.float32).reshape(vertex_count, 3)
        joints = np.asarray(event["joints"], dtype=np.float32).reshape(22, 3)
        root_position = np.asarray(event.get("root_position", joints[0]), dtype=np.float32).reshape(3)
    else:
        raw_data = event.get("data")
        if not isinstance(raw_data, str):
            raise FloodDiffusionAPIError("Remote SMPL-X frame event is missing data.")
        values = np.frombuffer(base64.b64decode(raw_data), dtype="<f4").copy()
        if values.shape == (payload_values + 3,):
            payload = values[:payload_values]
            root_position = values[payload_values:].reshape(3)
        elif values.shape == (9 + payload_values,):
            root_position = values[1:4].reshape(3)
            payload = values[9:]
        elif values.shape == (payload_values,):
            payload = values
            root_position = payload[vertex_values : vertex_values + 3].reshape(3)
        else:
            raise FloodDiffusionAPIError(f"Unexpected remote SMPL-X frame payload shape: {values.shape}.")

        vertices = payload[:vertex_values].reshape(vertex_count, 3)
        joints = payload[vertex_values : vertex_values + joint_values].reshape(22, 3)

    if not np.isfinite(vertices).all() or not np.isfinite(joints).all() or not np.isfinite(root_position).all():
        raise FloodDiffusionAPIError("Remote SMPL-X frame contains non-finite values.")
    return vertices, joints, root_position


def _decode_smplx_param_frame(event: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root_orient, pose_body, trans, joints, root_position = _decode_smplx_params_payload(event)
    vertices, _skinned_root, _skinned_joints = smplx_runtime.frame_arrays_from_smpl_params(
        root_orient=root_orient,
        pose_body=pose_body,
        trans=trans,
    )
    if not np.isfinite(vertices).all() or not np.isfinite(joints).all() or not np.isfinite(root_position).all():
        raise FloodDiffusionAPIError("Remote SMPL-X params frame contains non-finite values.")
    return vertices, joints, root_position


def _decode_smplx_params_payload(
    event: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_data = event.get("data")
    if not isinstance(raw_data, str):
        raise FloodDiffusionAPIError("Remote SMPL-X params frame event is missing data.")

    values = np.frombuffer(base64.b64decode(raw_data), dtype="<f4").copy()
    pose_body_values = int(event.get("pose_body_values") or (values.size - 3 - 3 - 22 * 3 - 3))
    expected_values = 3 + pose_body_values + 3 + 22 * 3 + 3
    if pose_body_values != 21 * 3 or values.shape != (expected_values,):
        raise FloodDiffusionAPIError(
            f"Unexpected remote SMPL-X params payload shape: {values.shape}, pose_body_values={pose_body_values}."
        )

    cursor = 0
    root_orient = values[cursor : cursor + 3]
    cursor += 3
    pose_body = values[cursor : cursor + pose_body_values]
    cursor += pose_body_values
    trans = values[cursor : cursor + 3]
    cursor += 3
    joints = values[cursor : cursor + 22 * 3].reshape(22, 3)
    cursor += 22 * 3
    root_position = values[cursor : cursor + 3]

    if not np.isfinite(root_orient).all() or not np.isfinite(pose_body).all() or not np.isfinite(trans).all():
        raise FloodDiffusionAPIError("Remote SMPL-X params frame contains non-finite params.")
    if not np.isfinite(joints).all() or not np.isfinite(root_position).all():
        raise FloodDiffusionAPIError("Remote SMPL-X params frame contains non-finite values.")
    return root_orient, pose_body, trans, joints, root_position


def _binary_smplx_params_frame(
    event: dict[str, Any],
    *,
    frame_id: int,
    audio_level: float,
    video_energy: float,
    budget_remaining: float,
    buffer_size: int,
    buffer_capacity: int,
) -> bytes:
    root_orient, pose_body, trans, joints, root_position = _decode_smplx_params_payload(event)
    header = np.array(
        [
            frame_id,
            root_position[0],
            root_position[1],
            root_position[2],
            audio_level,
            video_energy,
            budget_remaining,
            buffer_size,
            buffer_capacity,
        ],
        dtype=np.float32,
    )
    packet = np.concatenate(
        [
            header,
            root_orient.reshape(-1),
            pose_body.reshape(-1),
            trans.reshape(-1),
            joints.reshape(-1),
        ]
    ).astype("<f4", copy=False)
    return packet.tobytes()


async def _stream_offline_audio(
    *,
    send_json,
    send_bytes,
    session_id: str,
    segments: list[AudioSegment],
    seed: int | None,
) -> None:
    assert flood_audio_client is not None
    await send_json(
        {
            "type": "session.created",
            "session_id": session_id,
            "renderer": "smplx",
            "input_mode": "offline_audio",
            "seed": seed,
        }
    )
    await send_json(_motion_format("smplx"))
    await send_json({"type": "session.started", "session_id": session_id})
    await send_json(
        {
            "type": "motion_generation.started",
            "session_id": session_id,
            "renderer": "smplx",
            "provider": "flood_diffusion",
            "model": "chunkdit_concat_201d_audio",
            "mode": "offline_audio",
            "segments": len(segments),
            "checkpoint": str(flood_audio_client.checkpoint_path) if flood_audio_client.checkpoint_path else None,
        }
    )

    queue: asyncio.Queue[Any] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    started_at = time.time()

    def produce() -> None:
        try:
            for item in flood_audio_client.stream_offline_audio_frames(
                segments,
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
    renderer = renderers["smplx"]
    frame_id = 0
    streamed_frames = 0
    streamed_seconds = 0.0
    generation_seconds = 0.0
    segment_count = 0
    stream_completed = True
    playback_started_at: float | None = None

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
                        "renderer": "smplx",
                        "provider": "flood_diffusion",
                        "segment_index": int(item.get("index", item.get("segment_index", -1))),
                        "source_index": int(item.get("source_index", -1)),
                        "chunk_index": int(item.get("chunk_index", -1)),
                        "text": str(item.get("text", "")),
                        "frames": int(item.get("output_frames", item.get("frames", 0))),
                        "target_frames": int(item.get("target_frames", 0)),
                        "generated_frames": int(item.get("generated_frames", 0)),
                        "output_frames": int(item.get("output_frames", item.get("frames", 0))),
                        "fps": round(float(item.get("fps", 30.0)), 3),
                        "generation_seconds": round(float(item.get("generation_seconds", 0.0)), 3),
                        "received_seconds": round(time.time() - started_at, 3),
                    }
                )
                await send_json(
                    {
                        "type": "offline_cue.changed",
                        "session_id": session_id,
                        "cue_index": int(item.get("index", item.get("segment_index", -1))),
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

            frame_id += 1
            fps = float(item.get("fps", 30.0))
            frame_interval = 1.0 / fps
            if playback_started_at is None:
                playback_started_at = time.perf_counter()
            audio_level = float(item.get("audio_level", 0.65))
            video_energy = float(item.get("video_energy", 0.25))
            budget_remaining = round(_budget_remaining(), 2)
            buffer_size = min(4, max(1, queue.qsize()))
            if item.get("format") == "smplx_params.v1":
                await send_bytes(
                    _binary_smplx_params_frame(
                        item,
                        frame_id=frame_id,
                        audio_level=audio_level,
                        video_energy=video_energy,
                        budget_remaining=budget_remaining,
                        buffer_size=buffer_size,
                        buffer_capacity=4,
                    )
                )
            else:
                vertices, joints, root_position = _decode_smplx_frame(item)
                await send_bytes(
                    renderer.binary_frame_from_arrays(
                        vertices=vertices,
                        joints=joints,
                        root_position=root_position,
                        frame_id=frame_id,
                        audio_level=audio_level,
                        video_energy=video_energy,
                        budget_remaining=budget_remaining,
                        buffer_size=buffer_size,
                        buffer_capacity=4,
                    )
                )
            streamed_frames += 1
            streamed_seconds += frame_interval
            async with budget_lock:
                budget_used_seconds += frame_interval
            target_time = playback_started_at + streamed_frames * frame_interval
            await asyncio.sleep(max(0.0, target_time - time.perf_counter()))

        if stream_completed:
            await asyncio.sleep(0.05)
            await send_json(
                {
                    "type": "motion_generation.completed",
                    "session_id": session_id,
                    "renderer": "smplx",
                    "provider": "flood_diffusion",
                    "mode": "offline_audio",
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
    finally:
        with contextlib.suppress(asyncio.CancelledError):
            await producer_task


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (BASE_DIR / "templates" / "index.html").read_text()


@app.post("/api/recordings")
async def save_recording(request: Request) -> dict[str, Any]:
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Recording body is empty.")
    if len(body) > MAX_RECORDING_BYTES:
        raise HTTPException(status_code=413, detail="Recording is too large.")

    mime_type = request.headers.get("content-type", "application/octet-stream")
    requested_name = request.query_params.get("filename", "motion-recording.webm")
    source_suffix = _recording_suffix(mime_type, requested_name)
    recording_id = uuid.uuid4().hex[:12]
    source_path = RECORDINGS_DIR / f"{recording_id}{source_suffix}"
    source_path.write_bytes(body)

    response_path = source_path
    response_mime_type = "video/mp4" if source_suffix == ".mp4" else mime_type
    converted = False
    conversion_error = None
    if source_suffix != ".mp4":
        mp4_path = RECORDINGS_DIR / f"{recording_id}.mp4"
        converted, conversion_error = _transcode_recording_to_mp4(source_path, mp4_path)
        if converted:
            response_path = mp4_path
            response_mime_type = "video/mp4"
            with contextlib.suppress(OSError):
                source_path.unlink()

    return {
        "ok": True,
        "url": _recording_url(response_path),
        "filename": response_path.name,
        "mime_type": response_mime_type,
        "bytes": response_path.stat().st_size,
        "converted": converted,
        "conversion_error": conversion_error,
    }


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
        "flood_offline_audio_enabled": flood_audio_client is not None,
        "flood_diffusion_api_url": flood_audio_client.base_url if flood_audio_client else None,
        "flood_diffusion_audio_endpoint": flood_audio_client.endpoint if flood_audio_client else None,
        "flood_diffusion_checkpoint": str(flood_audio_client.checkpoint_path) if flood_audio_client and flood_audio_client.checkpoint_path else None,
        "flood_diffusion_checkpoint_dir": str(DEFAULT_FLOOD_DIFFUSION_OUTPUT_DIR),
        "offline_audio": {
            "websocket_endpoint": "/api/offline",
            "max_chunk_seconds": MAX_AUDIO_CHUNK_SECONDS,
            "renderer": "smplx",
            "model": "chunkdit_concat_201d_audio",
        },
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
        input_mode = str(payload.get("input_mode", payload.get("mode", "offline")) or "offline").lower()
        seed = None if options["seed"] is None else int(options["seed"])
        if input_mode == "offline_audio":
            if options["renderer"] != "smplx":
                raise ValueError("offline_audio requires renderer=smplx.")
            if flood_audio_client is None:
                raise ValueError("Flood Diffusion audio API is not configured.")
            audio_schedule = _normalize_audio_schedule(payload.get("schedule"))
            audio_segments = _expand_audio_segments(audio_schedule)
        else:
            if options["renderer"] != "g1":
                raise ValueError("Offline text generation currently supports renderer=g1.")
            schedule = _normalize_schedule(payload.get("schedule"))
            if kimodo_g1_client is None:
                raise ValueError("Remote Kimodo G1 API is not configured.")
    except Exception as exc:
        await send_json({"type": "error", "code": "invalid_request", "message": str(exc)})
        await websocket.close()
        return

    if input_mode == "offline_audio":
        try:
            await _stream_offline_audio(
                send_json=send_json,
                send_bytes=send_bytes,
                session_id=session_id,
                segments=audio_segments,
                seed=seed,
            )
        except WebSocketDisconnect:
            return
        except Exception as exc:
            with contextlib.suppress(Exception):
                await send_json({"type": "error", "code": "flood_diffusion_generation_failed", "message": str(exc)})
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
