"""Client for the remote Kimodo Unitree G1 generation service."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


class KimodoG1Error(RuntimeError):
    """Raised when the Kimodo G1 API cannot return a usable motion."""


@dataclass(frozen=True)
class GeneratedG1Motion:
    fps: float
    posed_joints: np.ndarray
    global_rot_mats: np.ndarray
    root_positions: np.ndarray
    prompts: list[dict[str, Any]]
    generation_seconds: float
    wall_seconds: float

    @property
    def num_frames(self) -> int:
        return int(self.posed_joints.shape[0])

    @property
    def duration_seconds(self) -> float:
        return self.num_frames / self.fps if self.fps > 0 else 0.0


@dataclass(frozen=True)
class GeneratedG1Chunk:
    segment_index: int
    text: str
    fps: float
    posed_joints: np.ndarray
    global_rot_mats: np.ndarray
    root_positions: np.ndarray
    generation_seconds: float
    received_seconds: float
    target_frames: int
    generated_frames: int
    output_frames: int
    next_start_frames: int

    @property
    def num_frames(self) -> int:
        return int(self.posed_joints.shape[0])

    @property
    def duration_seconds(self) -> float:
        return self.num_frames / self.fps if self.fps > 0 else 0.0


class KimodoG1Client:
    """Small stdlib HTTP client to keep the app dependency surface lean."""

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
        return cls(
            base_url,
            diffusion_steps=int(os.getenv("KIMODO_G1_DIFFUSION_STEPS", "20")),
            seed=None if seed_raw.lower() in {"", "none", "null"} else int(seed_raw),
            timeout_seconds=float(os.getenv("KIMODO_G1_REQUEST_TIMEOUT", "600")),
        )

    def generate_offline(self, schedule: Iterable[Any]) -> GeneratedG1Motion:
        payload: dict[str, Any] = {
            "schedule": [self._cue_payload(cue) for cue in schedule],
            "diffusion_steps": self.diffusion_steps,
        }
        if self.seed is not None:
            payload["seed"] = self.seed

        started_at = time.perf_counter()
        data = self._post_json("/v1/g1/generate", payload)
        wall_seconds = time.perf_counter() - started_at
        return self._decode_motion(data, wall_seconds)

    def generate_offline_sequence(self, schedule: Iterable[Any]) -> Iterable[GeneratedG1Chunk]:
        cues = list(schedule)
        segments = []
        for cue in cues:
            start = float(getattr(cue, "start"))
            end = getattr(cue, "end", None)
            if end is None:
                raise KimodoG1Error("Offline sequence cues must include end times.")
            duration = float(end) - start
            if duration <= 0:
                raise KimodoG1Error("Offline sequence cue duration must be positive.")
            segments.append({"text": str(getattr(cue, "text")), "duration": duration})

        if len(segments) == 1:
            motion = self.generate_offline(cues)
            yield GeneratedG1Chunk(
                segment_index=0,
                text=segments[0]["text"],
                fps=motion.fps,
                posed_joints=motion.posed_joints,
                global_rot_mats=motion.global_rot_mats,
                root_positions=motion.root_positions,
                generation_seconds=motion.generation_seconds,
                received_seconds=motion.wall_seconds,
                target_frames=motion.num_frames,
                generated_frames=motion.num_frames,
                output_frames=motion.num_frames,
                next_start_frames=0,
            )
            return

        payload: dict[str, Any] = {
            "segments": segments,
            "diffusion_steps": self.diffusion_steps,
        }
        if self.seed is not None:
            payload["seed"] = self.seed

        started_at = time.perf_counter()
        yield from self._post_ndjson("/v1/g1/generate_sequence", payload, started_at)

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise KimodoG1Error(f"Kimodo G1 API returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise KimodoG1Error(f"Kimodo G1 API is unreachable: {exc.reason}") from exc
        except TimeoutError as exc:
            raise KimodoG1Error("Kimodo G1 API request timed out.") from exc

        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise KimodoG1Error("Kimodo G1 API returned invalid JSON.") from exc
        if not isinstance(decoded, dict):
            raise KimodoG1Error("Kimodo G1 API returned an unexpected payload.")
        return decoded

    def _post_ndjson(
        self,
        path: str,
        payload: dict[str, Any],
        started_at: float,
    ) -> Iterable[GeneratedG1Chunk]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
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
                        raise KimodoG1Error("Kimodo G1 API returned invalid NDJSON.") from exc
                    if not isinstance(event, dict):
                        raise KimodoG1Error("Kimodo G1 API returned an unexpected stream event.")
                    if event.get("type") == "error":
                        raise KimodoG1Error(str(event.get("detail", "Kimodo G1 sequence generation failed.")))
                    if event.get("type") != "segment.completed":
                        continue
                    yield self._decode_chunk_event(event, time.perf_counter() - started_at)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise KimodoG1Error(f"Kimodo G1 API returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise KimodoG1Error(f"Kimodo G1 API is unreachable: {exc.reason}") from exc
        except TimeoutError as exc:
            raise KimodoG1Error("Kimodo G1 API request timed out.") from exc

    @staticmethod
    def _cue_payload(cue: Any) -> dict[str, Any]:
        item = {
            "text": str(getattr(cue, "text")),
            "start": float(getattr(cue, "start")),
        }
        end = getattr(cue, "end", None)
        if end is not None:
            item["end"] = float(end)
        return item

    @staticmethod
    def _decode_motion(data: dict[str, Any], wall_seconds: float) -> GeneratedG1Motion:
        fps = float(data.get("fps", 30.0))
        if fps <= 0:
            raise KimodoG1Error("Kimodo G1 API returned a non-positive FPS.")

        joints = np.asarray(data.get("posed_joints"), dtype=np.float32)
        rotations = np.asarray(data.get("global_rot_mats"), dtype=np.float32)
        if joints.ndim != 3 or joints.shape[1:] != (34, 3):
            raise KimodoG1Error(f"Unexpected posed_joints shape: {joints.shape}.")
        if rotations.ndim != 4 or rotations.shape[1:] != (34, 3, 3):
            raise KimodoG1Error(f"Unexpected global_rot_mats shape: {rotations.shape}.")
        if rotations.shape[0] != joints.shape[0]:
            raise KimodoG1Error("Joint and rotation frame counts do not match.")
        if not np.isfinite(joints).all() or not np.isfinite(rotations).all():
            raise KimodoG1Error("Kimodo G1 API returned non-finite motion values.")

        root_positions = np.asarray(data.get("root_positions", joints[:, 0, :]), dtype=np.float32)
        if root_positions.shape != (joints.shape[0], 3):
            root_positions = joints[:, 0, :]

        prompts = data.get("prompts", [])
        if not isinstance(prompts, list):
            prompts = []

        return GeneratedG1Motion(
            fps=fps,
            posed_joints=joints,
            global_rot_mats=rotations,
            root_positions=root_positions,
            prompts=prompts,
            generation_seconds=float(data.get("generation_seconds", wall_seconds)),
            wall_seconds=wall_seconds,
        )

    @staticmethod
    def _decode_chunk_event(event: dict[str, Any], received_seconds: float) -> GeneratedG1Chunk:
        chunk = event.get("chunk")
        if not isinstance(chunk, dict):
            raise KimodoG1Error("Kimodo G1 sequence event is missing chunk data.")
        output = chunk.get("output")
        if not isinstance(output, dict):
            raise KimodoG1Error("Kimodo G1 sequence chunk is missing output motion.")

        fps = float(chunk.get("fps", 30.0))
        if fps <= 0:
            raise KimodoG1Error("Kimodo G1 sequence chunk returned a non-positive FPS.")

        joints = np.asarray(output.get("posed_joints"), dtype=np.float32)
        rotations = np.asarray(output.get("global_rot_mats"), dtype=np.float32)
        if joints.ndim != 3 or joints.shape[1:] != (34, 3):
            raise KimodoG1Error(f"Unexpected sequence posed_joints shape: {joints.shape}.")
        if rotations.ndim != 4 or rotations.shape[1:] != (34, 3, 3):
            raise KimodoG1Error(f"Unexpected sequence global_rot_mats shape: {rotations.shape}.")
        if rotations.shape[0] != joints.shape[0]:
            raise KimodoG1Error("Sequence joint and rotation frame counts do not match.")
        if not np.isfinite(joints).all() or not np.isfinite(rotations).all():
            raise KimodoG1Error("Kimodo G1 sequence returned non-finite motion values.")

        root_positions = np.asarray(output.get("root_positions", joints[:, 0, :]), dtype=np.float32)
        if root_positions.shape != (joints.shape[0], 3):
            root_positions = joints[:, 0, :]
        if not np.isfinite(root_positions).all():
            raise KimodoG1Error("Kimodo G1 sequence returned non-finite root positions.")

        return GeneratedG1Chunk(
            segment_index=int(event.get("index", -1)),
            text=str(event.get("text", chunk.get("text", ""))),
            fps=fps,
            posed_joints=joints,
            global_rot_mats=rotations,
            root_positions=root_positions,
            generation_seconds=float(chunk.get("generation_seconds", 0.0)),
            received_seconds=received_seconds,
            target_frames=int(chunk.get("target_frames", joints.shape[0])),
            generated_frames=int(chunk.get("generated_frames", joints.shape[0])),
            output_frames=int(chunk.get("output_frames", joints.shape[0])),
            next_start_frames=int(chunk.get("next_start_frames", 0)),
        )
