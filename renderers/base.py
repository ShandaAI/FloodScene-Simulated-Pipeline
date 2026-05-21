"""Shared renderer runtime contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class RenderInput:
    prompt: str
    audio_level: float
    video_energy: float


class RendererRuntime(Protocol):
    load_error: str | None

    @property
    def available(self) -> bool: ...

    def topology(self) -> dict[str, Any]: ...

    def binary_frame(
        self,
        render_input: RenderInput,
        t: float,
        frame_id: int,
        budget_remaining: float,
        buffer_size: int,
        buffer_capacity: int,
    ) -> bytes: ...

