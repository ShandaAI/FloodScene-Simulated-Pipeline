"""Session lifecycle helpers for realtime motion streams."""

from __future__ import annotations

import uuid

from renderers.base import RendererRuntime

from .scheduler import TextCue, validate_schedule
from .state import MotionSession


class MotionSessionManager:
    def __init__(self, renderers: dict[str, RendererRuntime], default_renderer: str):
        self.renderers = renderers
        self.default_renderer = default_renderer
        self.sessions: dict[str, MotionSession] = {}

    def create(
        self,
        *,
        renderer_name: str | None,
        input_mode: str,
        initial_text: str,
        schedule: list[object],
        frame_rate: int,
        seed: int | None,
        kimodo_worker_index: int | None,
        stream_realtime: bool,
        charge_budget: bool,
    ) -> MotionSession:
        renderer = (renderer_name or self.default_renderer).lower()
        if renderer not in self.renderers:
            raise ValueError(f"Unknown renderer: {renderer}")
        if input_mode not in {"online", "offline"}:
            raise ValueError(f"Unknown input mode: {input_mode}")

        normalized_schedule: list[TextCue] = []
        current_text = initial_text.strip()
        if input_mode == "offline":
            normalized_schedule = validate_schedule(schedule)
            current_text = normalized_schedule[0].text
        elif not current_text:
            raise ValueError("Online mode requires non-empty initial_text.")

        session = MotionSession(
            session_id=uuid.uuid4().hex[:16],
            renderer_name=renderer,
            input_mode=input_mode,
            current_text=current_text,
            schedule=normalized_schedule,
            frame_rate=frame_rate,
            seed=seed,
            kimodo_worker_index=kimodo_worker_index,
            stream_realtime=stream_realtime,
            charge_budget=charge_budget,
        )
        self.sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> MotionSession | None:
        return self.sessions.get(session_id)

    def close(self, session_id: str) -> MotionSession | None:
        session = self.sessions.pop(session_id, None)
        if session:
            session.running = False
        return session
