"""Runtime state for realtime motion sessions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .scheduler import TextCue, active_cue


@dataclass
class MotionSession:
    session_id: str
    renderer_name: str
    input_mode: str
    current_text: str
    schedule: list[TextCue] = field(default_factory=list)
    frame_rate: int = 20
    created_at: float = field(default_factory=time.time)
    running: bool = True
    paused: bool = False
    frame_id: int = 0
    audio_level: float = 0.35
    video_energy: float = 0.25
    last_cue_index: int | None = None

    def text_for_elapsed(self, elapsed: float) -> tuple[str, int | None, bool]:
        if self.input_mode == "online":
            return self.current_text, None, False

        active = active_cue(self.schedule, elapsed)
        if active is None:
            return self.current_text, self.last_cue_index, False

        cue_index, cue = active
        self.current_text = cue.text
        changed = cue_index != self.last_cue_index
        self.last_cue_index = cue_index
        return cue.text, cue_index, changed

    def offline_finished(self, elapsed: float) -> bool:
        if self.input_mode != "offline" or not self.schedule:
            return False
        final_end = self.schedule[-1].end
        return final_end is not None and elapsed >= final_end

    def set_online_text(self, text: str) -> None:
        if self.input_mode != "online":
            raise ValueError("input_text.append is only available in online mode.")
        stripped = text.strip()
        if not stripped:
            raise ValueError("Online text cannot be empty.")
        self.current_text = stripped
