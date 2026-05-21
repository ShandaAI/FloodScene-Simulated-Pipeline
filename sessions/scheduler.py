"""Text schedule validation and lookup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TextCue:
    text: str
    start: float
    end: float | None = None


def validate_schedule(raw_cues: Iterable[object]) -> list[TextCue]:
    cues: list[TextCue] = []
    for cue in raw_cues:
        text = str(getattr(cue, "text", "")).strip()
        start = float(getattr(cue, "start", 0.0))
        end_value = getattr(cue, "end", None)
        end = None if end_value is None else float(end_value)

        if not text:
            raise ValueError("Offline cue text cannot be empty.")
        if start < 0:
            raise ValueError("Offline cue start times must be non-negative.")
        if end is not None and end <= start:
            raise ValueError("Offline cue end time must be greater than its start time.")
        cues.append(TextCue(text=text, start=start, end=end))

    if not cues:
        raise ValueError("Offline mode requires at least one cue.")
    if cues[0].start != 0:
        raise ValueError("The first offline cue must start at 0.")

    for prev, curr in zip(cues, cues[1:]):
        if curr.start <= prev.start:
            raise ValueError("Offline cue start times must be strictly increasing.")
        if prev.end is not None and prev.end > curr.start:
            raise ValueError("Offline cue end time cannot overlap the next cue.")

    final = cues[-1]
    if final.end is None:
        raise ValueError("The final offline cue must include an end time.")

    normalized: list[TextCue] = []
    for index, cue in enumerate(cues):
        end = cue.end
        if index < len(cues) - 1:
            end = cues[index + 1].start
        normalized.append(TextCue(text=cue.text, start=cue.start, end=end))
    return normalized


def active_cue(cues: list[TextCue], elapsed: float) -> tuple[int, TextCue] | None:
    for index, cue in enumerate(cues):
        assert cue.end is not None
        if cue.start <= elapsed < cue.end:
            return index, cue
    return None
