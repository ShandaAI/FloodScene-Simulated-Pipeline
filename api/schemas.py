"""Pydantic schemas for the realtime motion API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


InputMode = Literal["online", "offline"]


class TextCueRequest(BaseModel):
    text: str = Field(min_length=1)
    start: float = Field(ge=0)
    end: float | None = Field(default=None, gt=0)


class CreateRealtimeSessionRequest(BaseModel):
    renderer: str = Field(default="g1")
    input_mode: InputMode = Field(default="online")
    initial_text: str = Field(default="walk in a circle.")
    schedule: list[TextCueRequest] = Field(default_factory=list)
    frame_rate: int = Field(default=20, ge=1, le=60)
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)


class InputTextRequest(BaseModel):
    text: str = Field(min_length=1)
