"""Renderer runtime registry."""

from __future__ import annotations

from pathlib import Path

from .base import RendererRuntime
from .g1 import G1Runtime
from .smplx import SMPLXRuntime


DEFAULT_RENDERER = "g1"


def build_renderer_registry(base_dir: Path) -> dict[str, RendererRuntime]:
    return {
        "g1": G1Runtime(base_dir / ".g1_cache" / "g1skel34"),
        "smplx": SMPLXRuntime(
            [
                base_dir / ".smplx_cache" / "SMPLX_NEUTRAL_2020.npz",
                Path("/mnt/data/cpfs/motion_data/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz"),
            ]
        ),
    }

