"""Debug Kimodo G1 online chunk continuity without the browser UI."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from renderers.g1 import G1_CHAINS, G1_JOINT_NAMES


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    data["_client_wall_seconds"] = time.perf_counter() - started
    return data


def motion_arrays(motion: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "posed_joints": np.asarray(motion["posed_joints"], dtype=np.float32),
        "global_rot_mats": np.asarray(motion["global_rot_mats"], dtype=np.float32),
        "root_positions": np.asarray(motion.get("root_positions", motion["posed_joints"]), dtype=np.float32),
    }


def save_motion_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(
        path,
        posed_joints=arrays["posed_joints"],
        global_rot_mats=arrays["global_rot_mats"],
        root_positions=arrays["root_positions"],
    )


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def joint_step_stats(joints: np.ndarray, joint_idx: int) -> dict[str, float]:
    steps = np.linalg.norm(np.diff(joints[:, joint_idx, :], axis=0), axis=-1)
    if steps.size == 0:
        return {"max": 0.0, "mean": 0.0, "p95": 0.0}
    return {
        "max": float(steps.max()),
        "mean": float(steps.mean()),
        "p95": float(np.percentile(steps, 95)),
    }


def seam_step_stats(joints: np.ndarray, joint_idx: int, seam_frame: int, window: int = 15) -> dict[str, float]:
    steps = np.linalg.norm(np.diff(joints[:, joint_idx, :], axis=0), axis=-1)
    if steps.size == 0:
        return {"max": 0.0, "mean": 0.0, "p95": 0.0}
    start = max(0, seam_frame - window)
    end = min(len(steps), seam_frame + window)
    local = steps[start:end]
    return {
        "max": float(local.max()),
        "mean": float(local.mean()),
        "p95": float(np.percentile(local, 95)),
    }


def summarize(
    first_output: dict[str, np.ndarray],
    first_next: dict[str, np.ndarray],
    second_output: dict[str, np.ndarray],
    *,
    context_frames: int,
) -> dict[str, Any]:
    root_idx = 0
    waist_idx = G1_JOINT_NAMES.index("waist_pitch_skel")

    first_last = first_output["posed_joints"][-1]
    second_first = second_output["posed_joints"][0]
    start_first = first_next["posed_joints"][0]
    start_last = first_next["posed_joints"][context_frames - 1]
    second_context_last = second_output["posed_joints"][context_frames - 1]

    return {
        "joint_names": {
            "root": G1_JOINT_NAMES[root_idx],
            "waist": G1_JOINT_NAMES[waist_idx],
        },
        "frames": {
            "first_output": int(first_output["posed_joints"].shape[0]),
            "first_next_start": int(first_next["posed_joints"].shape[0]),
            "second_output": int(second_output["posed_joints"].shape[0]),
        },
        "seam_gaps": {
            "first_output_last_to_first_next_start_first_root": float(
                np.linalg.norm(first_last[root_idx] - start_first[root_idx])
            ),
            "first_output_last_to_second_output_first_root": float(
                np.linalg.norm(first_last[root_idx] - second_first[root_idx])
            ),
            "first_next_start_first_to_second_output_first_root": float(
                np.linalg.norm(start_first[root_idx] - second_first[root_idx])
            ),
            "first_next_start_last_to_second_output_context_last_root": float(
                np.linalg.norm(start_last[root_idx] - second_context_last[root_idx])
            ),
            "first_next_start_first_to_second_output_first_waist": float(
                np.linalg.norm(start_first[waist_idx] - second_first[waist_idx])
            ),
        },
        "second_output_step_stats": {
            "root": joint_step_stats(second_output["posed_joints"], root_idx),
            "waist": joint_step_stats(second_output["posed_joints"], waist_idx),
        },
        "second_output_ranges": {
            "root_min_xyz": second_output["posed_joints"][:, root_idx, :].min(axis=0).astype(float).tolist(),
            "root_max_xyz": second_output["posed_joints"][:, root_idx, :].max(axis=0).astype(float).tolist(),
            "all_joints_min_xyz": second_output["posed_joints"].reshape(-1, 3).min(axis=0).astype(float).tolist(),
            "all_joints_max_xyz": second_output["posed_joints"].reshape(-1, 3).max(axis=0).astype(float).tolist(),
        },
    }


def summarize_full_baseline(full_output: dict[str, np.ndarray], seam_frame: int) -> dict[str, Any]:
    root_idx = 0
    waist_idx = G1_JOINT_NAMES.index("waist_pitch_skel")
    return {
        "frames": int(full_output["posed_joints"].shape[0]),
        "seam_frame": int(seam_frame),
        "root_step_stats_all": joint_step_stats(full_output["posed_joints"], root_idx),
        "waist_step_stats_all": joint_step_stats(full_output["posed_joints"], waist_idx),
        "root_step_stats_near_seam": seam_step_stats(full_output["posed_joints"], root_idx, seam_frame),
        "waist_step_stats_near_seam": seam_step_stats(full_output["posed_joints"], waist_idx, seam_frame),
    }


def equal_axis_limits(joints: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    flat = joints.reshape(-1, 3)
    mins = flat.min(axis=0)
    maxs = flat.max(axis=0)
    center = (mins + maxs) / 2.0
    span = float((maxs - mins).max())
    span = max(span, 1.2)
    pad = span * 0.12
    span += pad
    xlim = (float(center[0] - span / 2), float(center[0] + span / 2))
    zlim = (float(center[2] - span / 2), float(center[2] + span / 2))
    ymin = 0.0 if mins[1] > -0.15 else float(center[1] - span / 2)
    ylim = (ymin, float(ymin + span))
    return xlim, ylim, zlim


def render_skeleton_video(joints: np.ndarray, path: Path, *, fps: float, title: str) -> None:
    xlim, ylim, zlim = equal_axis_limits(joints)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.set_zlim(*ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    ax.view_init(elev=18, azim=-70)
    ax.set_box_aspect((1, 1, 1))

    lines = []
    for chain in G1_CHAINS:
        (line,) = ax.plot([], [], [], lw=2.5)
        lines.append((line, chain))
    scatter = ax.scatter([], [], [], s=12, c="black")

    def update(frame_index: int):
        nonlocal scatter
        frame = joints[frame_index]
        for line, chain in lines:
            pts = frame[np.asarray(chain)]
            line.set_data(pts[:, 0], pts[:, 2])
            line.set_3d_properties(pts[:, 1])
        scatter.remove()
        scatter = ax.scatter(frame[:, 0], frame[:, 2], frame[:, 1], s=12, c="black")
        ax.set_title(f"{title}  frame {frame_index + 1}/{len(joints)}")
        return [line for line, _ in lines] + [scatter]

    writer = FFMpegWriter(fps=fps, bitrate=2200)
    with writer.saving(fig, str(path), dpi=130):
        for frame_index in range(len(joints)):
            update(frame_index)
            writer.grab_frame()
    plt.close(fig)


def render_trajectory_plot(
    first_output: dict[str, np.ndarray],
    first_next: dict[str, np.ndarray],
    second_output: dict[str, np.ndarray],
    path: Path,
) -> None:
    waist_idx = G1_JOINT_NAMES.index("waist_pitch_skel")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    series = [
        ("first_output", first_output["posed_joints"][:, waist_idx, :]),
        ("first_next_start", first_next["posed_joints"][:, waist_idx, :]),
        ("second_output", second_output["posed_joints"][:, waist_idx, :]),
    ]
    for label, pts in series:
        axes[0].plot(pts[:, 0], pts[:, 2], marker="o", markersize=2, label=label)
        axes[1].plot(np.arange(len(pts)), pts[:, 1], marker="o", markersize=2, label=label)
    axes[0].set_title("waist x-z trajectory")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("z")
    axes[0].axis("equal")
    axes[1].set_title("waist y over local frame")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("y")
    axes[1].set_ylim(bottom=0)
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:9000")
    parser.add_argument("--first-text", default="walking in a circle")
    parser.add_argument("--second-text", default="dance")
    parser.add_argument("--first-duration", type=float, default=2.0)
    parser.add_argument("--second-duration", type=float, default=2.0)
    parser.add_argument("--context-frames", type=int, default=5)
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path(".g1_cache") / "debug_g1_chunk" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    common = {
        "diffusion_steps": args.diffusion_steps,
        "seed": args.seed,
        "context_frames": args.context_frames,
    }
    first_payload = {
        **common,
        "text": args.first_text,
        "duration": args.first_duration,
    }
    first = post_json(args.base_url, "/v1/g1/generate_chunk", first_payload, args.timeout)
    first_output = motion_arrays(first["output"])
    first_next = motion_arrays(first["next_start"])

    write_json(out_dir / "first_next_start.json", first["next_start"])
    save_motion_npz(out_dir / "first_output.npz", first_output)
    save_motion_npz(out_dir / "first_next_start.npz", first_next)

    second_payload = {
        **common,
        "text": args.second_text,
        "duration": args.second_duration,
        "start_motion": first["next_start"],
    }
    second = post_json(args.base_url, "/v1/g1/generate_chunk", second_payload, args.timeout)
    second_output = motion_arrays(second["output"])
    save_motion_npz(out_dir / "second_output.npz", second_output)

    full_payload = {
        "schedule": [
            {"text": args.first_text, "start": 0.0},
            {"text": args.second_text, "start": args.first_duration, "end": args.first_duration + args.second_duration},
        ],
        "diffusion_steps": args.diffusion_steps,
        "seed": args.seed,
    }
    full = post_json(args.base_url, "/v1/g1/generate", full_payload, args.timeout)
    full_output = motion_arrays(full)
    save_motion_npz(out_dir / "official_full_generate.npz", full_output)

    stitched_joints = np.concatenate([first_output["posed_joints"], second_output["posed_joints"]], axis=0)
    save_motion_npz(
        out_dir / "stitched_first_then_second.npz",
        {
            "posed_joints": stitched_joints,
            "global_rot_mats": np.concatenate(
                [first_output["global_rot_mats"], second_output["global_rot_mats"]],
                axis=0,
            ),
            "root_positions": np.concatenate(
                [first_output["root_positions"], second_output["root_positions"]],
                axis=0,
            ),
        },
    )

    summary = {
        "base_url": args.base_url,
        "first": {
            "text": args.first_text,
            "duration": args.first_duration,
            "generation_seconds": first.get("generation_seconds"),
            "client_wall_seconds": round(first.get("_client_wall_seconds", 0.0), 3),
            "output_frames": first.get("output_frames"),
            "next_start_frames": first.get("next_start_frames"),
        },
        "second": {
            "text": args.second_text,
            "duration": args.second_duration,
            "generation_seconds": second.get("generation_seconds"),
            "client_wall_seconds": round(second.get("_client_wall_seconds", 0.0), 3),
            "input_start_frames": second.get("input_start_frames"),
            "output_frames": second.get("output_frames"),
            "next_start_frames": second.get("next_start_frames"),
        },
        "official_full_generate": {
            "generation_seconds": full.get("generation_seconds"),
            "client_wall_seconds": round(full.get("_client_wall_seconds", 0.0), 3),
            "num_frames": full.get("num_frames"),
            "continuity": summarize_full_baseline(
                full_output,
                seam_frame=int(round(args.first_duration * float(full.get("fps", 30.0)))),
            ),
        },
        "continuity": summarize(
            first_output,
            first_next,
            second_output,
            context_frames=args.context_frames,
        ),
        "artifacts": {},
    }

    trajectory_path = out_dir / "waist_trajectory.png"
    second_video_path = out_dir / "second_chunk_skeleton.mp4"
    stitched_video_path = out_dir / "stitched_first_then_second_skeleton.mp4"
    full_video_path = out_dir / "official_full_generate_skeleton.mp4"
    render_trajectory_plot(first_output, first_next, second_output, trajectory_path)
    fps = float(second.get("fps", first.get("fps", 30.0)))
    render_skeleton_video(second_output["posed_joints"], second_video_path, fps=fps, title=args.second_text)
    render_skeleton_video(stitched_joints, stitched_video_path, fps=fps, title=f"{args.first_text} -> {args.second_text}")
    render_skeleton_video(full_output["posed_joints"], full_video_path, fps=fps, title="official full generate")

    summary["artifacts"] = {
        "out_dir": str(out_dir.resolve()),
        "first_next_start_json": str((out_dir / "first_next_start.json").resolve()),
        "second_output_npz": str((out_dir / "second_output.npz").resolve()),
        "trajectory_png": str(trajectory_path.resolve()),
        "second_video": str(second_video_path.resolve()),
        "stitched_video": str(stitched_video_path.resolve()),
        "official_full_video": str(full_video_path.resolve()),
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
