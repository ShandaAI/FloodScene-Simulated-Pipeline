"""SMPL-X renderer runtime backed by a neutral SMPL-X model."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .base import RenderInput


class SMPLXRuntime:
    """Minimal SMPL-X LBS runtime backed by the real neutral model .npz."""

    def __init__(self, model_paths: list[Path]):
        self.model_paths = model_paths
        self.loaded = False
        self.load_error: str | None = None
        self.model_path: Path | None = None
        self.v_template: np.ndarray | None = None
        self.faces: np.ndarray | None = None
        self.posedirs: np.ndarray | None = None
        self.shape_beta_count = 0
        self.joints: np.ndarray | None = None
        self.parents: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.left_hand_mean: np.ndarray | None = None
        self.right_hand_mean: np.ndarray | None = None

    def load(self) -> None:
        if self.loaded:
            return
        model_path = next((path for path in self.model_paths if path.exists()), None)
        if not model_path:
            self.load_error = "SMPL-X neutral model not found"
            raise RuntimeError(self.load_error)

        try:
            data = np.load(str(model_path), allow_pickle=True)
            j_regressor = data["J_regressor"]
            if hasattr(j_regressor, "toarray"):
                j_regressor = j_regressor.toarray()
            elif hasattr(j_regressor, "A"):
                j_regressor = np.array(j_regressor.A)

            self.model_path = model_path
            self.v_template = data["v_template"].astype(np.float32)
            self.faces = data["f"].astype(np.uint32)
            self.shape_beta_count = int(data["shapedirs"].shape[-1])
            self.posedirs = data["posedirs"].astype(np.float32)
            self.joints = np.asarray(j_regressor, dtype=np.float32) @ self.v_template
            self.parents = data["kintree_table"][0].astype(np.int64)
            self.parents[0] = -1
            self.parents[self.parents > 10_000] = -1
            self.weights = data["weights"].astype(np.float32)
            self.left_hand_mean = data["hands_meanl"].astype(np.float32)
            self.right_hand_mean = data["hands_meanr"].astype(np.float32)
            self.loaded = True
            self.load_error = None
        except Exception as exc:  # pragma: no cover - surfaced through API.
            self.load_error = str(exc)
            raise

    @property
    def available(self) -> bool:
        try:
            self.load()
        except Exception:
            return False
        return True

    def topology(self) -> dict[str, Any]:
        self.load()
        assert self.v_template is not None
        assert self.faces is not None
        assert self.joints is not None
        assert self.parents is not None
        assert self.weights is not None
        top_indices = np.argsort(self.weights, axis=1)[:, -4:][:, ::-1].astype(np.uint16)
        top_weights = np.take_along_axis(self.weights, top_indices.astype(np.int64), axis=1).astype(np.float32)
        top_sums = np.maximum(top_weights.sum(axis=1, keepdims=True), 1e-8)
        top_weights = top_weights / top_sums
        return {
            "available": True,
            "model": "SMPLX_NEUTRAL_2020",
            "gender": "neutral",
            "beta_mode": "all_zero",
            "beta_count": self.shape_beta_count,
            "model_path": str(self.model_path),
            "vertex_count": int(self.v_template.shape[0]),
            "face_count": int(self.faces.shape[0]),
            "faces": self.faces.reshape(-1).astype(int).tolist(),
            "v_template": self.v_template.reshape(-1).astype(float).tolist(),
            "joints": self.joints.reshape(-1).astype(float).tolist(),
            "parents": self.parents.astype(int).tolist(),
            "skin_indices": top_indices.reshape(-1).astype(int).tolist(),
            "skin_weights": top_weights.reshape(-1).astype(float).tolist(),
        }

    @staticmethod
    def _axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
        aa = axis_angle.astype(np.float32)
        angle = np.linalg.norm(aa, axis=-1, keepdims=True)
        axis = np.divide(aa, np.maximum(angle, 1e-8), out=np.zeros_like(aa), where=angle > 1e-8)
        x = axis[:, 0]
        y = axis[:, 1]
        z = axis[:, 2]
        c = np.cos(angle[:, 0])
        s = np.sin(angle[:, 0])
        one_c = 1.0 - c

        rot = np.empty((aa.shape[0], 3, 3), dtype=np.float32)
        rot[:, 0, 0] = c + x * x * one_c
        rot[:, 0, 1] = x * y * one_c - z * s
        rot[:, 0, 2] = x * z * one_c + y * s
        rot[:, 1, 0] = y * x * one_c + z * s
        rot[:, 1, 1] = c + y * y * one_c
        rot[:, 1, 2] = y * z * one_c - x * s
        rot[:, 2, 0] = z * x * one_c - y * s
        rot[:, 2, 1] = z * y * one_c + x * s
        rot[:, 2, 2] = c + z * z * one_c

        small = angle[:, 0] <= 1e-8
        if np.any(small):
            rot[small] = np.eye(3, dtype=np.float32)
        return rot

    def _pose_for_input(self, render_input: RenderInput, t: float) -> tuple[np.ndarray, np.ndarray]:
        prompt = render_input.prompt.lower()
        audio_boost = 0.65 + render_input.audio_level * 0.55
        video_boost = 0.9 + render_input.video_energy * 0.35
        speed = 1.2 * audio_boost * video_boost

        if "dance" in prompt:
            speed *= 1.55
        elif "slow" in prompt or "relaxed" in prompt:
            speed *= 0.72

        phase = t * speed * math.tau
        step = math.sin(phase)
        counter_step = math.sin(phase + math.pi)
        arm = math.sin(phase + math.pi * 0.7)
        counter_arm = math.sin(phase + math.pi * 1.7)

        poses = np.zeros((55, 3), dtype=np.float32)

        if "circle" in prompt:
            radius = 1.25
            root_x = math.cos(t * 0.45) * radius
            root_z = math.sin(t * 0.45) * radius
            yaw = -t * 0.45 + math.pi * 0.5
        else:
            root_x = math.sin(t * 0.35) * 0.55
            root_z = (t * 0.34) % 5.0 - 2.5
            yaw = 0.18 * math.sin(t * 0.35)

        jump = max(0.0, math.sin(phase * 0.5)) * 0.35 if "jump" in prompt else 0.0
        pelvis = np.array([root_x, 0.96 + jump, root_z], dtype=np.float32)

        poses[0] = [0.02 * math.sin(phase * 0.5), yaw, 0.03 * math.sin(phase * 0.5)]

        hip_swing = 0.48
        knee_base = 0.14
        knee_swing = 0.5
        ankle_swing = 0.18
        poses[1] = [hip_swing * step, 0.02, 0.06]
        poses[2] = [hip_swing * counter_step, -0.02, -0.06]
        poses[4] = [knee_base + knee_swing * max(0.0, -step), 0.0, 0.0]
        poses[5] = [knee_base + knee_swing * max(0.0, -counter_step), 0.0, 0.0]
        poses[7] = [-ankle_swing * max(0.0, step), 0.0, 0.0]
        poses[8] = [-ankle_swing * max(0.0, counter_step), 0.0, 0.0]
        poses[10] = [0.08 * step, 0.0, 0.0]
        poses[11] = [0.08 * counter_step, 0.0, 0.0]

        poses[3] = [0.03 * math.sin(phase * 0.5), 0.0, 0.04 * math.sin(phase * 0.5)]
        poses[6] = [0.02 * math.sin(phase * 0.5), 0.0, 0.03 * math.sin(phase * 0.5)]
        poses[9] = [0.02 * math.sin(phase * 0.5), 0.0, 0.02 * math.sin(phase * 0.5)]
        poses[12] = [0.0, 0.0, 0.02 * math.sin(phase * 0.4)]
        poses[15] = [0.0, 0.0, -0.02 * math.sin(phase * 0.4)]

        wave = max(0.0, math.sin(phase * 1.6)) if "wave" in prompt or render_input.audio_level > 0.7 else 0.0
        if "dance" in prompt:
            wave = max(wave, 0.6 + 0.4 * math.sin(phase * 1.2))

        poses[13] = [0.0, 0.0, 0.12]
        poses[14] = [0.0, 0.0, -0.12]
        poses[16] = [0.18 * counter_arm, 0.0, 0.35 + 0.08 * arm]
        poses[17] = [0.18 * arm - 1.2 * wave, 0.0, -0.35 - 0.08 * counter_arm]
        poses[18] = [0.25 + 0.18 * max(0.0, -counter_arm), 0.0, 0.02]
        poses[19] = [0.25 + 0.45 * wave + 0.18 * max(0.0, -arm), 0.0, -0.02]
        poses[20] = [0.08 * counter_arm, 0.0, 0.0]
        poses[21] = [0.35 * wave, 0.0, 0.0]

        assert self.left_hand_mean is not None
        assert self.right_hand_mean is not None
        poses[25:40] = self.left_hand_mean.reshape(15, 3)
        poses[40:55] = self.right_hand_mean.reshape(15, 3)
        return poses, pelvis

    def frame_arrays(self, render_input: RenderInput, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.load()
        assert self.joints is not None

        poses, pelvis_abs = self._pose_for_input(render_input, t)
        return self._frame_arrays_from_full_pose(poses, pelvis_abs - self.joints[0])

    def frame_arrays_from_smpl_params(
        self,
        *,
        root_orient: np.ndarray,
        pose_body: np.ndarray,
        trans: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        poses = np.zeros((55, 3), dtype=np.float32)
        poses[0] = np.asarray(root_orient, dtype=np.float32).reshape(3)
        poses[1:22] = np.asarray(pose_body, dtype=np.float32).reshape(21, 3)
        return self._frame_arrays_from_full_pose(poses, np.asarray(trans, dtype=np.float32).reshape(3))

    def _frame_arrays_from_full_pose(
        self,
        poses: np.ndarray,
        translation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.load()
        assert self.v_template is not None
        assert self.posedirs is not None
        assert self.joints is not None
        assert self.parents is not None
        assert self.weights is not None

        poses = np.asarray(poses, dtype=np.float32).reshape(55, 3)
        translation = np.asarray(translation, dtype=np.float32).reshape(3)
        rot_mats = self._axis_angle_to_matrix(poses)
        pose_feature = (rot_mats[1:] - np.eye(3, dtype=np.float32)).reshape(-1)
        v_posed = self.v_template + np.einsum("vcp,p->vc", self.posedirs, pose_feature, optimize=True)

        rel_joints = self.joints.copy()
        for joint_idx in range(1, 55):
            parent = int(self.parents[joint_idx])
            if 0 <= parent < 55:
                rel_joints[joint_idx] = self.joints[joint_idx] - self.joints[parent]

        local_tf = np.zeros((55, 4, 4), dtype=np.float32)
        local_tf[:, :3, :3] = rot_mats
        local_tf[:, :3, 3] = rel_joints
        local_tf[:, 3, 3] = 1.0

        global_tf = np.zeros_like(local_tf)
        global_tf[0] = local_tf[0]
        for joint_idx in range(1, 55):
            parent = int(self.parents[joint_idx])
            global_tf[joint_idx] = global_tf[parent] @ local_tf[joint_idx] if 0 <= parent < 55 else local_tf[joint_idx]

        joints_homo = np.zeros((55, 4), dtype=np.float32)
        joints_homo[:, :3] = self.joints
        rest_offsets = np.einsum("jcd,jd->jc", global_tf[:, :3, :], joints_homo, optimize=True)
        rel_tf = global_tf.copy()
        rel_tf[:, :3, 3] -= rest_offsets[:, :3]

        blend_tf = np.einsum("vj,jcd->vcd", self.weights, rel_tf, optimize=True)
        v_homo = np.ones((v_posed.shape[0], 4), dtype=np.float32)
        v_homo[:, :3] = v_posed
        verts = np.einsum("vcd,vd->vc", blend_tf[:, :3, :], v_homo, optimize=True)
        verts += translation[None, :]
        posed_joints = global_tf[:22, :3, 3] + translation[None, :]
        return verts.astype(np.float32), posed_joints[0].astype(np.float32), posed_joints.astype(np.float32)

    def binary_frame(
        self,
        render_input: RenderInput,
        t: float,
        frame_id: int,
        budget_remaining: float,
        buffer_size: int,
        buffer_capacity: int,
    ) -> bytes:
        verts, pelvis_abs, joints = self.frame_arrays(render_input, t)
        return self.binary_frame_from_arrays(
            vertices=verts,
            joints=joints,
            root_position=pelvis_abs,
            frame_id=frame_id,
            audio_level=render_input.audio_level,
            video_energy=render_input.video_energy,
            budget_remaining=budget_remaining,
            buffer_size=buffer_size,
            buffer_capacity=buffer_capacity,
        )

    def binary_frame_from_arrays(
        self,
        *,
        vertices: np.ndarray,
        joints: np.ndarray,
        root_position: np.ndarray,
        frame_id: int,
        audio_level: float,
        video_energy: float,
        budget_remaining: float,
        buffer_size: int,
        buffer_capacity: int,
    ) -> bytes:
        vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        joints = np.asarray(joints, dtype=np.float32).reshape(22, 3)
        root_position = np.asarray(root_position, dtype=np.float32).reshape(3)
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
                vertices.reshape(-1).astype(np.float32),
                joints.reshape(-1).astype(np.float32),
            ]
        )
        return packet.astype("<f4", copy=False).tobytes()
