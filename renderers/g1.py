"""Unitree G1 renderer runtime backed by Kimodo's G1 skeleton assets."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from .base import RenderInput


MUJOCO_TO_KIMODO = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)

G1_JOINT_NAMES = [
    "pelvis_skel",
    "left_hip_pitch_skel",
    "left_hip_roll_skel",
    "left_hip_yaw_skel",
    "left_knee_skel",
    "left_ankle_pitch_skel",
    "left_ankle_roll_skel",
    "left_toe_base",
    "right_hip_pitch_skel",
    "right_hip_roll_skel",
    "right_hip_yaw_skel",
    "right_knee_skel",
    "right_ankle_pitch_skel",
    "right_ankle_roll_skel",
    "right_toe_base",
    "waist_yaw_skel",
    "waist_roll_skel",
    "waist_pitch_skel",
    "left_shoulder_pitch_skel",
    "left_shoulder_roll_skel",
    "left_shoulder_yaw_skel",
    "left_elbow_skel",
    "left_wrist_roll_skel",
    "left_wrist_pitch_skel",
    "left_wrist_yaw_skel",
    "left_hand_roll_skel",
    "right_shoulder_pitch_skel",
    "right_shoulder_roll_skel",
    "right_shoulder_yaw_skel",
    "right_elbow_skel",
    "right_wrist_roll_skel",
    "right_wrist_pitch_skel",
    "right_wrist_yaw_skel",
    "right_hand_roll_skel",
]

G1_PARENTS = np.array(
    [
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        0,
        8,
        9,
        10,
        11,
        12,
        13,
        0,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        17,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
    ],
    dtype=np.int64,
)

G1_CHAINS = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [0, 8, 9, 10, 11, 12, 13, 14],
    [0, 15, 16, 17],
    [17, 18, 19, 20, 21, 22, 23, 24, 25],
    [17, 26, 27, 28, 29, 30, 31, 32, 33],
]

G1_MESH_JOINT_MAP = {
    "pelvis_skel": ["pelvis.STL", "pelvis_contour_link.STL"],
    "left_hip_pitch_skel": ["left_hip_pitch_link.STL"],
    "left_hip_roll_skel": ["left_hip_roll_link.STL"],
    "left_hip_yaw_skel": ["left_hip_yaw_link.STL"],
    "left_knee_skel": ["left_knee_link.STL"],
    "left_ankle_pitch_skel": ["left_ankle_pitch_link.STL"],
    "left_ankle_roll_skel": ["left_ankle_roll_link.STL"],
    "right_hip_pitch_skel": ["right_hip_pitch_link.STL"],
    "right_hip_roll_skel": ["right_hip_roll_link.STL"],
    "right_hip_yaw_skel": ["right_hip_yaw_link.STL"],
    "right_knee_skel": ["right_knee_link.STL"],
    "right_ankle_pitch_skel": ["right_ankle_pitch_link.STL"],
    "right_ankle_roll_skel": ["right_ankle_roll_link.STL"],
    "waist_yaw_skel": ["waist_yaw_link_rev_1_0.STL", "waist_yaw_link.STL"],
    "waist_roll_skel": ["waist_roll_link_rev_1_0.STL", "waist_roll_link.STL"],
    "waist_pitch_skel": [
        "torso_link_rev_1_0.STL",
        "torso_link.STL",
        "logo_link.STL",
        "head_link.STL",
    ],
    "left_shoulder_pitch_skel": ["left_shoulder_pitch_link.STL"],
    "left_shoulder_roll_skel": ["left_shoulder_roll_link.STL"],
    "left_shoulder_yaw_skel": ["left_shoulder_yaw_link.STL"],
    "left_elbow_skel": ["left_elbow_link.STL"],
    "left_wrist_roll_skel": ["left_wrist_roll_link.STL"],
    "left_wrist_pitch_skel": ["left_wrist_pitch_link.STL"],
    "left_wrist_yaw_skel": ["left_wrist_yaw_link.STL", "left_rubber_hand.STL"],
    "right_shoulder_pitch_skel": ["right_shoulder_pitch_link.STL"],
    "right_shoulder_roll_skel": ["right_shoulder_roll_link.STL"],
    "right_shoulder_yaw_skel": ["right_shoulder_yaw_link.STL"],
    "right_elbow_skel": ["right_elbow_link.STL"],
    "right_wrist_roll_skel": ["right_wrist_roll_link.STL"],
    "right_wrist_pitch_skel": ["right_wrist_pitch_link.STL"],
    "right_wrist_yaw_skel": ["right_wrist_yaw_link.STL", "right_rubber_hand.STL"],
}


class G1Runtime:
    """Small numpy runtime for streaming a Kimodo-compatible Unitree G1 rig."""

    def __init__(self, asset_root: Path):
        self.asset_root = asset_root
        self.mesh_dir = asset_root / "meshes" / "g1"
        self.xml_path = asset_root / "xml" / "g1.xml"
        self.skeleton_data_path = asset_root / "skeleton_data.npz"
        self.loaded = False
        self.load_error: str | None = None

        self.neutral_joints: np.ndarray | None = None
        self.rest_pose_local_rot: np.ndarray | None = None
        self.rest_root_height = 0.82
        self.mesh_items: list[dict[str, Any]] = []
        self.joint_axes: dict[str, np.ndarray] = {}
        self.joint_limits: dict[str, tuple[float, float]] = {}

    @property
    def available(self) -> bool:
        try:
            self.load()
        except Exception:
            return False
        return True

    def load(self) -> None:
        if self.loaded:
            return
        if not self.asset_root.exists():
            self.load_error = f"G1 asset root not found: {self.asset_root}"
            raise RuntimeError(self.load_error)
        if not self.mesh_dir.exists():
            self.load_error = f"G1 mesh directory not found: {self.mesh_dir}"
            raise RuntimeError(self.load_error)
        if not self.xml_path.exists():
            self.load_error = f"G1 MuJoCo XML not found: {self.xml_path}"
            raise RuntimeError(self.load_error)
        if not self.skeleton_data_path.exists():
            self.load_error = f"G1 skeleton_data.npz not found: {self.skeleton_data_path}"
            raise RuntimeError(self.load_error)

        try:
            data = np.load(self.skeleton_data_path)
            self.neutral_joints = data["neutral_joints"].astype(np.float32)
            self.rest_pose_local_rot = data["rest_pose_local_rot"].astype(np.float32)
            self.joint_axes, self.joint_limits = self._load_joint_axes_and_limits()
            self.mesh_items = self._load_mesh_items()
            rest_rots = self.rest_pose_local_rot.copy()
            _, rest_joints = self._fk(rest_rots, np.zeros(3, dtype=np.float32))
            self.rest_root_height = float(-rest_joints[:, 1].min() + 0.025)
            self.loaded = True
            self.load_error = None
        except Exception as exc:  # pragma: no cover - surfaced through API.
            self.load_error = str(exc)
            raise

    def topology(self) -> dict[str, Any]:
        self.load()
        return {
            "available": True,
            "model": "Kimodo-G1Skeleton34",
            "robot": "Unitree G1",
            "coordinate_system": "kimodo-y-up-z-forward",
            "joint_count": len(G1_JOINT_NAMES),
            "joint_names": G1_JOINT_NAMES,
            "parents": G1_PARENTS.astype(int).tolist(),
            "chains": G1_CHAINS,
            "mesh_count": len(self.mesh_items),
            "mesh_items": self.mesh_items,
            "joint_axes": {
                name: axis.astype(float).round(6).tolist()
                for name, axis in self.joint_axes.items()
            },
        }

    def _load_mesh_items(self) -> list[dict[str, Any]]:
        mesh_transforms = self._load_mesh_local_transforms()
        items: list[dict[str, Any]] = []
        joint_index = {name: idx for idx, name in enumerate(G1_JOINT_NAMES)}
        for joint_name, mesh_files in G1_MESH_JOINT_MAP.items():
            if joint_name not in joint_index:
                continue
            for mesh_file in mesh_files:
                if not (self.mesh_dir / mesh_file).exists():
                    continue
                geom_pos, geom_rot = mesh_transforms.get(
                    mesh_file,
                    (np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32)),
                )
                items.append(
                    {
                        "mesh_file": mesh_file,
                        "mesh_url": f"/g1_assets/meshes/g1/{mesh_file}",
                        "joint_idx": joint_index[joint_name],
                        "joint_name": joint_name,
                        "geom_pos": geom_pos.astype(float).round(8).tolist(),
                        "geom_rot": geom_rot.astype(float).round(8).reshape(-1).tolist(),
                    }
                )
        return items

    def _load_mesh_local_transforms(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        mesh_file_to_mesh_name: dict[str, str] = {}
        for mesh in root.findall(".//asset/mesh"):
            mesh_name = mesh.get("name")
            mesh_file = mesh.get("file")
            if mesh_name and mesh_file:
                mesh_file_to_mesh_name[mesh_file] = mesh_name

        mesh_name_to_transform: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for geom in root.findall(".//geom"):
            mesh_name = geom.get("mesh")
            if mesh_name is None:
                continue
            geom_pos = self._parse_vector(geom.get("pos"), default=np.zeros(3, dtype=np.float32))
            geom_rot = self._quat_wxyz_to_matrix(geom.get("quat"))
            geom_pos = MUJOCO_TO_KIMODO @ geom_pos
            geom_rot = MUJOCO_TO_KIMODO @ geom_rot @ MUJOCO_TO_KIMODO.T
            mesh_name_to_transform[mesh_name] = (geom_pos.astype(np.float32), geom_rot.astype(np.float32))

        mesh_file_transforms = {}
        for mesh_file, mesh_name in mesh_file_to_mesh_name.items():
            mesh_file_transforms[mesh_file] = mesh_name_to_transform.get(
                mesh_name,
                (np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32)),
            )
        return mesh_file_transforms

    def _load_joint_axes_and_limits(self) -> tuple[dict[str, np.ndarray], dict[str, tuple[float, float]]]:
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        class_axes: dict[str, str] = {}
        class_ranges: dict[str, tuple[float, float]] = {}

        for xml_class in tree.findall(".//default"):
            class_name = xml_class.get("class")
            if not class_name:
                continue
            joint_nodes = xml_class.findall("joint")
            if not joint_nodes:
                continue
            joint_node = joint_nodes[0]
            if joint_node.get("axis"):
                class_axes[class_name] = joint_node.get("axis", "")
            if joint_node.get("range"):
                range_vals = [float(v) for v in joint_node.get("range", "").split()]
                if len(range_vals) == 2:
                    class_ranges[class_name] = (range_vals[0], range_vals[1])

        axes: dict[str, np.ndarray] = {}
        limits: dict[str, tuple[float, float]] = {}
        worldbody = root.find("worldbody")
        if worldbody is None:
            return axes, limits

        for joint in worldbody.findall(".//joint"):
            joint_name = joint.get("name")
            if not joint_name:
                continue
            skel_name = joint_name.replace("_joint", "_skel")
            axis_str = joint.get("axis") or class_axes.get(joint.get("class", ""))
            if axis_str:
                axis = MUJOCO_TO_KIMODO @ np.array([float(v) for v in axis_str.split()], dtype=np.float32)
                norm = np.linalg.norm(axis)
                if norm > 1e-8:
                    axes[skel_name] = (axis / norm).astype(np.float32)

            range_str = joint.get("range")
            if range_str:
                range_vals = [float(v) for v in range_str.split()]
                if len(range_vals) == 2:
                    limits[skel_name] = (range_vals[0], range_vals[1])
            elif joint.get("class") in class_ranges:
                limits[skel_name] = class_ranges[joint.get("class", "")]
        return axes, limits

    @staticmethod
    def _parse_vector(value: str | None, default: np.ndarray) -> np.ndarray:
        if value is None:
            return default.astype(np.float32)
        return np.array([float(v) for v in value.split()], dtype=np.float32)

    @staticmethod
    def _quat_wxyz_to_matrix(value: str | None) -> np.ndarray:
        if value is None:
            return np.eye(3, dtype=np.float32)
        w, x, y, z = [float(v) for v in value.split()]
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if norm <= 1e-8:
            return np.eye(3, dtype=np.float32)
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
        axis = axis.astype(np.float32)
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-8 or abs(angle) <= 1e-8:
            return np.eye(3, dtype=np.float32)
        x, y, z = axis / norm
        c = math.cos(angle)
        s = math.sin(angle)
        one_c = 1.0 - c
        return np.array(
            [
                [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
                [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
                [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
            ],
            dtype=np.float32,
        )

    def _fk(self, local_rots: np.ndarray, root_position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.neutral_joints is not None
        rel_joints = self.neutral_joints.copy()
        for joint_idx, parent in enumerate(G1_PARENTS):
            if parent >= 0:
                rel_joints[joint_idx] = self.neutral_joints[joint_idx] - self.neutral_joints[parent]

        global_rots = np.zeros_like(local_rots, dtype=np.float32)
        posed_joints = np.zeros((len(G1_JOINT_NAMES), 3), dtype=np.float32)
        for joint_idx, parent in enumerate(G1_PARENTS):
            if parent < 0:
                global_rots[joint_idx] = local_rots[joint_idx]
                posed_joints[joint_idx] = rel_joints[joint_idx]
            else:
                global_rots[joint_idx] = global_rots[parent] @ local_rots[joint_idx]
                posed_joints[joint_idx] = posed_joints[parent] + global_rots[parent] @ rel_joints[joint_idx]

        posed_joints += root_position[None, :]
        return global_rots.astype(np.float32), posed_joints.astype(np.float32)

    def frame_arrays(self, render_input: RenderInput, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.load()
        assert self.rest_pose_local_rot is not None

        prompt = render_input.prompt.lower()
        audio_boost = 0.65 + render_input.audio_level * 0.55
        video_boost = 0.9 + render_input.video_energy * 0.35
        speed = 1.2 * audio_boost * video_boost
        if "dance" in prompt:
            speed *= 1.5
        elif "slow" in prompt or "relaxed" in prompt:
            speed *= 0.72

        phase = t * speed * math.tau
        step = math.sin(phase)
        counter_step = math.sin(phase + math.pi)
        arm = math.sin(phase + math.pi * 0.72)
        counter_arm = math.sin(phase + math.pi * 1.72)

        if "circle" in prompt:
            radius = 1.15
            root_x = math.cos(t * 0.42) * radius
            root_z = math.sin(t * 0.42) * radius
            yaw = -t * 0.42 + math.pi * 0.5
        else:
            root_x = math.sin(t * 0.35) * 0.5
            root_z = (t * 0.34) % 5.0 - 2.5
            yaw = 0.16 * math.sin(t * 0.35)

        jump = max(0.0, math.sin(phase * 0.5)) * 0.22 if "jump" in prompt else 0.0
        root_position = np.array([root_x, self.rest_root_height + jump, root_z], dtype=np.float32)

        local_rots = self.rest_pose_local_rot.copy()
        local_rots[0] = self._axis_angle_to_matrix(np.array([0.0, 1.0, 0.0], dtype=np.float32), yaw) @ local_rots[0]

        def set_joint_angle(joint_name: str, angle: float) -> None:
            axis = self.joint_axes.get(joint_name)
            if axis is None:
                return
            lo, hi = self.joint_limits.get(joint_name, (-math.inf, math.inf))
            clamped = max(lo, min(hi, angle))
            joint_idx = G1_JOINT_NAMES.index(joint_name)
            delta = self._axis_angle_to_matrix(axis, clamped)
            local_rots[joint_idx] = self.rest_pose_local_rot[joint_idx] @ delta

        # Legs.
        set_joint_angle("left_hip_pitch_skel", 0.42 * step)
        set_joint_angle("right_hip_pitch_skel", 0.42 * counter_step)
        set_joint_angle("left_hip_roll_skel", 0.035 * math.sin(phase * 0.5) + 0.03)
        set_joint_angle("right_hip_roll_skel", 0.035 * math.sin(phase * 0.5) - 0.03)
        set_joint_angle("left_knee_skel", 0.18 + 0.62 * max(0.0, -step))
        set_joint_angle("right_knee_skel", 0.18 + 0.62 * max(0.0, -counter_step))
        set_joint_angle("left_ankle_pitch_skel", -0.2 * max(0.0, -step) + 0.08 * max(0.0, step))
        set_joint_angle("right_ankle_pitch_skel", -0.2 * max(0.0, -counter_step) + 0.08 * max(0.0, counter_step))
        set_joint_angle("left_ankle_roll_skel", -0.03 * math.sin(phase * 0.5))
        set_joint_angle("right_ankle_roll_skel", 0.03 * math.sin(phase * 0.5))

        # Torso.
        set_joint_angle("waist_yaw_skel", 0.1 * math.sin(phase * 0.5))
        set_joint_angle("waist_roll_skel", 0.04 * math.sin(phase * 0.5))
        set_joint_angle("waist_pitch_skel", 0.04 * math.sin(phase * 0.5 + 0.4))

        wave = max(0.0, math.sin(phase * 1.5)) if "wave" in prompt or render_input.audio_level > 0.7 else 0.0
        if "dance" in prompt:
            wave = max(wave, 0.55 + 0.45 * math.sin(phase * 1.1))

        # Arms.
        set_joint_angle("left_shoulder_pitch_skel", -0.5 * counter_arm)
        set_joint_angle("right_shoulder_pitch_skel", -0.5 * arm - 1.05 * wave)
        set_joint_angle("left_shoulder_roll_skel", 0.18 + 0.08 * math.sin(phase))
        set_joint_angle("right_shoulder_roll_skel", -0.18 - 0.35 * wave - 0.08 * math.sin(phase))
        set_joint_angle("left_shoulder_yaw_skel", 0.1 * math.sin(phase * 0.8))
        set_joint_angle("right_shoulder_yaw_skel", -0.1 * math.sin(phase * 0.8))
        set_joint_angle("left_elbow_skel", 0.35 + 0.16 * max(0.0, -counter_arm))
        set_joint_angle("right_elbow_skel", 0.35 + 0.55 * wave + 0.16 * max(0.0, -arm))
        set_joint_angle("left_wrist_yaw_skel", 0.1 * counter_arm)
        set_joint_angle("right_wrist_yaw_skel", 0.35 * wave)

        global_rots, posed_joints = self._fk(local_rots, root_position)
        return posed_joints, global_rots, root_position

    def binary_frame(
        self,
        render_input: RenderInput,
        t: float,
        frame_id: int,
        budget_remaining: float,
        buffer_size: int,
        buffer_capacity: int,
    ) -> bytes:
        joints, global_rots, root_position = self.frame_arrays(render_input, t)
        return self.binary_frame_from_arrays(
            joints=joints,
            global_rots=global_rots,
            root_position=root_position,
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
        joints: np.ndarray,
        global_rots: np.ndarray,
        root_position: np.ndarray,
        frame_id: int,
        audio_level: float,
        video_energy: float,
        budget_remaining: float,
        buffer_size: int,
        buffer_capacity: int,
    ) -> bytes:
        joints = np.asarray(joints, dtype=np.float32).reshape(len(G1_JOINT_NAMES), 3)
        global_rots = np.asarray(global_rots, dtype=np.float32).reshape(len(G1_JOINT_NAMES), 3, 3)
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
                joints.reshape(-1).astype(np.float32),
                global_rots.reshape(-1).astype(np.float32),
            ]
        )
        return packet.astype("<f4", copy=False).tobytes()
