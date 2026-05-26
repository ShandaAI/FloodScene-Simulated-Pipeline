import base64
import io
import unittest
import wave
from pathlib import Path

import numpy as np

import app


def make_wav(duration: float, sample_rate: int = 8000) -> bytes:
    frames = int(round(duration * sample_rate))
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(b"\x00\x00" * frames)
    return output.getvalue()


def audio_object(duration: float, name: str = "clip.wav") -> dict[str, str]:
    return {
        "name": name,
        "mime_type": "audio/wav",
        "data": base64.b64encode(make_wav(duration)).decode("ascii"),
    }


class OfflineAudioScheduleTests(unittest.TestCase):
    def test_audio_schedule_uses_audio_duration_and_splits_ten_second_chunks(self) -> None:
        cues = app._normalize_audio_schedule(
            [
                {
                    "audio": audio_object(23.25),
                    "text": "wave hello",
                    "text_start": 8,
                    "text_end": 22,
                }
            ]
        )

        self.assertAlmostEqual(cues[0].duration, 23.25, places=3)
        segments = app._expand_audio_segments(cues)

        self.assertEqual(len(segments), 5)
        self.assertEqual([round(segment.duration, 2) for segment in segments], [8.0, 2.0, 10.0, 2.0, 1.25])
        self.assertEqual(
            [(segment.text_start, segment.text_end) for segment in segments],
            [(None, None), (0.0, 2.0), (0.0, 10.0), (0.0, 2.0), (None, None)],
        )
        self.assertEqual([round(app._wav_duration(segment.audio_bytes), 2) for segment in segments], [8.0, 2.0, 10.0, 2.0, 1.25])

    def test_text_is_optional_for_audio_cues(self) -> None:
        cues = app._normalize_audio_schedule([{"audio": audio_object(2.0)}])

        self.assertEqual(cues[0].text, "")
        self.assertIsNone(cues[0].text_start)
        self.assertIsNone(cues[0].text_end)
        self.assertEqual(len(app._expand_audio_segments(cues)), 1)

    def test_client_payload_marks_chunkdit_smplx_contract(self) -> None:
        segment = app._expand_audio_segments(app._normalize_audio_schedule([{"audio": audio_object(1.0)}]))[0]
        client = app.FloodDiffusionAudioClient(
            "http://flood.example",
            checkpoint_path=Path("/tmp/latest.ckpt"),
            config_path=Path("/tmp/config.yaml"),
            seed=13,
        )

        payload = client._build_payload([segment], seed=None, diffusion_steps=25)

        self.assertEqual(payload["model"], "chunkdit_concat_201d_audio")
        self.assertEqual(payload["render_format"], "smplx_params")
        self.assertEqual(payload["checkpoint"], "/tmp/latest.ckpt")
        self.assertEqual(payload["config"], "/tmp/config.yaml")
        self.assertEqual(payload["seed"], 13)
        self.assertEqual(payload["diffusion_steps"], 25)
        self.assertEqual(payload["segments"][0]["duration"], 1.0)
        self.assertIn("audio", payload["segments"][0])

    def test_decode_smplx_frame_payload(self) -> None:
        vertex_count = app._smplx_vertex_count()
        vertices = np.zeros((vertex_count, 3), dtype=np.float32)
        joints = np.ones((22, 3), dtype=np.float32)
        root = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        payload = np.concatenate([vertices.reshape(-1), joints.reshape(-1), root]).astype("<f4")
        event = {"data": base64.b64encode(payload.tobytes()).decode("ascii")}

        decoded_vertices, decoded_joints, decoded_root = app._decode_smplx_frame(event)

        self.assertEqual(decoded_vertices.shape, (vertex_count, 3))
        self.assertEqual(decoded_joints.shape, (22, 3))
        np.testing.assert_allclose(decoded_root, root)

    def test_decode_smplx_params_frame_payload(self) -> None:
        if not app.smplx_runtime.available:
            self.skipTest("SMPL-X model is not available in this environment.")
        joints = np.ones((22, 3), dtype=np.float32)
        root = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        payload = np.concatenate(
            [
                np.zeros(3, dtype=np.float32),
                np.zeros(21 * 3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                joints.reshape(-1),
                root,
            ]
        ).astype("<f4")
        event = {
            "format": "smplx_params.v1",
            "pose_body_values": 21 * 3,
            "data": base64.b64encode(payload.tobytes()).decode("ascii"),
        }

        decoded_vertices, decoded_joints, decoded_root = app._decode_smplx_frame(event)

        self.assertEqual(decoded_vertices.shape, (app._smplx_vertex_count(), 3))
        np.testing.assert_allclose(decoded_joints, joints)
        np.testing.assert_allclose(decoded_root, root)

    def test_smplx_params_binary_frame_is_compact(self) -> None:
        joints = np.ones((22, 3), dtype=np.float32)
        root = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        payload = np.concatenate(
            [
                np.zeros(3, dtype=np.float32),
                np.zeros(21 * 3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                joints.reshape(-1),
                root,
            ]
        ).astype("<f4")
        event = {
            "format": "smplx_params.v1",
            "pose_body_values": 21 * 3,
            "data": base64.b64encode(payload.tobytes()).decode("ascii"),
        }

        packet = app._binary_smplx_params_frame(
            event,
            frame_id=7,
            audio_level=0.5,
            video_energy=0.25,
            budget_remaining=120.0,
            buffer_size=1,
            buffer_capacity=4,
        )

        values = np.frombuffer(packet, dtype="<f4")
        self.assertEqual(len(packet), (9 + 3 + 21 * 3 + 3 + 22 * 3) * 4)
        self.assertEqual(values[0], 7)
        np.testing.assert_allclose(values[1:4], root)


if __name__ == "__main__":
    unittest.main()
