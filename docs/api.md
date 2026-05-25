# API

## Remote API

Start on `motion-haiyang`:

```bash
cd /mnt/data/cpfs/haiyang/kimodo-api
KIMODO_API_PORT=9001 CUDA_VISIBLE_DEVICES=2 scripts/start_server.sh
```

Tunnel to local:

```bash
ssh -N -L 9001:127.0.0.1:9001 motion-haiyang
```

Supported remote endpoint:

```text
POST http://127.0.0.1:9001/v1/g1/generate_sequence_frames
```

Remote input:

```json
{
  "segments": [{ "text": "walking in a circle", "duration": 4.0 }],
  "diffusion_steps": 20,
  "seed": 11
}
```

Remote output: NDJSON events `segment.completed`, `frame`, `sequence.completed`, `error`.

## Local API

Start local:

```bash
KIMODO_G1_API_URL=http://127.0.0.1:9001 KIMODO_G1_SEED=11 KIMODO_G1_DIFFUSION_STEPS=20 \
uvicorn app:app --host 127.0.0.1 --port 7861
```

Supported local endpoints:

```text
GET /api/config
GET /api/g1/topology
GET /api/smplx/topology
WS  /api/offline
```

Local websocket input:

```json
{
  "renderer": "g1",
  "config": { "seed": 11, "smooth": 0 },
  "schedule": [
    { "text": "walking in a circle", "start": 0 },
    { "text": "dance", "start": 4, "end": 8 }
  ]
}
```

Schedule rules: first `start` is `0`; starts increase; only the last cue needs `end`.

Local websocket output:

```json
{ "type": "session.created", "session_id": "...", "renderer": "g1", "seed": 11 }
{ "type": "motion.format", "format": "g1.v1", "header_float32": 9, "payload": ["joints_34x3", "global_rotations_34x9"] }
{ "type": "motion_generation.segment_completed", "segment_index": 0, "frames": 120, "fps": 30.0, "generation_seconds": 1.2 }
{ "type": "offline_schedule.completed", "elapsed": 8.0 }
{ "type": "error", "code": "invalid_request", "message": "..." }
```

Binary frame output: little-endian `float32`, `417` floats / `1668` bytes.

```text
0..8      header: frame_id, root_xyz, audio_level, video_energy, budget_remaining, buffer_size, buffer_capacity
9..110    joints, shape (34, 3)
111..416  global_rotations, shape (34, 3, 3)
```
