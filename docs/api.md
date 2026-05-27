# API

## Remote APIs

### FloodDiffusion online API

The online tab uses a GPU-backed FloodDiffusion API. The local FastAPI app does
not load model weights; it proxies browser requests to the configured API:

```bash
FLOODDIFFUSION_API_URL=http://127.0.0.1:7870
```

Expected upstream endpoints:

```text
GET    /health
GET    /v1/smplh/static
POST   /v1/sessions
POST   /v1/sessions/{session_id}/text
GET    /v1/sessions/{session_id}/stream?batch_size=4&realtime=1
DELETE /v1/sessions/{session_id}
```

The stream returns SSE `motion` events with SMPL-H 22-joint rotmats,
translations, and joints. The browser renders the SMPL-H mesh client-side.

### Kimodo offline API

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
GET /api/flooddiffusion/health
GET /api/flooddiffusion/smplh/static
POST /api/flooddiffusion/sessions
POST /api/flooddiffusion/sessions/{session_id}/text
GET /api/flooddiffusion/sessions/{session_id}/stream
DELETE /api/flooddiffusion/sessions/{session_id}
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
