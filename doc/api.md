# API

This document describes the APIs that are actually supported by the current
local demo branch.

## Local Server

Run the local server with one remote Kimodo G1 API endpoint:

```bash
KIMODO_G1_API_URL=http://127.0.0.1:9001 \
KIMODO_G1_SEED=11 \
KIMODO_G1_DIFFUSION_STEPS=20 \
uvicorn app:app --host 127.0.0.1 --port 7861
```

Base URL:

```text
http://127.0.0.1:7861
```

## Supported Local APIs

```text
GET /api/config
GET /api/g1/topology
GET /api/smplx/topology
WS  /api/offline
```

`/api/offline` currently supports G1 generation only. SMPL-X is currently a
local renderer/topology path, not a Kimodo generation path.

## GET /api/config

Returns local runtime configuration and asset availability.

Example:

```bash
curl http://127.0.0.1:7861/api/config
```

Response shape:

```json
{
  "mode": "offline-motion",
  "renderer": "g1",
  "visualization": "unitree-g1-stl",
  "g1_available": true,
  "g1_error": null,
  "g1_asset_root": "/path/to/.g1_cache/g1skel34",
  "kimodo_g1_offline_enabled": true,
  "kimodo_g1_api_url": "http://127.0.0.1:9001",
  "smplx_available": true,
  "smplx_error": null,
  "smplx_gender": "neutral",
  "smplx_beta_mode": "all_zero",
  "frame_rate": 20,
  "kimodo_g1_default_seed": 11,
  "public_budget_seconds": 180,
  "budget_used_seconds": 0.0,
  "budget_remaining_seconds": 180.0,
  "offline": {
    "websocket_endpoint": "/api/offline"
  }
}
```

## GET /api/g1/topology

Returns G1 skeleton and mesh metadata used by the frontend avatar.

Example:

```bash
curl http://127.0.0.1:7861/api/g1/topology
```

Important response fields:

```json
{
  "available": true,
  "model": "Kimodo-G1Skeleton34",
  "robot": "Unitree G1",
  "coordinate_system": "kimodo-y-up-z-forward",
  "joint_count": 34,
  "joint_names": ["pelvis_skel"],
  "parents": [-1],
  "chains": [[0, 1, 2, 3, 4, 5, 6, 7]],
  "mesh_count": 0,
  "mesh_items": [],
  "joint_axes": {},
  "joint_limits": {}
}
```

The example above is shortened. The real response includes all 34 joint names,
all parent indices, all skeleton chains, and all mesh items when assets exist.

## GET /api/smplx/topology

Returns SMPL-X mesh metadata for the local SMPL-X avatar path.

Example:

```bash
curl http://127.0.0.1:7861/api/smplx/topology
```

Response shape:

```json
{
  "available": true,
  "model": "SMPLX_NEUTRAL_2020",
  "gender": "neutral",
  "beta_mode": "all_zero",
  "beta_count": 400,
  "model_path": "/path/to/SMPLX_NEUTRAL_2020.npz",
  "vertex_count": 10475,
  "face_count": 20908,
  "faces": [0, 1, 2]
}
```

`faces` is a flattened triangle index array and is large.

## WS /api/offline

Open one WebSocket connection and send exactly one JSON request as the first
client message.

URL:

```text
ws://127.0.0.1:7861/api/offline
```

Client request:

```json
{
  "renderer": "g1",
  "config": {
    "seed": 11,
    "smooth": 0
  },
  "schedule": [
    { "text": "walking in a circle", "start": 0 },
    { "text": "dance", "start": 4 },
    { "text": "sit down", "start": 8, "end": 12 }
  ]
}
```

Current backend behavior:

- `renderer` must be `g1`.
- `config.seed` controls the remote Kimodo seed.
- `config.smooth` is a frontend playback setting. The browser uses it for
  interpolation display; the local backend forwards neither `smooth` nor any
  smoothing value to Kimodo.
- `schedule` must be a non-empty list.
- The first cue must start at `0`.
- Cue starts must be strictly increasing.
- For non-final cues, duration is computed from the next cue's `start`.
- The final cue must include `end`, and `end` must be greater than `start`.

Equivalent minimal Python client:

```python
import asyncio
import json
import websockets


async def main():
    async with websockets.connect("ws://127.0.0.1:7861/api/offline", max_size=None) as ws:
        await ws.send(json.dumps({
            "renderer": "g1",
            "config": {"seed": 11, "smooth": 0},
            "schedule": [
                {"text": "walking in a circle", "start": 0, "end": 4}
            ],
        }))

        while True:
            message = await ws.recv()
            if isinstance(message, bytes):
                print("binary frame bytes:", len(message))
                continue

            event = json.loads(message)
            print(event)
            if event["type"] in {"offline_schedule.completed", "error"}:
                break


asyncio.run(main())
```

## /api/offline JSON Events

The WebSocket sends JSON events and binary frame messages on the same
connection.

Session created:

```json
{
  "type": "session.created",
  "session_id": "abc123",
  "renderer": "g1",
  "input_mode": "offline",
  "seed": 11
}
```

Motion format:

```json
{
  "type": "motion.format",
  "renderer": "g1",
  "format": "g1.v1",
  "header_float32": 9,
  "payload": ["joints_34x3", "global_rotations_34x9"]
}
```

Session started:

```json
{
  "type": "session.started",
  "session_id": "abc123"
}
```

Generation started:

```json
{
  "type": "motion_generation.started",
  "session_id": "abc123",
  "renderer": "g1",
  "provider": "kimodo",
  "mode": "offline"
}
```

Segment completed:

```json
{
  "type": "motion_generation.segment_completed",
  "session_id": "abc123",
  "renderer": "g1",
  "provider": "kimodo",
  "segment_index": 0,
  "text": "walking in a circle",
  "frames": 120,
  "target_frames": 120,
  "generated_frames": 120,
  "output_frames": 120,
  "next_start_frames": 5,
  "fps": 30.0,
  "generation_seconds": 1.23,
  "received_seconds": 1.45
}
```

Cue changed:

```json
{
  "type": "offline_cue.changed",
  "session_id": "abc123",
  "cue_index": 0,
  "text": "walking in a circle",
  "elapsed": 0.0
}
```

Generation completed:

```json
{
  "type": "motion_generation.completed",
  "session_id": "abc123",
  "renderer": "g1",
  "provider": "kimodo",
  "mode": "offline",
  "segments": 1,
  "frames": 120,
  "duration": 4.0,
  "generation_seconds": 1.23,
  "wall_seconds": 5.48
}
```

Offline schedule completed:

```json
{
  "type": "offline_schedule.completed",
  "session_id": "abc123",
  "elapsed": 4.0
}
```

Budget exhausted:

```json
{
  "type": "budget_exhausted",
  "detail": "Public demo budget exhausted for this Space."
}
```

Error:

```json
{
  "type": "error",
  "code": "invalid_request",
  "message": "Offline request requires a non-empty schedule."
}
```

or:

```json
{
  "type": "error",
  "code": "kimodo_g1_generation_failed",
  "message": "Remote Kimodo API is unreachable: ..."
}
```

## /api/offline Binary Frame Format

Binary messages are little-endian `float32`.

G1 packet:

```text
header: 9 floats
payload: joints + global rotations
total: 417 floats = 1668 bytes
```

Header layout:

```text
0 frame_id
1 root_x
2 root_y
3 root_z
4 audio_level
5 video_energy
6 budget_remaining
7 buffer_size
8 buffer_capacity
```

Payload layout:

```text
floats 9..110     joints, shape (34, 3)
floats 111..416   global_rotations, shape (34, 3, 3)
```

The frontend reads this as:

```js
const packet = new Float32Array(event.data);
```

## Remote Kimodo API Used By The Local Server

The local server calls this remote endpoint:

```text
POST {KIMODO_G1_API_URL}/v1/g1/generate_sequence_frames
Accept: application/x-ndjson
Content-Type: application/json
```

Request body:

```json
{
  "segments": [
    { "text": "walking in a circle", "duration": 4.0 },
    { "text": "dance", "duration": 4.0 }
  ],
  "diffusion_steps": 20,
  "seed": 11
}
```

Remote response is NDJSON. The local server currently consumes these remote
event types:

- `segment.completed`
- `frame`
- `sequence.completed`
- `error`

Remote `frame.data` is base64-encoded little-endian `float32`:

```text
joints:           34 * 3
global_rotations: 34 * 9
root_position:    3
total:            411 floats
```

The local server decodes that remote frame and re-encodes it into the local
frontend binary frame format described above.
