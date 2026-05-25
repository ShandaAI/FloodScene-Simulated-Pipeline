# API

Base URL:

```text
http://127.0.0.1:7861
```

## Endpoints

```text
GET /api/config
GET /api/g1/topology
GET /api/smplx/topology
WS  /api/offline
```

## WS /api/offline

Open the websocket, then send one JSON message:

```json
{
  "renderer": "g1",
  "config": {
    "seed": 11,
    "smooth": 0
  },
  "schedule": [
    { "text": "walking in a circle", "start": 0 },
    { "text": "dance", "start": 4, "end": 8 }
  ]
}
```

Rules:

- `renderer` must be `g1`.
- `config.seed` is forwarded to Kimodo.
- `config.smooth` is frontend playback smoothing only. Default is `0`.
- `schedule[0].start` must be `0`.
- cue `start` values must increase.
- only the final cue needs `end`; non-final cue end time is the next cue start.

## WebSocket Output

The websocket returns JSON events and binary frame messages.

JSON events:

```json
{ "type": "session.created", "session_id": "...", "renderer": "g1", "input_mode": "offline", "seed": 11 }
{ "type": "motion.format", "renderer": "g1", "format": "g1.v1", "header_float32": 9, "payload": ["joints_34x3", "global_rotations_34x9"] }
{ "type": "session.started", "session_id": "..." }
{ "type": "motion_generation.started", "session_id": "...", "renderer": "g1", "provider": "kimodo", "mode": "offline" }
{ "type": "motion_generation.segment_completed", "session_id": "...", "segment_index": 0, "text": "dance", "frames": 120, "fps": 30.0, "generation_seconds": 1.2 }
{ "type": "offline_cue.changed", "session_id": "...", "cue_index": 0, "text": "dance", "elapsed": 0.0 }
{ "type": "motion_generation.completed", "session_id": "...", "segments": 1, "frames": 120, "duration": 4.0, "generation_seconds": 1.2, "wall_seconds": 5.3 }
{ "type": "offline_schedule.completed", "session_id": "...", "elapsed": 4.0 }
{ "type": "budget_exhausted", "detail": "Public demo budget exhausted for this Space." }
{ "type": "error", "code": "invalid_request", "message": "..." }
```

Binary frame format: little-endian `float32`, total `417` floats / `1668` bytes.

```text
0   frame_id
1   root_x
2   root_y
3   root_z
4   audio_level
5   video_energy
6   budget_remaining
7   buffer_size
8   buffer_capacity
9   joints, shape (34, 3), 102 floats
111 global_rotations, shape (34, 3, 3), 306 floats
```

## Config And Topology

`GET /api/config` returns runtime config and asset availability.

`GET /api/g1/topology` returns G1 joint names, parent indices, chains, mesh items, axes, and limits.

`GET /api/smplx/topology` returns SMPL-X neutral model metadata and flattened face indices.

## Remote Kimodo Dependency

The local server calls:

```text
POST {KIMODO_G1_API_URL}/v1/g1/generate_sequence_frames
```

Remote request body:

```json
{
  "segments": [
    { "text": "walking in a circle", "duration": 4.0 }
  ],
  "diffusion_steps": 20,
  "seed": 11
}
```
