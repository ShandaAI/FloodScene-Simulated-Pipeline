---
title: FloodScene Realtime Motion API
sdk: docker
app_port: 7860
---

# FloodScene Realtime Motion API

Local FastAPI prototype for FloodDiffusion-style realtime motion streaming.

The browser opens a realtime session, receives binary motion frames over one
WebSocket, and sends control/text events on that same socket. The visualization
runtime is pluggable: Unitree G1 is the default mesh renderer, and SMPL-X remains
available through the renderer registry.

## Run

```bash
uvicorn app:app --host 127.0.0.1 --port 7861
```

Open `http://127.0.0.1:7861`.

## Asset Sources

Large model and mesh assets are not committed. Put them in the ignored cache
folders below when setting up a new machine.

- SMPL-X body model for the local renderer:
  `.smplx_cache/SMPLX_NEUTRAL_2020.npz`. Download the neutral NPZ body model
  from the official SMPL-X download page after logging in:
  `https://smpl-x.is.tue.mpg.de/download.php`. The renderer uses neutral
  gender with all-zero betas. On the motion server, an existing fallback path is
  `/mnt/data/cpfs/motion_data/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz`.
- Kimodo SMPL-X checkpoint, if SMPL-X generation is enabled later:
  `https://huggingface.co/nvidia/Kimodo-SMPLX-RP-v1`. This model is gated; see
  NVIDIA's Kimodo SMPL-X setup note:
  `https://research.nvidia.com/labs/sil/projects/kimodo/docs/getting_started/installation_smpl.html`.
- Unitree G1 mesh/skeleton assets for the local renderer:
  `.g1_cache/g1skel34/`. Copy the `g1skel34` directory from the official Kimodo
  repo at
  `https://github.com/nv-tlabs/kimodo/tree/main/kimodo/assets/skeletons/g1skel34`.
  The directory should contain `joints.p`, `rest_pose_local_rot.p`,
  `skeleton_data.npz`, `xml/g1.xml`, and `meshes/g1/*.STL`.
- Kimodo G1 generation checkpoint for the remote API:
  `https://huggingface.co/nvidia/Kimodo-G1-RP-v1`. The current motion server
  path is `/mnt/data/cpfs/haiyang/kimodo-api/checkpoints/Kimodo-G1-RP-v1`.

## Code Shape

- `app.py`: FastAPI routes, realtime session WebSocket, budget gate.
- `api/`: Pydantic request schemas.
- `sessions/`: session lifecycle, online/offline text state, schedule lookup.
- `renderers/`: renderer registry and runtime adapters for `g1` and `smplx`.
- `static/js/avatars/`: frontend avatar implementations.
- `static/js/main.js`: UI state machine and realtime client.

## Realtime API

Create a session:

```http
POST /api/realtime/sessions
```

Online payload:

```json
{
  "renderer": "g1",
  "input_mode": "online",
  "initial_text": "walk in a circle.",
  "frame_rate": 20
}
```

Offline payload:

```json
{
  "renderer": "g1",
  "input_mode": "offline",
  "frame_rate": 20,
  "schedule": [
    { "text": "walk forward", "start": 0 },
    { "text": "turn left", "start": 5 },
    { "text": "wave hand", "start": 9, "end": 14 }
  ]
}
```

Connect to:

```text
ws://HOST/api/realtime/sessions/{session_id}
```

The WebSocket sends JSON lifecycle events plus binary motion frames. It accepts
JSON control events:

```json
{ "type": "input_text.append", "text": "wave hand while walking" }
{ "type": "session.pause" }
{ "type": "session.resume" }
{ "type": "session.close" }
```

Offline schedules require the first cue to start at `0`, strictly increasing cue
starts, and an `end` time on the final cue.
