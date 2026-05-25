# Motion Generation Demo

Local FastAPI demo for text-to-motion visualization. The local app owns only the
browser UI, topology endpoints, and WebSocket forwarding. Actual G1 motion
generation runs in the remote Kimodo API.

## Run

Start or tunnel the remote Kimodo API first, then run the local demo:

```bash
KIMODO_G1_API_URL=http://127.0.0.1:9001 \
KIMODO_G1_SEED=11 \
KIMODO_G1_DIFFUSION_STEPS=20 \
uvicorn app:app --host 127.0.0.1 --port 7861
```

Open:

```text
http://127.0.0.1:7861/
```

## Current Code Shape

```text
app.py
  # Local FastAPI entrypoint
  # Serves the page
  # Provides /api/config, /api/g1/topology, /api/smplx/topology
  # Provides WS /api/offline
  # Calls remote Kimodo G1 API and forwards binary motion frames to the frontend

templates/index.html
  # Single demo page

static/css/style.css
  # Page styling

static/js/main.js
  # UI state
  # Reads online/offline text input
  # Opens /api/offline websocket
  # Receives motion frames
  # Handles playback, status, latency, seed, smooth

static/js/avatars/
  # Three.js avatar display code
  # G1 mesh avatar
  # SMPL-X avatar

renderers/
  # Python renderer adapters
  # G1 topology + binary frame encoding
  # SMPL-X topology + binary frame encoding
```

## Local API

Full API details are in `docs/api.md`.

```text
GET /api/config
GET /api/g1/topology
GET /api/smplx/topology
WS  /api/offline
```

`/api/offline` expects the first WebSocket message to contain a schedule and
config:

```json
{
  "renderer": "g1",
  "config": {
    "seed": 11,
    "smooth": 0
  },
  "schedule": [
    { "text": "walk forward", "start": 0 },
    { "text": "dance", "start": 4, "end": 8 }
  ]
}
```

The WebSocket returns JSON lifecycle events plus binary motion frames.

## Assets

Large model and mesh assets are not committed.

- Unitree G1 local renderer assets:
  `.g1_cache/g1skel34/`
- SMPL-X local renderer model:
  `.smplx_cache/SMPLX_NEUTRAL_2020.npz`
- Existing SMPL-X copies on the motion server:
  `/mnt/data/cpfs/motion_data/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz`
  `/mnt/data/cpfs/motion_data/smplx_models/smplx/SMPLX_NEUTRAL.npz`
- Remote Kimodo G1 checkpoint path:
  `/mnt/data/cpfs/haiyang/kimodo-api/checkpoints/Kimodo-G1-RP-v1`

## Requirements

Install Python deps with:

```bash
pip install -r requirements.txt
```
