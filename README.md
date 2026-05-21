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
