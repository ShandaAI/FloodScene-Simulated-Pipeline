---
title: FloodScene HF Endpoint Simulation
sdk: docker
app_port: 7860
---

# FloodScene HF Endpoint Simulation

Local prototype for a FloodDiffusion-style streaming demo.

The app intentionally does not load heavy models. ASR, VLM, TTS, and motion generation are represented by simulated HF Endpoint clients in `app.py`, so the browser flow can be tested before wiring real Hugging Face Endpoints.

## Run

```bash
uvicorn app:app --host 127.0.0.1 --port 7861
```

Open `http://127.0.0.1:7861`.

## Shape

- Frontend: static HTML/CSS/Three.js.
- Backend: FastAPI.
- Streaming: WebSocket motion frames at 20 FPS.
- Simulated inputs: audio chunks and video keyframes are sent to backend WebSockets.
- Simulated HF Endpoint clients: ASR, VLM, TTS, and FloodDiffusion motion.

The next replacement point is to swap the fake clients for real endpoint clients while keeping the same session and WebSocket contract.

## Hugging Face Space

This shape is meant for a Docker or CPU Space that calls external HF Inference Endpoints. It is not a ZeroGPU Space shape: ZeroGPU is for Gradio SDK Spaces, not this FastAPI/Docker app.

Multiple browser users get separate in-memory sessions and WebSockets in one Space process. The public budget counter is shared across all users in that process; for production, move budget/session state to Redis or a database and add a max-active-session gate.
