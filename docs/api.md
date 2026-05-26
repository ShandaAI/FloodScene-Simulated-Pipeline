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
POST http://127.0.0.1:9002/v1/smplx/generate_audio_sequence_frames
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

Flood Diffusion audio input uses the `chunkdit_concat_201d_audio` model and
requests compact SMPL-X parameter frames. The local browser keeps the SMPL-X
template, skeleton, and skinning weights loaded once, then applies these
parameters per frame:

```json
{
  "model": "chunkdit_concat_201d_audio",
  "render_format": "smplx_params",
  "checkpoint": "/mnt/data/cpfs/haiyang/FloodDiffusion-Dev/outputs/.../latest.ckpt",
  "segments": [
    {
      "audio": {
        "name": "clip_0.wav",
        "mime_type": "audio/wav",
        "data": "<base64>"
      },
      "duration": 10.0,
      "text": "optional prompt",
      "text_start": 0.0,
      "text_end": 6.4
    }
  ],
  "seed": 11
}
```

## Local API

Start local:

```bash
KIMODO_G1_API_URL=http://127.0.0.1:9001 FLOOD_DIFFUSION_API_URL=http://127.0.0.1:9002 \
FLOOD_DIFFUSION_CHECKPOINT_DIR=/mnt/data/cpfs/haiyang/FloodDiffusion-Dev/outputs/20260526_003412_chunk_dit_201d_concat_mix_l2_l3_random_mini16_beat_test128 \
KIMODO_G1_SEED=11 KIMODO_G1_DIFFUSION_STEPS=20 \
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

Local websocket input for `offline_audio`:

```json
{
  "input_mode": "offline_audio",
  "renderer": "smplx",
  "config": { "seed": 11, "smooth": 0 },
  "schedule": [
    {
      "audio": {
        "name": "clip.wav",
        "mime_type": "audio/wav",
        "data": "<base64-or-data-url>"
      },
      "text": "optional prompt",
      "text_start": 1.2,
      "text_end": 4.8
    }
  ]
}
```

Audio schedule rules: `audio` is required; `text` is optional; `text_start` and
`text_end` are relative to that audio file and only valid when `text` is
present. The server derives duration from the uploaded audio. Inputs longer
than 10 seconds are expanded into multiple chunks before the Flood Diffusion
request. The websocket does not send audio back; the web UI decodes the
uploaded local audio with Web Audio and starts playback when the first returned
motion frame is applied.

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

SMPL-X binary frame output for `offline_audio`: little-endian `float32`,
`144` floats / `576` bytes.

```text
0..8     header: frame_id, root_xyz, audio_level, video_energy, budget_remaining, buffer_size, buffer_capacity
9..11    root_orient, shape (3,)
12..74   pose_body, shape (21, 3)
75..77   trans, shape (3,)
78..143  joints, shape (22, 3)
```
