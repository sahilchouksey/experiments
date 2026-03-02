# experiments

A collection of small experiments and side projects.

## Projects

| Folder | Description |
|--------|-------------|
| [`real-time-stt-demo`](./real-time-stt-demo) | Dual-engine real-time STT in the browser — Moonshine (medium-streaming, incremental) and Whisper (large-v3-turbo, VAD-chunked) running side by side on the same mic input. Fully on-device, no cloud. FastAPI + AudioWorklet frontend. |
| [`multi-engine-tts-demo`](./multi-engine-tts-demo) | Three TTS engines (Kokoro 82M, Qwen3-TTS 0.6B, Chatterbox Turbo 350M) in one UI — audio streams sentence-by-sentence before full synthesis completes. Plug-and-play engine registry, waveform visualizer, minimal dark UI. FastAPI + WebSocket. |
