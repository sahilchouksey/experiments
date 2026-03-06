# experiments

A collection of small experiments and side projects.

## Projects

| Folder | Description |
|--------|-------------|
| [`llm-router-workflow-demo`](./llm-router-workflow-demo) | Workflow-style LLM routing demo with Arch + RouteLLM + vLLM Semantic Router, tier visualization, sync `Predict all`, and policy-driven model selection from `copilot-routing-policy.json`. Demo video: [`demo.mp4`](./llm-router-workflow-demo/demo.mp4). |
| [`real-time-stt-demo`](./real-time-stt-demo) | Dual-engine real-time STT in the browser — Moonshine (medium-streaming, incremental) and Whisper (large-v3-turbo, VAD-chunked) running side by side on the same mic input. Fully on-device, no cloud. FastAPI + AudioWorklet frontend. |
| [`multi-engine-tts-demo`](./multi-engine-tts-demo) | Three TTS engines (Kokoro 82M, Qwen3-TTS 0.6B, Chatterbox Turbo 350M) in one UI — audio streams sentence-by-sentence before full synthesis completes. Plug-and-play engine registry, waveform visualizer, minimal dark UI. FastAPI + WebSocket. |
| [`opencode-sdk-selection-demo`](./opencode-sdk-selection-demo) | OpenCode SDK model-selection demo with prompt + model dropdown + submit, streaming thinking/response/tool calls (including MCP tools) using GitHub Copilot provider via local OpenCode server. |
