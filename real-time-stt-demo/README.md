# Real-time STT Demo

Dual-engine real-time on-device speech-to-text in the browser — **Moonshine** and **Whisper** running side by side, receiving the same live mic audio simultaneously. No cloud, no API key — runs entirely on your machine.

![screenshot](./screenshot.png)

## Engines

| Engine | Model | Mode |
|--------|-------|------|
| [Moonshine](https://github.com/usefulsensors/moonshine) | medium-streaming | incremental — partials update word-by-word |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | large-v3-turbo · int8 | VAD-chunked — rolling interim every 1.5s |

Both engines receive the **same PCM audio** simultaneously via WebSocket fan-out from a single AudioWorklet.

## Features

- Dual-panel layout — Moonshine (left) · Whisper (right)
- Real-time transcription via AudioWorklet → WebSocket → engine
- Whisper lazy-loads on first connection (auto-downloads ~1.6 GB model on first run)
- Shared **filter noise** toggle — drops hallucination artifacts and filler words
- Shared **auto-copy** toggle — copies finalized text to clipboard automatically
- Per-panel **Copy** and **Clear** buttons
- **"new line" voice command** — saying "new line" inserts a paragraph break (command text not shown)
- Keyboard shortcuts: `R` toggle recording · `Esc` stop
- Dark theme, minimal UI

## Requirements

- Python 3.10+ (tested with Python 3.13)
- CUDA GPU recommended (Whisper large-v3-turbo is slow on CPU)
- Microphone access in browser

Check your Python version before install:

```bash
python --version
```

Expected output: `Python 3.10` or newer.

## Setup

```bash
pip install -r requirements.txt
```

The project is tested against `moonshine_voice>=0.0.49`, where `Transcriber`
requires an explicit `model_path`. The app resolves and downloads the proper
Moonshine model path automatically at runtime.

On first run, Whisper large-v3-turbo (~1.6 GB) will auto-download from HuggingFace into your local cache. Moonshine medium-streaming model is also downloaded automatically by the `moonshine-voice` package.

## Run

```bash
python main.py
```

If you see `ctranslate2 was not compiled with CUDA support`, run on CPU by setting:

```bash
WHISPER_DEVICE=cpu python main.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

## Project structure

```
├── main.py                  # FastAPI app — mounts all engines, serves frontend
├── engines/
│   ├── base.py              # STTEngine ABC + REGISTRY + @register decorator
│   ├── filters.py           # Noise filter + post-processing rules
│   ├── moonshine_engine.py  # Moonshine WS endpoint (/ws/moonshine)
│   └── whisper_engine.py    # Whisper WS endpoint (/ws/whisper) + Silero VAD
├── static/
│   └── index.html           # Full frontend — AudioWorklet, dual-panel UI
└── requirements.txt
```

## How it works

1. Browser captures mic audio via `AudioWorklet`, downsamples to 16 kHz float32 PCM
2. Each PCM chunk is sent **simultaneously** to `/ws/moonshine` and `/ws/whisper` as binary frames
3. **Moonshine**: feeds audio to `moonshine-voice` SDK (`Transcriber`); partials stream back incrementally word-by-word
4. **Whisper**: buffers audio, uses Silero VAD to detect speech boundaries, runs `faster-whisper` on completed chunks; rolling interim partials every 1.5s during active speech
5. Finalized transcripts rendered in real time in each panel

## Previous version

The original single-engine Moonshine-only demo lived in this folder before this rewrite. The refactor introduced the multi-engine architecture, Whisper support, and the dual-panel UI.
