# multi-engine-tts-demo

Real-time multi-engine Text-to-Speech demo. Audio streams to the browser sentence-by-sentence — first chunk plays before the full text is synthesized.

## Demo

https://github.com/user-attachments/assets/1641f23d-952c-46d6-af7a-a7aa6cf3d722

## Engines

| Engine | Params | Backend | Notes |
|---|---|---|---|
| **Kokoro** | 82M | ONNX · CPU/GPU | 54 voices, 8 languages |
| **Qwen3-TTS** | 0.6B | PyTorch · bfloat16 · SDPA | GPU, voice-consistent ICL bootstrap |
| **Chatterbox Turbo** | 350M | PyTorch · CUDA | Paralinguistic tags `[laugh]` `[cough]`, temperature / top-p controls |

## Features

- Real-time streaming via WebSocket — no waiting for full synthesis
- Three swappable TTS engines in one UI
- Sentence-chunked pipeline — producer/consumer with async queue
- Waveform visualizer
- Minimal dark UI
- Plug-and-play engine registry — add a new engine by dropping a file in `engines/`

## Requirements

- Python 3.13+
- CUDA GPU (required for Qwen3-TTS and Chatterbox Turbo)
- Model files in `models/` (see below)

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Kokoro model files

```bash
mkdir -p models
wget -O models/kokoro-v1.0.onnx \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
wget -O models/voices-v1.0.bin \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

### Chatterbox Turbo

Install the wheel with `--no-deps` (avoids numpy version conflicts on Python 3.13):

```bash
pip install chatterbox_tts-0.1.6-py3-none-any.whl --no-deps
```

Model weights are auto-downloaded from Hugging Face on first use.

### Qwen3-TTS

Model is downloaded automatically via `transformers` on first use (~1.4 GB).

## Run

```bash
python main.py
```

Open http://localhost:8000

## Architecture

```
main.py
  └── FastAPI app
        ├── GET  /                   → serves static/index.html
        └── WS   /ws/tts/<engine>   (one route per engine)

engines/
  ├── __init__.py     engine registry (ENGINES list)
  ├── kokoro.py       Kokoro ONNX engine
  ├── qwen.py         Qwen3-TTS engine
  └── chatterbox.py   Chatterbox Turbo engine

static/
  └── index.html      fully dynamic — tabs/panels/JS injected from registry
```

Each engine exposes: `LABEL`, `SUBTITLE`, `TAB_ID`, `WS_PATH`, `CONTROLS`, `EXTRA_JS`, `ws_handler`.  
The HTML is assembled at startup from the registry — no manual HTML edits needed to add an engine.

## GPU acceleration

Kokoro uses ONNX Runtime. To enable CUDA:

```bash
pip install onnxruntime-gpu
```

Qwen3-TTS and Chatterbox Turbo require CUDA and load in `bfloat16` automatically.
