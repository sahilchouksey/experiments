# Moonshine STT Demo

Real-time Speech-to-Text demo using Moonshine Streaming model with WebSocket audio streaming.

## Requirements

- Python 3.10+
- GPU with CUDA support (recommended for faster inference)
- Microphone access

## Setup

1. Activate the virtual environment:
```bash
cd moonshine-stt-demo
source venv/bin/activate
```

2. Install dependencies (already done):
```bash
pip install -r requirements.txt
```

## Run the Demo

```bash
python main.py
```

The server will start on http://localhost:8000

## Usage

1. Open http://localhost:8000 in your browser
2. Wait for the model to load (status will show "✓ Model loaded and ready")
3. Click "Start Recording" and speak
4. See real-time transcription as you speak
5. Click "Stop Recording" when done

## Model

Uses `UsefulSensors/moonshine-streaming-medium` - the best Moonshine streaming model for real-time transcription.

## How it Works

1. **Frontend**: Browser captures microphone audio using MediaRecorder API
2. **Streaming**: Audio is converted to PCM 16-bit and sent via WebSocket
3. **Backend**: Receives audio chunks and processes with Moonshine model
4. **Real-time**: Transcriptions are streamed back and displayed immediately
