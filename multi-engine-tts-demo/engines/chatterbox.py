"""
Chatterbox Turbo TTS engine.

350M-parameter model by Resemble AI.
Fast, expressive, paralinguistic tag support ([laugh], [cough], etc).
Auto-downloads model weights from Hugging Face on first use.

Engine descriptor consumed by the registry in engines/__init__.py.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import io
import json
import re
import time

import numpy as np
import soundfile as sf
from fastapi import WebSocket, WebSocketDisconnect

# ---------------------------------------------------------------------------
# Engine metadata — read by the registry to build tabs / routes
# ---------------------------------------------------------------------------

LABEL = "Chatterbox Turbo"
SUBTITLE = "350M · local · CUDA"
TAB_ID = "chatterbox"
WS_PATH = "/ws/tts/chatterbox"

CONTROLS = [
    {
        "type": "range",
        "id": "cb-temperature",
        "label": "Temperature",
        "min": 0.1,
        "max": 1.5,
        "step": 0.05,
        "value": 0.8,
        "unit": "",
    },
    {
        "type": "range",
        "id": "cb-topp",
        "label": "Top-p",
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "value": 0.95,
        "unit": "",
    },
]

EXTRA_JS = r"""
function engineInit_chatterbox() {
  ["cb-temperature", "cb-topp"].forEach(id => {
    const el  = document.getElementById(id);
    const val = document.getElementById(id + "-val");
    if (el && val) {
      el.addEventListener("input", () => {
        val.textContent = parseFloat(el.value).toFixed(2);
      });
    }
  });
}

function buildPayload_chatterbox(text) {
  return {
    text,
    temperature: parseFloat(document.getElementById("cb-temperature").value),
    top_p:       parseFloat(document.getElementById("cb-topp").value),
  };
}
"""

# ---------------------------------------------------------------------------
# Thread pool — model is not thread-safe; keep to 1 worker
# ---------------------------------------------------------------------------
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# ---------------------------------------------------------------------------
# Lazy model loader
# ---------------------------------------------------------------------------
_model = None
_model_lock = asyncio.Lock()
_SR = 24_000  # Chatterbox Turbo native sample rate


async def _get_model():
    global _model
    if _model is not None:
        return _model
    async with _model_lock:
        if _model is not None:
            return _model
        import torch
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [chatterbox] loading model on {device} …")
        loop = asyncio.get_event_loop()
        _model = await loop.run_in_executor(
            None, lambda: ChatterboxTurboTTS.from_pretrained(device=device)
        )
    return _model


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    sentences = _SPLIT_RE.split(text.strip())
    merged: list[str] = []
    buf = ""
    for s in sentences:
        if s:
            buf = (buf + " " + s).strip() if buf else s
            if len(buf.split()) >= 4:
                merged.append(buf)
                buf = ""
    if buf:
        merged.append(buf)
    return merged if merged else [text.strip()]


# ---------------------------------------------------------------------------
# Audio helper
# ---------------------------------------------------------------------------


def _pcm_to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(
        buf, samples.astype(np.float32), sample_rate, format="WAV", subtype="FLOAT"
    )
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# WebSocket handler — called by main.py
# ---------------------------------------------------------------------------


async def ws_handler(websocket: WebSocket):
    await websocket.accept()

    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=30)
    except asyncio.TimeoutError:
        await websocket.close()
        return

    try:
        req = json.loads(raw)
    except Exception:
        await websocket.send_text(
            json.dumps({"type": "error", "message": "Bad request"})
        )
        await websocket.close()
        return

    text: str = req.get("text", "").strip()
    temperature: float = float(req.get("temperature", 0.8))
    top_p: float = float(req.get("top_p", 0.95))

    if not text:
        await websocket.send_text(
            json.dumps({"type": "error", "message": "Empty text"})
        )
        await websocket.close()
        return

    sentences = _split_sentences(text)
    await websocket.send_text(json.dumps({"type": "meta", "sentences": len(sentences)}))

    model = await _get_model()
    loop = asyncio.get_event_loop()

    try:
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            t0 = time.perf_counter()

            def _gen(s=sentence, temp=temperature, tp=top_p):
                wav = model.generate(s, temperature=temp, top_p=tp)
                audio_np = wav.squeeze().cpu().numpy()
                return _pcm_to_wav(audio_np, _SR)

            wav_bytes = await loop.run_in_executor(_executor, _gen)
            await websocket.send_bytes(wav_bytes)
            print(
                f"  [chatterbox] latency: {(time.perf_counter() - t0) * 1000:.0f} ms  "
                f"({len(wav_bytes) // 1024} KB)"
            )

        await websocket.send_text(json.dumps({"type": "done"}))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        import traceback

        traceback.print_exc()
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
