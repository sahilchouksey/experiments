"""
Kokoro ONNX TTS engine.

Engine descriptor consumed by the registry in engines/__init__.py.
"""

from __future__ import annotations

import asyncio
import io
import json
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import WebSocket, WebSocketDisconnect

# ---------------------------------------------------------------------------
# Engine metadata — read by the registry to build tabs / routes
# ---------------------------------------------------------------------------

LABEL = "Kokoro 82M"
SUBTITLE = "ONNX · local · streaming"
TAB_ID = "kokoro"
WS_PATH = "/ws/tts/kokoro"

# Optional per-engine controls rendered in the UI.
# Each dict must have: type, id, label, and type-specific keys.
CONTROLS = [
    {
        "type": "select",
        "id": "k-voice",
        "label": "Voice",
        "options": [],  # populated dynamically by JS via VOICES_JS_DATA
    },
    {
        "type": "select",
        "id": "k-lang",
        "label": "Lang",
        "options": [
            {"value": "en-us", "label": "English (US)"},
            {"value": "en-gb", "label": "English (GB)"},
        ],
    },
    {
        "type": "range",
        "id": "k-speed",
        "label": "Speed",
        "min": 0.5,
        "max": 2.0,
        "step": 0.05,
        "value": 1.0,
        "unit": "×",
    },
]

# Extra JS snippet injected into the page for this engine only.
# Must define: engineInit_<TAB_ID>(cfg) → called once on DOMContentLoaded,
# and buildPayload_<TAB_ID>() → returns the JSON object sent over the WS.
EXTRA_JS = r"""
const KOKORO_VOICES = {
  "en-us": ["af_heart","af_bella","af_nicole","af_sarah","af_sky","af_nova",
            "af_aoede","af_kore","af_jessica","af_river","af_alloy",
            "am_adam","am_michael","am_fenrir","am_puck","am_echo",
            "am_liam","am_onyx","am_santa"],
  "en-gb": ["bf_emma","bf_alice","bf_lily","bm_george","bm_daniel","bm_fable","bm_lewis"],
};

function engineInit_kokoro() {
  const langSel  = document.getElementById("k-lang");
  const voiceSel = document.getElementById("k-voice");
  const speedEl  = document.getElementById("k-speed");
  const speedVal = document.getElementById("k-speed-val");

  function populateVoices() {
    const lang = langSel.value;
    const list = KOKORO_VOICES[lang] || KOKORO_VOICES["en-us"];
    const prev = voiceSel.value;
    voiceSel.innerHTML = "";
    list.forEach(v => {
      const o = document.createElement("option");
      o.value = v; o.textContent = v;
      voiceSel.appendChild(o);
    });
    if (list.includes(prev)) voiceSel.value = prev;
  }

  langSel.addEventListener("change", populateVoices);
  populateVoices();

  if (speedEl && speedVal) {
    speedEl.addEventListener("input", () => {
      speedVal.textContent = parseFloat(speedEl.value).toFixed(2).replace(/\.?0+$/, "") + "×";
    });
  }
}

function buildPayload_kokoro(text) {
  return {
    text,
    voice: document.getElementById("k-voice").value,
    lang:  document.getElementById("k-lang").value,
    speed: parseFloat(document.getElementById("k-speed").value),
  };
}
"""

# ---------------------------------------------------------------------------
# Lazy model loader
# ---------------------------------------------------------------------------
_kokoro = None
_kokoro_lock = asyncio.Lock()


async def _get_model(model_path: Path, voices_path: Path):
    global _kokoro
    if _kokoro is not None:
        return _kokoro
    async with _kokoro_lock:
        if _kokoro is not None:
            return _kokoro
        from kokoro_onnx import Kokoro

        loop = asyncio.get_event_loop()
        _kokoro = await loop.run_in_executor(
            None, lambda: Kokoro(str(model_path), str(voices_path))
        )
    return _kokoro


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


async def ws_handler(websocket: WebSocket, model_path: Path, voices_path: Path):
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
    voice: str = req.get("voice", "af_heart")
    lang: str = req.get("lang", "en-us")
    speed: float = float(req.get("speed", 1.0))

    if not text:
        await websocket.send_text(
            json.dumps({"type": "error", "message": "Empty text"})
        )
        await websocket.close()
        return

    sentences = _split_sentences(text)
    await websocket.send_text(json.dumps({"type": "meta", "sentences": len(sentences)}))

    kokoro = await _get_model(model_path, voices_path)

    try:
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            t0 = time.perf_counter()
            chunk_idx = 0
            async for samples, sr in kokoro.create_stream(
                sentence, voice=voice, speed=speed, lang=lang
            ):
                wav = _pcm_to_wav(samples, sr)
                await websocket.send_bytes(wav)
                if chunk_idx == 0:
                    print(
                        f"  [kokoro/{voice}] latency: "
                        f"{(time.perf_counter() - t0) * 1000:.0f} ms  "
                        f"({len(wav) // 1024} KB)"
                    )
                chunk_idx += 1

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
