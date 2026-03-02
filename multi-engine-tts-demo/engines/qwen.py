"""
Qwen3-TTS engine.

Engine descriptor consumed by the registry in engines/__init__.py.

Architecture — pipeline producer/consumer:
  - producer() generates WAV bytes sentence-by-sentence into an asyncio.Queue
    (maxsize=2); consumer() drains the queue and sends over the WebSocket.
  - Because _executor has max_workers=1 only one sentence is generated at a
    time, but the send never blocks the next generation kick-off and there is
    zero event-loop gap between successive run_in_executor calls.
  - A shared `cancelled` asyncio.Event lets either side abort the other cleanly
    (e.g. on client disconnect or generation error).

Voice consistency — bootstrap ICL approach:
  - On first use _get_voice_prompt() runs a two-step bootstrap entirely in the
    thread pool (so it never blocks the event loop):

    Step 1 — deterministic reference clip:
      A short phrase is synthesised with do_sample=False and a fixed manual
      seed (torch.manual_seed(42)) using x_vector_only_mode=True.  Greedy
      decoding + a fixed RNG state means the output is bit-for-bit identical
      on every cold start regardless of CUDA state.

    Step 2 — ICL prompt from the reference clip:
      The generated audio is fed back into create_voice_clone_prompt() with
      x_vector_only_mode=False (ICL mode).  ICL mode embeds *both* the
      speaker x-vector AND the codec tokens of the reference clip, giving
      much stronger voice conditioning than x-vector alone.

  - Every subsequent sentence conditions on this single ICL prompt, so all
    sentences share one consistent speaker identity.

WAV encoding:
  - _pcm_to_wav() runs inside _gen() (the thread-pool closure) so it never
    blocks the event loop.

Token budget:
  - max_new_tokens is estimated per-sentence from character count.
  - The codec runs at 12 Hz; English speech at ~150 wpm ≈ 13 chars/sec,
    giving ~0.9 tokens/char.  We use 3× that as a safety buffer (min 128,
    max 1024) — so a 50-char sentence gets ~150 tokens instead of a fixed
    512, cutting generation time for short sentences by ~3×.
  - In ICL mode the ref codes are prepended during *decoding* only, so
    max_new_tokens still controls only newly generated tokens and the
    _estimate_max_tokens formula remains correct.

Streaming note:
  - non_streaming_mode=False only simulates streaming text *input*; the
    library does not support true token-streaming output.  We omit it
    (defaults to False internally) since it has no latency benefit.
"""

from __future__ import annotations

import asyncio
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import WebSocket, WebSocketDisconnect

# ---------------------------------------------------------------------------
# Engine metadata — read by the registry to build tabs / routes
# ---------------------------------------------------------------------------

LABEL = "Qwen3-TTS 0.6B"
SUBTITLE = "GPU · bfloat16 · SDPA"
TAB_ID = "qwen"
WS_PATH = "/ws/tts/qwen"

CONTROLS = [
    {
        "type": "select",
        "id": "q-lang",
        "label": "Language",
        "options": [
            {"value": "english", "label": "English"},
            {"value": "chinese", "label": "Chinese"},
            {"value": "french", "label": "French"},
            {"value": "german", "label": "German"},
            {"value": "italian", "label": "Italian"},
            {"value": "japanese", "label": "Japanese"},
            {"value": "korean", "label": "Korean"},
            {"value": "portuguese", "label": "Portuguese"},
            {"value": "russian", "label": "Russian"},
            {"value": "spanish", "label": "Spanish"},
            {"value": "auto", "label": "Auto"},
        ],
    },
]

EXTRA_JS = r"""
function engineInit_qwen() {}

function buildPayload_qwen(text) {
  return {
    text,
    language: document.getElementById("q-lang").value,
  };
}
"""

# ---------------------------------------------------------------------------
# Thread-pool + lazy model loader
# ---------------------------------------------------------------------------
_executor = ThreadPoolExecutor(max_workers=1)
_qwen = None
_qwen_lock = asyncio.Lock()

# Bootstrap ICL voice prompt — built once after model load and reused for
# every request so every sentence shares one consistent speaker identity.
# See module docstring for the two-step bootstrap process.
_BOOTSTRAP_PHRASE = "The quick brown fox jumps over the lazy dog."
_voice_prompt = None
_voice_prompt_lock = asyncio.Lock()


async def _get_model(model_path: Path):
    global _qwen
    if _qwen is not None:
        return _qwen
    async with _qwen_lock:
        if _qwen is not None:
            return _qwen
        from qwen_tts import Qwen3TTSModel

        loop = asyncio.get_event_loop()
        _qwen = await loop.run_in_executor(
            _executor,
            lambda: Qwen3TTSModel.from_pretrained(
                str(model_path),
                device_map="cuda:0",
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            ),
        )
    return _qwen


async def _get_voice_prompt(model_path: Path):
    """
    Build the shared ICL voice prompt exactly once (bootstrap approach).

    Step 1: generate a deterministic reference clip from silence using
    greedy decoding (do_sample=False) and a fixed RNG seed so the output
    is always identical regardless of prior CUDA state.

    Step 2: feed that clip back as a real ICL reference
    (x_vector_only_mode=False) so subsequent generations condition on both
    the speaker x-vector AND the reference codec tokens — giving much
    stronger and more consistent voice identity than x-vector alone.
    """
    global _voice_prompt
    if _voice_prompt is not None:
        return _voice_prompt
    async with _voice_prompt_lock:
        if _voice_prompt is not None:
            return _voice_prompt
        qwen = await _get_model(model_path)
        loop = asyncio.get_event_loop()

        def _build():
            # Step 1 — deterministic reference clip
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            silence_prompt = qwen.create_voice_clone_prompt(
                ref_audio=(np.zeros(24000, dtype=np.float32), 24000),
                x_vector_only_mode=True,
            )
            ref_wavs, ref_sr = qwen.generate_voice_clone(
                text=_BOOTSTRAP_PHRASE,
                language="english",
                voice_clone_prompt=silence_prompt,
                do_sample=False,
                max_new_tokens=300,
            )
            ref_audio = (ref_wavs[0].astype(np.float32), int(ref_sr or 24000))

            # Step 2 — ICL prompt from the reference clip
            return qwen.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=_BOOTSTRAP_PHRASE,
                x_vector_only_mode=False,
            )

        _voice_prompt = await loop.run_in_executor(_executor, _build)
        print("[qwen] bootstrap ICL voice prompt ready")
    return _voice_prompt


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
# Token budget estimation
# ---------------------------------------------------------------------------
# Codec runs at 12 Hz.  English speech: ~150 wpm ≈ 13 chars/sec → ~0.9
# tokens/char.  We use 3× that as a safety buffer so we never truncate audio.
_TOKENS_PER_CHAR = 0.9 * 3  # ≈ 2.7
_MIN_TOKENS = 128
_MAX_TOKENS = 1024


def _estimate_max_tokens(sentence: str) -> int:
    n = max(_MIN_TOKENS, int(len(sentence) * _TOKENS_PER_CHAR))
    return min(n, _MAX_TOKENS)


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------


async def ws_handler(websocket: WebSocket, model_path: Path):
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
    language: str = req.get("language", "english")

    if not text:
        await websocket.send_text(
            json.dumps({"type": "error", "message": "Empty text"})
        )
        await websocket.close()
        return

    try:
        qwen = await _get_model(model_path)
        voice_prompt = await _get_voice_prompt(model_path)

        sentences = [s.strip() for s in _split_sentences(text) if s.strip()]
        if not sentences:
            await websocket.send_text(json.dumps({"type": "done"}))
            return

        loop = asyncio.get_event_loop()

        # ------------------------------------------------------------------
        # Pipeline: producer generates WAV chunks into a queue; consumer
        # drains it and sends over the WebSocket.
        #
        # Because _executor has max_workers=1 only one sentence is generated
        # at a time, but the send never blocks the next generation kick-off.
        # A shared `cancelled` event lets either side abort the other cleanly.
        # ------------------------------------------------------------------
        queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=2)
        cancelled = asyncio.Event()
        gen_error: list[Exception] = []

        async def _safe_put(item: "bytes | None") -> None:
            """Put into queue without hanging if the consumer has gone away."""
            while not cancelled.is_set():
                try:
                    queue.put_nowait(item)
                    return
                except asyncio.QueueFull:
                    await asyncio.sleep(0.005)
            # Cancelled — make one last non-blocking attempt so the sentinel
            # can wake a still-running consumer.
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                pass

        async def producer() -> None:
            t0 = time.perf_counter()
            try:
                for idx, sentence in enumerate(sentences):
                    if cancelled.is_set():
                        break

                    def _gen(s: str = sentence) -> bytes:
                        wavs, sr = qwen.generate_voice_clone(
                            text=s,
                            language=language,
                            voice_clone_prompt=voice_prompt,
                            temperature=0.1,
                            top_k=10,
                            top_p=0.9,
                            repetition_penalty=1.1,
                            max_new_tokens=_estimate_max_tokens(s),
                        )
                        return _pcm_to_wav(wavs[0], sr or 24000)

                    wav_bytes = await loop.run_in_executor(_executor, _gen)

                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    label = "first audio" if idx == 0 else f"sentence {idx + 1}"
                    print(f"  [qwen] {label} ready: {elapsed_ms:.0f} ms")

                    if cancelled.is_set():
                        break

                    await _safe_put(wav_bytes)

            except Exception as e:
                gen_error.append(e)
            finally:
                await _safe_put(None)  # always deliver sentinel

        async def consumer() -> None:
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    await websocket.send_bytes(item)
            except Exception:
                cancelled.set()

        await asyncio.gather(producer(), consumer())

        if gen_error:
            raise gen_error[0]

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
