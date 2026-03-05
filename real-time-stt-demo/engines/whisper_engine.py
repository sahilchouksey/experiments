"""
Whisper STT engine — /ws/whisper

Uses faster-whisper large-v3-turbo (int8, CUDA) with Silero VAD for
speech/silence detection. Both models are lazy-loaded on first connection.
Each connection gets a deep-copied VAD state to avoid cross-connection
interference. Self-registers into the engine REGISTRY.
"""

import asyncio
import contextlib
import copy
import gc
import json
import os
import queue
import threading
import time

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .base import STTEngine, register
from .filters import is_noise, post_process

# ── Whisper lazy singleton ────────────────────────────────────────────────────
_whisper_model = None
_whisper_lock = threading.Lock()


def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                from faster_whisper import WhisperModel

                preferred_device = os.getenv("WHISPER_DEVICE", "cuda")
                preferred_compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
                print(
                    f"[whisper] loading faster-whisper large-v3-turbo {preferred_compute_type} on {preferred_device} ..."
                )
                try:
                    _whisper_model = WhisperModel(
                        "large-v3-turbo",
                        device=preferred_device,
                        compute_type=preferred_compute_type,
                    )
                except Exception as e:
                    if preferred_device == "cuda":
                        cpu_compute_type = os.getenv("WHISPER_CPU_COMPUTE_TYPE", "int8")
                        print(
                            "[whisper] CUDA unavailable for ctranslate2, falling back to CPU "
                            f"({cpu_compute_type}). Error: {e}"
                        )
                        _whisper_model = WhisperModel(
                            "large-v3-turbo",
                            device="cpu",
                            compute_type=cpu_compute_type,
                        )
                    else:
                        raise
                print("[whisper] model loaded")
    return _whisper_model


def unload_whisper() -> None:
    global _whisper_model
    with _whisper_lock:
        _whisper_model = None
    gc.collect()
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()


# ── Silero VAD lazy singleton ─────────────────────────────────────────────────
_vad_model = None
_vad_lock = threading.Lock()


def get_vad():
    global _vad_model
    if _vad_model is None:
        with _vad_lock:
            if _vad_model is None:
                print("[vad] loading silero-vad ...")
                from silero_vad import load_silero_vad  # noqa: F401

                _vad_model = load_silero_vad(onnx=False)
                print("[vad] loaded")
    return _vad_model


# ── VAD constants ─────────────────────────────────────────────────────────────
_VAD_SR = 16000
_VAD_FRAME = 512  # samples per Silero VAD frame (~32 ms)
_SILENCE_THRESH = 0.35  # probability below this = silence
_SILENCE_FRAMES = 12  # ~400 ms of silence → flush segment
_MIN_SPEECH_SAMPLES = 3200  # 200 ms minimum speech to bother transcribing
_INTERIM_INTERVAL = 1.5  # seconds between rolling partial updates


@register("whisper")
class WhisperEngine(STTEngine):
    def mount(self, app: FastAPI) -> None:
        @app.websocket("/ws/whisper")
        async def ws_whisper(websocket: WebSocket):
            await websocket.accept()
            print("[whisper] client connected")

            filter_noise_enabled = True
            auto_copy_enabled = False

            # Load models on first connection (lazy)
            model = await asyncio.to_thread(get_whisper)
            _vad_base = await asyncio.to_thread(get_vad)

            # Deep-copy the small VAD JIT model so each connection has
            # isolated state. reset_states() on one won't affect others.
            vad = copy.deepcopy(_vad_base)
            vad.reset_states()

            msg_queue: queue.Queue = queue.Queue()

            async def drain_queue():
                while True:
                    try:
                        while True:
                            msg = msg_queue.get_nowait()
                            await websocket.send_json(msg)
                    except queue.Empty:
                        pass
                    except Exception:
                        break
                    await asyncio.sleep(0.04)

            drain_task = asyncio.create_task(drain_queue())

            # Per-connection VAD state
            speech_buf: list[float] = []
            silence_count = 0
            in_speech = False
            line_counter = 0
            last_interim = 0.0

            def flush_and_transcribe() -> None:
                nonlocal line_counter, speech_buf
                if len(speech_buf) < _MIN_SPEECH_SAMPLES:
                    speech_buf = []
                    return
                audio_np = np.array(speech_buf, dtype=np.float32)
                speech_buf = []
                line_counter += 1
                lid = line_counter
                try:
                    segments, _ = model.transcribe(
                        audio_np,
                        language="en",
                        beam_size=1,
                        vad_filter=False,
                        without_timestamps=True,
                    )
                    text = " ".join(s.text.strip() for s in segments).strip()
                except Exception as e:
                    print(f"[whisper] transcribe error: {e}")
                    return
                if filter_noise_enabled and is_noise(text):
                    print(f"[whisper][filter] dropped: {repr(text)}")
                    msg_queue.put_nowait(
                        {
                            "type": "transcription",
                            "line_id": lid,
                            "text": "",
                            "filtered": True,
                        }
                    )
                    return
                text = post_process(text)
                msg_queue.put_nowait(
                    {
                        "type": "transcription",
                        "line_id": lid,
                        "text": text,
                        "auto_copy": auto_copy_enabled,
                        "filtered": False,
                    }
                )

            def rolling_interim() -> None:
                nonlocal line_counter
                if len(speech_buf) < _MIN_SPEECH_SAMPLES:
                    return
                audio_np = np.array(speech_buf, dtype=np.float32)
                lid = line_counter + 1
                try:
                    segments, _ = model.transcribe(
                        audio_np,
                        language="en",
                        beam_size=1,
                        vad_filter=False,
                        without_timestamps=True,
                    )
                    text = " ".join(s.text.strip() for s in segments).strip()
                except Exception:
                    return
                if text:
                    msg_queue.put_nowait(
                        {"type": "partial", "line_id": lid, "text": text}
                    )

            def process_audio_chunk(raw: bytes) -> None:
                nonlocal silence_count, in_speech, last_interim

                samples = np.frombuffer(raw, dtype=np.float32).copy()
                tensor = torch.from_numpy(samples)

                # Pad to a multiple of _VAD_FRAME
                pad = (_VAD_FRAME - len(tensor) % _VAD_FRAME) % _VAD_FRAME
                if pad:
                    tensor = torch.cat([tensor, torch.zeros(pad)])

                # Run Silero VAD frame-by-frame
                speech_prob = 0.0
                for i in range(0, len(tensor), _VAD_FRAME):
                    frame = tensor[i : i + _VAD_FRAME]
                    speech_prob = vad(frame.unsqueeze(0), _VAD_SR).item()

                if speech_prob >= _SILENCE_THRESH:
                    # active speech
                    speech_buf.extend(samples.tolist())
                    silence_count = 0
                    if not in_speech:
                        in_speech = True
                        last_interim = time.time()
                    else:
                        # rolling interim
                        now = time.time()
                        if now - last_interim >= _INTERIM_INTERVAL:
                            last_interim = now
                            rolling_interim()
                else:
                    # silence
                    if in_speech:
                        silence_count += 1
                        # keep buffering during brief silence gaps
                        speech_buf.extend(samples.tolist())
                        if silence_count >= _SILENCE_FRAMES:
                            in_speech = False
                            silence_count = 0
                            flush_and_transcribe()

            def reset_state() -> None:
                nonlocal silence_count, in_speech, last_interim
                speech_buf.clear()
                silence_count = 0
                in_speech = False
                last_interim = 0.0
                vad.reset_states()

            try:
                await websocket.send_json({"type": "ready"})
                while True:
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        break
                    if "bytes" in message and message["bytes"]:
                        await asyncio.to_thread(process_audio_chunk, message["bytes"])
                    elif "text" in message and message["text"]:
                        data = json.loads(message["text"])
                        if data.get("type") == "ping":
                            await websocket.send_json({"type": "pong"})
                        elif data.get("type") == "stop":
                            await asyncio.to_thread(reset_state)
                        elif data.get("type") == "set_filter_noise":
                            filter_noise_enabled = bool(data.get("enabled", True))
                        elif data.get("type") == "set_auto_copy":
                            auto_copy_enabled = bool(data.get("enabled", False))
            except WebSocketDisconnect:
                print("[whisper] client disconnected")
            except RuntimeError as e:
                if "disconnect" in str(e).lower():
                    print("[whisper] client disconnected")
                else:
                    print(f"[whisper] error: {e}")
                    import traceback

                    traceback.print_exc()
            except Exception as e:
                print(f"[whisper] error: {e}")
                import traceback

                traceback.print_exc()
            finally:
                drain_task.cancel()
                print("[whisper] cleaned up")
