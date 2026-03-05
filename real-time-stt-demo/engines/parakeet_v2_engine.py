"""
Parakeet v2 STT engine — /ws/parakeet-v2

NVIDIA NeMo `nvidia/parakeet-tdt-0.6b-v2` (English-optimized)
with Silero VAD chunking for real-time transcription.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import os
import queue
import time

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .base import STTEngine, register
from .filters import is_noise, post_process
from .parakeet_common import get_parakeet_model, get_vad_base, transcribe_with_parakeet

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"
MODEL_LABEL = "parakeet-v2"

_VAD_SR = 16000
_VAD_FRAME = 512
_SILENCE_THRESH = 0.35
_SILENCE_FRAMES = 12
_MIN_SPEECH_SAMPLES = 3200
_INTERIM_INTERVAL = 1.7


@register("parakeet-v2")
class ParakeetV2Engine(STTEngine):
    def mount(self, app: FastAPI) -> None:
        @app.websocket("/ws/parakeet-v2")
        async def ws_parakeet_v2(websocket: WebSocket):
            await websocket.accept()
            print(f"[{MODEL_LABEL}] client connected")

            filter_noise_enabled = True
            auto_copy_enabled = False
            msg_queue: queue.Queue = queue.Queue()

            try:
                preferred_device = os.getenv("PARAKEET_DEVICE", "cuda")
                model, loaded_device = await asyncio.to_thread(
                    get_parakeet_model, MODEL_NAME, preferred_device
                )
                vad_base = await asyncio.to_thread(get_vad_base)
            except Exception as e:
                err = (
                    f"{MODEL_LABEL} unavailable. Install optional deps: "
                    "pip install -r requirements.txt"
                )
                print(f"[{MODEL_LABEL}] init error: {e}")
                with contextlib.suppress(Exception):
                    await websocket.send_json({"type": "error", "message": err})
                with contextlib.suppress(Exception):
                    await websocket.close()
                return

            print(f"[{MODEL_LABEL}] active device: {loaded_device}")

            with contextlib.suppress(Exception):
                await websocket.send_json(
                    {
                        "type": "info",
                        "message": f"{MODEL_LABEL} device: {loaded_device}",
                    }
                )

            vad = copy.deepcopy(vad_base)
            vad.reset_states()

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
                    text = transcribe_with_parakeet(MODEL_NAME, model, audio_np)
                except Exception as e:
                    print(f"[{MODEL_LABEL}] transcribe error: {e}")
                    return

                if filter_noise_enabled and is_noise(text):
                    msg_queue.put_nowait(
                        {
                            "type": "transcription",
                            "line_id": lid,
                            "text": "",
                            "filtered": True,
                        }
                    )
                    return

                msg_queue.put_nowait(
                    {
                        "type": "transcription",
                        "line_id": lid,
                        "text": post_process(text),
                        "auto_copy": auto_copy_enabled,
                        "filtered": False,
                    }
                )

            def rolling_interim() -> None:
                if len(speech_buf) < _MIN_SPEECH_SAMPLES:
                    return

                audio_np = np.array(speech_buf, dtype=np.float32)
                lid = line_counter + 1
                try:
                    text = transcribe_with_parakeet(MODEL_NAME, model, audio_np)
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

                pad = (_VAD_FRAME - len(tensor) % _VAD_FRAME) % _VAD_FRAME
                if pad:
                    tensor = torch.cat([tensor, torch.zeros(pad)])

                speech_prob = 0.0
                for i in range(0, len(tensor), _VAD_FRAME):
                    frame = tensor[i : i + _VAD_FRAME]
                    speech_prob = vad(frame.unsqueeze(0), _VAD_SR).item()

                if speech_prob >= _SILENCE_THRESH:
                    speech_buf.extend(samples.tolist())
                    silence_count = 0
                    if not in_speech:
                        in_speech = True
                        last_interim = time.time()
                    else:
                        now = time.time()
                        if now - last_interim >= _INTERIM_INTERVAL:
                            last_interim = now
                            rolling_interim()
                elif in_speech:
                    silence_count += 1
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
                print(f"[{MODEL_LABEL}] client disconnected")
            except RuntimeError as e:
                if "disconnect" in str(e).lower():
                    print(f"[{MODEL_LABEL}] client disconnected")
                else:
                    print(f"[{MODEL_LABEL}] error: {e}")
            except Exception as e:
                print(f"[{MODEL_LABEL}] error: {e}")
            finally:
                drain_task.cancel()
                print(f"[{MODEL_LABEL}] cleaned up")
