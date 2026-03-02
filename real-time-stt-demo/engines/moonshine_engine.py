"""
Moonshine STT engine — /ws/moonshine

Lazy-loads the Moonshine medium-streaming model on the first WebSocket
connection and keeps a long-lived Transcriber per connection.
Self-registers into the engine REGISTRY.
"""

import asyncio
import json
import queue

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from moonshine_voice import Transcriber, TranscriptEventListener
from moonshine_voice.transcriber import LineTextChanged, LineCompleted, LineStarted
from moonshine_voice.moonshine_api import ModelArch

from .base import STTEngine, register
from .filters import is_noise, post_process

MOONSHINE_MODEL_ARCH = ModelArch.MEDIUM_STREAMING


@register("moonshine")
class MoonshineEngine(STTEngine):
    def mount(self, app: FastAPI) -> None:
        @app.websocket("/ws/moonshine")
        async def ws_moonshine(websocket: WebSocket):
            await websocket.accept()
            print("[moonshine] client connected")

            filter_noise_enabled = True
            auto_copy_enabled = False
            msg_queue: queue.Queue = queue.Queue()

            class WsListener(TranscriptEventListener):
                def on_line_started(self, event: LineStarted):
                    msg_queue.put_nowait(
                        {
                            "type": "partial",
                            "line_id": event.line.line_id,
                            "text": event.line.text or "",
                        }
                    )

                def on_line_text_changed(self, event: LineTextChanged):
                    msg_queue.put_nowait(
                        {
                            "type": "partial",
                            "line_id": event.line.line_id,
                            "text": event.line.text or "",
                        }
                    )

                def on_line_completed(self, event: LineCompleted):
                    line_id = event.line.line_id
                    text = event.line.text or ""
                    if filter_noise_enabled and is_noise(text):
                        print(f"[moonshine][filter] dropped: {repr(text)}")
                        msg_queue.put_nowait(
                            {
                                "type": "transcription",
                                "line_id": line_id,
                                "text": "",
                                "filtered": True,
                            }
                        )
                        return
                    text = post_process(text)
                    msg_queue.put_nowait(
                        {
                            "type": "transcription",
                            "line_id": line_id,
                            "text": text,
                            "auto_copy": auto_copy_enabled,
                            "filtered": False,
                        }
                    )

            listener = WsListener()
            transcriber = Transcriber(
                model_arch=MOONSHINE_MODEL_ARCH,
            )
            transcriber.add_listener(listener)
            transcriber.start()
            print("[moonshine] transcriber started")

            async def drain_queue():
                while True:
                    try:
                        while True:
                            msg = msg_queue.get_nowait()
                            await websocket.send_json(msg)
                    except queue.Empty:
                        pass
                    await asyncio.sleep(0.05)

            drain_task = asyncio.create_task(drain_queue())

            try:
                await websocket.send_json({"type": "ready"})
                while True:
                    message = await websocket.receive()
                    if "bytes" in message and message["bytes"]:
                        raw = message["bytes"]
                        audio = np.frombuffer(raw, dtype=np.float32)
                        await asyncio.to_thread(
                            transcriber.add_audio, audio.tolist(), 16000
                        )
                    elif "text" in message and message["text"]:
                        data = json.loads(message["text"])
                        if data.get("type") == "ping":
                            await websocket.send_json({"type": "pong"})
                        elif data.get("type") == "stop":
                            await asyncio.to_thread(transcriber.stop)
                            await asyncio.to_thread(transcriber.start)
                        elif data.get("type") == "set_filter_noise":
                            filter_noise_enabled = bool(data.get("enabled", True))
                        elif data.get("type") == "set_auto_copy":
                            auto_copy_enabled = bool(data.get("enabled", False))
            except WebSocketDisconnect:
                print("[moonshine] client disconnected")
            except Exception as e:
                print(f"[moonshine] error: {e}")
                import traceback

                traceback.print_exc()
            finally:
                drain_task.cancel()
                try:
                    transcriber.stop()
                except Exception:
                    pass
                transcriber.close()
                print("[moonshine] transcriber cleaned up")
