"""
Moonshine STT Demo - Real-time Speech-to-Text using moonshine-voice SDK
"""

import asyncio
import json
import re
import numpy as np
import queue
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from moonshine_voice import Transcriber, TranscriptEventListener
from moonshine_voice.transcriber import LineTextChanged, LineCompleted, LineStarted
from moonshine_voice.moonshine_api import ModelArch

MODEL_PATH = "/home/xix3r/.cache/moonshine_voice/download.moonshine.ai/model/medium-streaming-en/quantized"
MODEL_ARCH = ModelArch.MEDIUM_STREAMING

# ─── Hallucination / noise filter ────────────────────────────────────────────
_NOISE_PATTERNS = re.compile(
    r"^[\s.,!?;:\-\u2013\u2014\u2026]+$"
    r"|\[.*?\]"
    r"|\(.*?\)"
    r"|^(uh+|um+|mm+|hmm+|huh|ah+|oh|eh)$",
    re.IGNORECASE,
)

_HALLUCINATION_PHRASES = {
    "stop",
    "stop.",
    "stop!",
    "okay",
    "okay.",
    "ok",
    "ok.",
    "thanks",
    "thanks.",
    "thank you",
    "thank you.",
    "you",
    "the",
    ".",
    ",",
    "...",
    "bye",
    "bye.",
    "subscribe",
    "like and subscribe",
}


def is_noise(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if t.lower() in _HALLUCINATION_PHRASES:
        return True
    if _NOISE_PATTERNS.match(t):
        return True
    if len(t) <= 3 and not any(c.isalpha() for c in t):
        return True
    return False


# ─── Rule-based post-processing ──────────────────────────────────────────────
_ENDS_WITH_PUNCT = re.compile(r"[.!?,;:\u2026]$")


def post_process(text: str) -> str:
    """Capitalize the first letter and append a period if no terminal punctuation."""
    t = text.strip()
    if not t:
        return t
    t = t[0].upper() + t[1:]
    if not _ENDS_WITH_PUNCT.search(t):
        t += "."
    return t


# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def get_demo():
    return get_html()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

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
                print(f"[filter] dropped: {repr(text)}")
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
    transcriber = Transcriber(model_path=MODEL_PATH, model_arch=MODEL_ARCH)
    transcriber.add_listener(listener)
    transcriber.start()
    print("Transcriber started for client")

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
                await asyncio.to_thread(transcriber.add_audio, audio.tolist(), 16000)

            elif "text" in message and message["text"]:
                data = json.loads(message["text"])

                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

                elif data.get("type") == "stop":
                    await asyncio.to_thread(transcriber.stop)
                    await asyncio.to_thread(transcriber.start)

                elif data.get("type") == "set_filter_noise":
                    filter_noise_enabled = bool(data.get("enabled", True))
                    print(
                        f"Noise filter {'enabled' if filter_noise_enabled else 'disabled'}"
                    )
                    await websocket.send_json(
                        {"type": "filter_noise_state", "enabled": filter_noise_enabled}
                    )

                elif data.get("type") == "set_auto_copy":
                    auto_copy_enabled = bool(data.get("enabled", False))
                    print(f"Auto-copy {'enabled' if auto_copy_enabled else 'disabled'}")
                    await websocket.send_json(
                        {"type": "auto_copy_state", "enabled": auto_copy_enabled}
                    )

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        drain_task.cancel()
        try:
            transcriber.stop()
        except Exception:
            pass
        transcriber.close()
        print("Transcriber cleaned up")


def get_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Moonshine STT</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #000;
            min-height: 100vh;
            color: #fff;
            padding: 48px 24px;
        }
        .container { max-width: 720px; margin: 0 auto; }
        .header { margin-bottom: 48px; }
        h1 {
            font-size: 1rem;
            font-weight: 500;
            letter-spacing: 0.02em;
            color: #fff;
            margin-bottom: 4px;
        }
        .subtitle { font-size: 0.8rem; color: #555; letter-spacing: 0.01em; }

        .controls {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        button {
            padding: 6px 14px;
            font-size: 0.78rem;
            font-weight: 500;
            border: 1px solid #333;
            border-radius: 0;
            cursor: pointer;
            transition: background 0.1s, color 0.1s, border-color 0.1s;
            letter-spacing: 0.02em;
            background: #000;
            color: #555;
        }
        button:hover:not(:disabled) { background: #fff; color: #000; border-color: #fff; }
        button:disabled { cursor: not-allowed; opacity: 0.3; }
        .btn-start.active { background: #fff; color: #000; border-color: #fff; }
        .btn-copy { margin-left: auto; }
        .btn-copy.copied { color: #fff; border-color: #444; }

        .toggles-row {
            display: flex;
            gap: 20px;
            margin-bottom: 28px;
            flex-wrap: wrap;
        }
        .pill-toggle {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.72rem;
            color: #444;
            letter-spacing: 0.02em;
            cursor: pointer;
            user-select: none;
        }
        .pill-toggle input[type="checkbox"] { display: none; }
        .toggle-track {
            width: 28px; height: 14px;
            background: #222;
            border: 1px solid #333;
            border-radius: 7px;
            position: relative;
            transition: background 0.15s, border-color 0.15s;
            flex-shrink: 0;
        }
        .toggle-track::after {
            content: '';
            position: absolute;
            top: 2px; left: 2px;
            width: 8px; height: 8px;
            background: #555;
            border-radius: 50%;
            transition: transform 0.15s, background 0.15s;
        }
        .pill-toggle input:checked + .toggle-track { background: #fff; border-color: #fff; }
        .pill-toggle input:checked + .toggle-track::after { transform: translateX(14px); background: #000; }
        .toggle-label { color: #444; }
        .pill-toggle input:checked ~ .toggle-label { color: #888; }

        .status-bar {
            display: flex; align-items: center; gap: 8px;
            margin-bottom: 24px;
            font-size: 0.75rem; color: #444;
            letter-spacing: 0.01em; height: 16px;
        }
        .status-bar.listening { color: #888; }
        .status-bar.loading   { color: #666; }
        .pulse-dot { width: 5px; height: 5px; background: #333; flex-shrink: 0; }
        .pulse-dot.active { background: #fff; animation: pulse 1.2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.2; } }

        .transcript-container { border-top: 1px solid #1a1a1a; padding-top: 24px; min-height: 300px; }
        .transcript-header {
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 20px;
        }
        .transcript-label {
            font-size: 0.7rem; color: #333;
            letter-spacing: 0.08em; text-transform: uppercase; font-weight: 500;
        }
        .clear-btn {
            padding: 4px 10px; font-size: 0.7rem;
            border: 1px solid #222; color: #444; background: #000; letter-spacing: 0.02em;
        }
        .clear-btn:hover { background: #fff; color: #000; border-color: #fff; }

        #transcript-text {
            font-size: 1.15rem;
            line-height: 1.85;
            color: #ffffff;
            white-space: pre-wrap;
            word-break: break-word;
            font-weight: 300;
        }
        .partial-span   { color: #3a3a3a; }
        .confirmed-span { color: #ffffff; }
        .empty-hint     { color: #222; font-size: 0.9rem; font-weight: 400; }

        .footer { margin-top: 48px; font-size: 0.7rem; color: #2a2a2a; letter-spacing: 0.02em; }
    </style>
</head>
<body>
<div class="container">

    <div class="header">
        <h1>Moonshine STT</h1>
        <p class="subtitle">on-device &middot; medium streaming</p>
    </div>

    <div class="controls">
        <button id="startBtn" class="btn-start">Record</button>
        <button id="stopBtn" disabled>Stop</button>
        <button id="copyBtn" class="btn-copy" title="Copy transcript to clipboard">Copy</button>
    </div>

    <div class="toggles-row">
        <label class="pill-toggle" title="Drop filler words and hallucinated noise (recommended)">
            <input type="checkbox" id="filterNoiseToggle" checked>
            <span class="toggle-track"></span>
            <span class="toggle-label" id="filterNoiseLabel">filter noise (on)</span>
        </label>
        <label class="pill-toggle" title="Auto-copy full transcript to clipboard after each finalized line">
            <input type="checkbox" id="autoCopyToggle">
            <span class="toggle-track"></span>
            <span class="toggle-label" id="autoCopyLabel">auto-copy</span>
        </label>
    </div>

    <div id="statusBar" class="status-bar idle">
        <div class="pulse-dot" id="pulseDot"></div>
        <span id="statusText">connecting...</span>
    </div>

    <div class="transcript-container">
        <div class="transcript-header">
            <span class="transcript-label">Transcript</span>
            <button class="clear-btn" onclick="clearTranscript()">Clear</button>
        </div>
        <div id="transcript-text"><span class="empty-hint">Waiting for input...</span></div>
    </div>

    <p class="footer">Moonshine Streaming Medium &mdash; VAD built into backend &mdash; <kbd style="font-size:0.65rem;color:#333;border:1px solid #2a2a2a;padding:1px 4px;">R</kbd> toggle recording &nbsp; <kbd style="font-size:0.65rem;color:#333;border:1px solid #2a2a2a;padding:1px 4px;">Esc</kbd> stop</p>

</div>
<script>
let websocket    = null;
let audioContext = null;
let mediaStream  = null;
let workletNode  = null;
let sourceNode   = null;
let isRecording  = false;

let confirmedText = '';
let partialLineId = null;
let partialText   = '';

const startBtn          = document.getElementById('startBtn');
const stopBtn           = document.getElementById('stopBtn');
const copyBtn           = document.getElementById('copyBtn');
const statusBar         = document.getElementById('statusBar');
const statusText        = document.getElementById('statusText');
const pulseDot          = document.getElementById('pulseDot');
const transcriptEl      = document.getElementById('transcript-text');
const filterNoiseToggle = document.getElementById('filterNoiseToggle');
const filterNoiseLabel  = document.getElementById('filterNoiseLabel');
const autoCopyToggle    = document.getElementById('autoCopyToggle');
const autoCopyLabel     = document.getElementById('autoCopyLabel');

filterNoiseToggle.addEventListener('change', () => {
    const enabled = filterNoiseToggle.checked;
    filterNoiseLabel.textContent = enabled ? 'filter noise (on)' : 'filter noise';
    if (websocket && websocket.readyState === WebSocket.OPEN)
        websocket.send(JSON.stringify({ type: 'set_filter_noise', enabled }));
});

autoCopyToggle.addEventListener('change', () => {
    const enabled = autoCopyToggle.checked;
    autoCopyLabel.textContent = enabled ? 'auto-copy (on)' : 'auto-copy';
    if (websocket && websocket.readyState === WebSocket.OPEN)
        websocket.send(JSON.stringify({ type: 'set_auto_copy', enabled }));
});

copyBtn.addEventListener('click', () => {
    const text = (confirmedText + (partialText ? ' ' + partialText : '')).trim();
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
        copyBtn.textContent = 'Copied!';
        copyBtn.classList.add('copied');
        setTimeout(() => { copyBtn.textContent = 'Copy'; copyBtn.classList.remove('copied'); }, 1500);
    });
});

function connectWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    websocket = new WebSocket(protocol + '//' + location.host + '/ws');
    websocket.binaryType = 'arraybuffer';
    websocket.onopen = () => {
        console.log('WS open');
        websocket.send(JSON.stringify({ type: 'set_filter_noise', enabled: filterNoiseToggle.checked }));
        websocket.send(JSON.stringify({ type: 'set_auto_copy',    enabled: autoCopyToggle.checked }));
    };
    websocket.onclose = () => { setStatus('idle', 'disconnected - reload to reconnect'); stopRecording(); };
    websocket.onerror = (e) => console.error('WS error', e);
    websocket.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'ready') {
            setStatus('idle', 'ready');
        } else if (msg.type === 'partial') {
            handlePartial(msg.line_id, msg.text);
        } else if (msg.type === 'transcription') {
            handleTranscription(msg.line_id, msg.text, msg.auto_copy, msg.filtered);
        } else if (msg.type === 'filter_noise_state') {
            filterNoiseToggle.checked = msg.enabled;
            filterNoiseLabel.textContent = msg.enabled ? 'filter noise (on)' : 'filter noise';
        } else if (msg.type === 'auto_copy_state') {
            autoCopyToggle.checked = msg.enabled;
            autoCopyLabel.textContent = msg.enabled ? 'auto-copy (on)' : 'auto-copy';
        }
    };
}

function renderTranscript() {
    if (!confirmedText && !partialText) {
        transcriptEl.innerHTML = '<span class="empty-hint">Your speech will appear here...</span>';
        return;
    }
    let html = '';
    if (confirmedText) html += '<span class="confirmed-span">' + escHtml(confirmedText) + '</span>';
    if (partialText) {
        if (confirmedText) html += ' ';
        html += '<span class="partial-span">' + escHtml(partialText) + '</span>';
    }
    transcriptEl.innerHTML = html;
    transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function handlePartial(lineId, text) {
    partialLineId = lineId;
    partialText   = text;
    renderTranscript();
}

function handleTranscription(lineId, text, autoCopy, filtered) {
    if (partialLineId === lineId) { partialLineId = null; partialText = ''; }
    if (filtered || !text) { renderTranscript(); return; }
    confirmedText = confirmedText ? confirmedText + ' ' + text : text;
    renderTranscript();
    if (autoCopy) navigator.clipboard.writeText(confirmedText).catch(() => {});
}

function clearTranscript() {
    confirmedText = ''; partialLineId = null; partialText = '';
    renderTranscript();
}

function escHtml(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

const WORKLET_CODE = `
class PcmStreamer extends AudioWorkletProcessor {
    constructor(options) {
        super();
        this._targetRate = options.processorOptions.targetRate || 16000;
        this._inputRate  = sampleRate;
        this._ratio      = this._inputRate / this._targetRate;
        this._buf        = [];
        this._CHUNK      = 1600;
    }
    process(inputs) {
        const ch = inputs[0][0];
        if (!ch) return true;
        for (let i = 0; i < ch.length; i += this._ratio)
            this._buf.push(ch[Math.floor(i)]);
        while (this._buf.length >= this._CHUNK) {
            const chunk = new Float32Array(this._CHUNK);
            for (let i = 0; i < this._CHUNK; i++) chunk[i] = this._buf[i];
            this._buf.splice(0, this._CHUNK);
            this.port.postMessage(chunk.buffer, [chunk.buffer]);
        }
        return true;
    }
}
registerProcessor('pcm-streamer', PcmStreamer);
`;

async function startRecording() {
    startBtn.disabled = true;
    setStatus('loading', 'requesting microphone...');
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: {
            sampleRate: { ideal: 16000 }, channelCount: 1,
            echoCancellation: true, noiseSuppression: true,
        }});
        audioContext = new AudioContext();
        const blob = new Blob([WORKLET_CODE], { type: 'application/javascript' });
        const url  = URL.createObjectURL(blob);
        await audioContext.audioWorklet.addModule(url);
        URL.revokeObjectURL(url);
        sourceNode  = audioContext.createMediaStreamSource(mediaStream);
        workletNode = new AudioWorkletNode(audioContext, 'pcm-streamer', {
            processorOptions: { targetRate: 16000 }
        });
        workletNode.port.onmessage = (ev) => {
            if (websocket && websocket.readyState === WebSocket.OPEN)
                websocket.send(ev.data);
        };
        sourceNode.connect(workletNode);
        isRecording = true;
        stopBtn.disabled = false;
        pulseDot.classList.add('active');
        startBtn.classList.add('active');
        setStatus('listening', 'listening');
    } catch (err) {
        console.error(err);
        startBtn.disabled = false;
        startBtn.classList.remove('active');
        setStatus('idle', 'error: ' + err.message);
    }
}

function stopRecording() {
    if (!isRecording) return;
    isRecording = false;
    if (workletNode) { workletNode.disconnect(); workletNode = null; }
    if (sourceNode)  { sourceNode.disconnect();  sourceNode = null; }
    if (audioContext) { audioContext.close(); audioContext = null; }
    if (mediaStream)  { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
    if (websocket && websocket.readyState === WebSocket.OPEN)
        websocket.send(JSON.stringify({ type: 'stop' }));
    startBtn.disabled = false;
    startBtn.classList.remove('active');
    stopBtn.disabled = true;
    pulseDot.classList.remove('active');
    setStatus('idle', 'stopped');
}

function setStatus(cls, text) {
    statusBar.className = 'status-bar ' + cls;
    statusText.textContent = text;
}

startBtn.addEventListener('click', startRecording);
stopBtn.addEventListener('click',  stopRecording);

// Keyboard shortcut: R to toggle recording, Escape to stop
document.addEventListener('keydown', (e) => {
    // Ignore if focus is inside an input/textarea
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === 'r' || e.key === 'R') {
        if (!isRecording) startRecording();
        else stopRecording();
    }
    if (e.key === 'Escape' && isRecording) stopRecording();
});

connectWebSocket();
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
