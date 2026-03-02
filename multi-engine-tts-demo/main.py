"""
Real-time TTS demo — Kokoro ONNX + Qwen3-TTS.
Thin orchestrator: reads engine descriptors from engines/__init__.py,
assembles the HTML from static/index.html, and mounts WebSocket routes.

Start:
    cd /home/xix3r/tts-demo && source venv/bin/activate && python main.py
"""

import json
from functools import partial
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from engines import ENGINES

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent

# Model paths — resolved once at startup and passed into each engine handler.
_MODEL_PATHS = {
    "kokoro": {
        "model_path": BASE_DIR / "models" / "kokoro-v1.0.onnx",
        "voices_path": BASE_DIR / "models" / "voices-v1.0.bin",
    },
    "qwen": {
        "model_path": (
            BASE_DIR
            / "models"
            / "qwen3-tts"
            / "models--Qwen--Qwen3-TTS-12Hz-0.6B-Base"
            / "snapshots"
            / "5d83992436eae1d760afd27aff78a71d676296fc"
        ),
    },
}


def _get_paths(engine) -> dict:
    """Return the model-path kwargs for a given engine module."""
    return _MODEL_PATHS.get(engine.TAB_ID, {})


# ---------------------------------------------------------------------------
# HTML assembly helpers
# ---------------------------------------------------------------------------


def _build_tab_button(engine, first: bool) -> str:
    active = " active" if first else ""
    return (
        f'  <button class="tab-btn{active}" data-tab="{engine.TAB_ID}">'
        f"{engine.LABEL}"
        f"</button>"
    )


def _build_control_html(ctrl: dict) -> str:
    """Render a single control dict as HTML."""
    label = f'<span class="ctrl-label">{ctrl["label"]}</span>'

    if ctrl["type"] == "select":
        options_html = "".join(
            f'<option value="{o["value"]}">{o["label"]}</option>'
            for o in ctrl.get("options", [])
        )
        sel = (
            f'<select id="{ctrl["id"]}" title="{ctrl["label"]}">{options_html}</select>'
        )
        return f'<div class="ctrl-group">{label}{sel}</div>'

    if ctrl["type"] == "range":
        val_id = ctrl["id"] + "-val"
        rng = (
            f'<input type="range" id="{ctrl["id"]}"'
            f' min="{ctrl["min"]}" max="{ctrl["max"]}"'
            f' step="{ctrl["step"]}" value="{ctrl["value"]}">'
        )
        val_span = (
            f'<span class="speed-val" id="{val_id}">'
            f"{ctrl['value']}{ctrl.get('unit', '')}"
            f"</span>"
        )
        return (
            f'<div class="ctrl-group">'
            f"{label}"
            f'<div class="range-wrap">{rng}{val_span}</div>'
            f"</div>"
        )

    return ""


def _build_panel(engine, first: bool) -> str:
    active = " active" if first else ""
    tid = engine.TAB_ID

    # Controls row (may be empty)
    ctrl_html = ""
    if engine.CONTROLS:
        inner = "\n      ".join(_build_control_html(c) for c in engine.CONTROLS)
        ctrl_html = f'\n    <div class="ctrl-row">\n      {inner}\n    </div>'

    return f"""\
  <div class="tab-panel{active}" id="panel-{tid}">
    <textarea id="{tid}-text" placeholder="Type or paste text here\u2026"
              spellcheck="false"{" autofocus" if first else ""}></textarea>{ctrl_html}
    <div class="actions">
      <button class="btn-speak" id="{tid}-btn-speak">Speak</button>
      <button class="btn-stop"  id="{tid}-btn-stop">&#9632; Stop</button>
    </div>
    <hr class="divider">
    <div class="status-bar">
      <div class="dot ready" id="{tid}-dot"></div>
      <span class="status-text" id="{tid}-status">Ready</span>
    </div>
    <div class="chunks-row" id="{tid}-chunks"></div>
    <div class="latency-row" id="{tid}-latency-row">
      <span class="latency-label">first audio</span>
      <span class="latency-val" id="{tid}-latency"></span>
    </div>
    <div class="waveform-wrap" id="{tid}-waveform-wrap">
      <canvas id="{tid}-canvas"></canvas>
    </div>
  </div>"""


def _assemble_html(template: str) -> str:
    """Inject engine-specific fragments into the HTML template."""
    # Tab buttons
    tab_buttons = "\n".join(
        _build_tab_button(eng, i == 0) for i, eng in enumerate(ENGINES)
    )

    # Tab panels
    panels = "\n".join(_build_panel(eng, i == 0) for i, eng in enumerate(ENGINES))

    # JS registry array  [{ tabId: "kokoro", wsPath: "/ws/tts/kokoro" }, ...]
    registry_items = ", ".join(
        f'{{ tabId: "{eng.TAB_ID}", wsPath: "{eng.WS_PATH}" }}' for eng in ENGINES
    )
    registry_js = f"[{registry_items}]"

    # Per-engine extra JS — injected INLINE before boot() so engineInit_* and
    # buildPayload_* functions are defined before boot() calls them.
    extra_js_inline = "\n\n".join(eng.EXTRA_JS for eng in ENGINES)

    html = template
    html = html.replace("  <!-- ENGINE_TABS -->", tab_buttons)
    html = html.replace("  <!-- ENGINE_PANELS -->", panels)
    html = html.replace("ENGINE_REGISTRY", registry_js)
    html = html.replace("// ENGINE_EXTRA_JS_INLINE", extra_js_inline)
    return html


# ---------------------------------------------------------------------------
# Load and assemble the HTML once at startup
# ---------------------------------------------------------------------------
_TEMPLATE = (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")
_HTML = _assemble_html(_TEMPLATE)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI()

# Serve any other static assets (if added later)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return _HTML


# ---------------------------------------------------------------------------
# Register one WebSocket route per engine
# ---------------------------------------------------------------------------
for _engine in ENGINES:
    _paths = _get_paths(_engine)

    # Use partial to bind the correct paths into the handler closure.
    # We need a factory to avoid late-binding issues in the loop.
    def _make_handler(eng, paths):
        async def _handler(websocket: WebSocket):
            await eng.ws_handler(websocket, **paths)

        return _handler

    app.add_api_websocket_route(
        _engine.WS_PATH,
        _make_handler(_engine, _paths),
    )


# ---------------------------------------------------------------------------
# Backwards-compat alias: /ws/tts → Kokoro
# ---------------------------------------------------------------------------
_kokoro_engine = next(e for e in ENGINES if e.TAB_ID == "kokoro")
_kokoro_paths = _get_paths(_kokoro_engine)


async def _ws_tts_compat(websocket: WebSocket):
    """Legacy /ws/tts → Kokoro."""
    await _kokoro_engine.ws_handler(websocket, **_kokoro_paths)


app.add_api_websocket_route("/ws/tts", _ws_tts_compat)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print(
        "  TTS Demo — Kokoro 82M (ONNX) + Qwen3-TTS 0.6B + Chatterbox Turbo 350M (GPU)"
    )
    print("  http://localhost:8000")
    print("=" * 60)

    missing = []
    for eng in ENGINES:
        for key, path in _get_paths(eng).items():
            if not path.exists():
                missing.append(str(path))

    if missing:
        for m in missing:
            print(f"\n[ERROR] Not found: {m}")
        raise SystemExit(1)

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
