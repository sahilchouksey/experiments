"""
Engine registry — plug-and-play TTS backends.

To add a new engine:
  1. Create engines/myengine.py with the same interface as kokoro.py / qwen.py.
  2. Import it here and append to ENGINES list.
  3. Add its model-path resolution in main.py inside `_resolve_paths`.

Each engine module must expose:
  LABEL    : str          — display name for the tab (e.g. "Kokoro 82M")
  SUBTITLE : str          — one-liner shown under the tab name
  TAB_ID   : str          — unique slug used for DOM ids and route suffix
  WS_PATH  : str          — WebSocket route, e.g. "/ws/tts/kokoro"
  CONTROLS : list[dict]   — per-engine UI controls (select / range)
  EXTRA_JS : str          — JS snippet; must define engineInit_<TAB_ID>(cfg)
                            and buildPayload_<TAB_ID>(text)
  ws_handler : coroutine  — signature varies per engine, called by main.py
"""

from engines import kokoro, qwen, chatterbox

# ---------------------------------------------------------------------------
# Ordered list of active engines.
# Reorder or comment-out entries to change tab order / availability.
# ---------------------------------------------------------------------------
ENGINES: list = [kokoro, qwen, chatterbox]
