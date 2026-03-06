# OpenCode SDK Selection Demo

Minimal web demo for OpenCode Python SDK with:

- Prompt input
- Model dropdown (GitHub Copilot models only)
- Submit button that stays disabled until turn completion
- Live thinking stream
- Live response stream
- Live tool-call stream (including MCP tools)

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

Open `http://localhost:8020`.

## Notes

- By default the app targets `http://127.0.0.1:4096` for OpenCode server.
- If no server is running there, the app attempts to start `opencode serve` locally.
- You can override with `OPENCODE_BASE_URL`, for example:

```bash
OPENCODE_BASE_URL=http://127.0.0.1:4096 python3 main.py
```
