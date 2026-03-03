# LLM Router Workflow Demo

Demo video: [`demo.mp4`](./demo.mp4)

Minimal workflow predictor UI + router backend using:

- Registry + Strategy + Ports/Adapters
- Policy source: `../copilot-routing-policy.json`
- Router tabs for active router engines

Current status:

- Implemented: `arch`, `routellm`, `vllm-semantic`

Router notes:

- `routellm`: uses the RouteLLM BERT checkpoint (`routellm/bert_gpt4_augmented`) for win-rate scoring.
- `vllm-semantic`: strict native backend mode. Requires a live Semantic Router API (`VLLM_SR_BASE_URL`, default `http://127.0.0.1:8080`).
- Non-native fallback paths are disabled for `vllm-semantic`. If backend is unavailable, status surfaces an error and prediction is blocked.

## Important: Run Router Backends Separately

- `vllm-semantic` is not started by `python main.py`; you must run `vllm-sr serve` as a separate service.

Example setup:

```bash
# Terminal 1: vLLM Semantic Router backend
cd /home/xix3r/Documents/fun/experiments/llm-router-workflow-demo
source .venv/bin/activate
pip install vllm-sr
vllm-sr init          # first time only
vllm-sr serve

# Terminal 2: workflow demo API/UI
cd /home/xix3r/Documents/fun/experiments/llm-router-workflow-demo
source .venv/bin/activate
export VLLM_SR_BASE_URL=http://127.0.0.1:8080
python main.py
```

Note: with `vllm-sr`, use port `8080` for router/classification API (`/health`, `/api/v1/classify/intent`).
Port `8888` is the external inference listener and may return upstream model errors if your model backend is not running.

## Full Setup (Fresh Machine)

```bash
cd /home/xix3r/Documents/fun/experiments/llm-router-workflow-demo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install vllm-sr
vllm-sr init
```

Then run in two terminals:

```bash
# Terminal 1
cd /home/xix3r/Documents/fun/experiments/llm-router-workflow-demo
source .venv/bin/activate
vllm-sr serve

# Terminal 2
cd /home/xix3r/Documents/fun/experiments/llm-router-workflow-demo
source .venv/bin/activate
export VLLM_SR_BASE_URL=http://127.0.0.1:8080
python main.py
```

Quick checks:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8010/api/routers/vllm-semantic/status
```

Expected `vllm-semantic` status includes `"ready": true`.

## Troubleshooting

- If `vllm-sr` fails with `/usr/bin/docker: read-only file system`, run:

  ```bash
  VLLM_SR_DOCKER_BIN=/no-such-binary vllm-sr serve --minimal
  ```

- First startup can take several minutes while vLLM Semantic Router downloads models.
- If UI still shows old router behavior, restart `python main.py` and hard refresh the browser.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Open: `http://localhost:8010`

## API

- `GET /api/routers`
- `POST /api/predict`

Example predict request:

```json
{
  "router_id": "arch",
  "text": "Design a scalable event-driven architecture for notifications"
}
```
