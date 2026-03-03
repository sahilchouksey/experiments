from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import app.engines  # noqa: F401
from app.policy import JsonRoutingPolicy
from app.service import PredictionService

BASE_DIR = Path(__file__).parent
POLICY_PATH = BASE_DIR.parent / "copilot-routing-policy.json"

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

policy = JsonRoutingPolicy(POLICY_PATH)
service = PredictionService(policy)


class PredictRequest(BaseModel):
    router_id: str
    text: str


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/routers")
async def list_routers():
    routers = service.list_routers()
    return {
        "routers": [
            {
                "id": r.id,
                "label": r.label,
                "implemented": r.implemented,
                "notes": r.notes,
            }
            for r in routers
        ]
    }


@app.post("/api/predict")
async def predict(req: PredictRequest):
    try:
        result = await service.predict(req.router_id, req.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {exc}"
        ) from exc

    return {
        "router": {"id": result.router_id, "label": result.router_label},
        "classification": {
            "complexity": result.classification.complexity.value,
            "task": result.classification.task.value,
            "confidence": result.classification.confidence,
            "latency_ms": result.classification.latency_ms,
            "raw_response": result.classification.raw_response,
        },
        "decision": {
            "tier": result.decision.tier.value,
            "selected_model": result.decision.selected_model,
            "fallback_models": result.decision.fallback_models,
            "multiplier": None
            if result.decision.multiplier is None
            else {
                "paid_plan": result.decision.multiplier.paid_plan,
                "copilot_free": result.decision.multiplier.copilot_free,
                "paid_plan_auto_model_selection": result.decision.multiplier.paid_plan_auto_model_selection,
            },
        },
    }


@app.get("/api/routers/{router_id}/status")
async def router_status(router_id: str):
    try:
        return await service.get_router_status(router_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/routers/{router_id}/warmup")
async def warmup_router(router_id: str):
    try:
        await service.warmup_router(router_id)
        status = await service.get_router_status(router_id)
        return {"ok": True, "status": status}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Warmup failed: {exc}") from exc


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)
