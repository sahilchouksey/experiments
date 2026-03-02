"""
Dual-Engine STT Demo — thin orchestrator.

Imports each engine module for its side-effect (self-registration into
REGISTRY), then mounts every registered engine onto the FastAPI app.
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Side-effect imports: each module calls @register(...) which populates REGISTRY
import engines.moonshine_engine  # noqa: F401
import engines.whisper_engine  # noqa: F401

from engines.base import REGISTRY

app = FastAPI()

# Mount all registered engines
for engine_cls in REGISTRY.values():
    engine_cls().mount(app)

# Serve static files (CSS, JS, etc.) if present
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
