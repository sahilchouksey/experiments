from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from opencode_ai import AsyncOpencode


BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

DEFAULT_BASE_URL = "http://127.0.0.1:4096"
OPENCODE_BASE_URL = os.environ.get("OPENCODE_BASE_URL", DEFAULT_BASE_URL).rstrip("/")

CONFIG_PATH = Path.home() / ".config" / "opencode" / "opencode.json"
AUTH_PATH = Path.home() / ".local" / "share" / "opencode" / "auth.json"

MCP_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitize_mcp_prefix(name: str) -> str:
    return MCP_SANITIZE_RE.sub("_", name)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _local_auth_diagnostics() -> dict[str, Any]:
    out: dict[str, Any] = {
        "config_found": CONFIG_PATH.exists(),
        "auth_found": AUTH_PATH.exists(),
        "github_copilot_auth_found": False,
    }
    if not AUTH_PATH.exists():
        return out

    try:
        auth_data = json.loads(AUTH_PATH.read_text(encoding="utf-8"))
    except Exception:
        return out

    copilot = auth_data.get("github-copilot")
    if not isinstance(copilot, dict):
        return out

    has_access = bool(copilot.get("access"))
    has_refresh = bool(copilot.get("refresh"))
    out["github_copilot_auth_found"] = bool(has_access or has_refresh)
    return out


@dataclass
class OpenCodeServerManager:
    base_url: str
    process: subprocess.Popen[Any] | None = None

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        self.scheme = parsed.scheme or "http"
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or (443 if self.scheme == "https" else 80)
        self.can_spawn_local = self.host in {"127.0.0.1", "localhost"}

    async def is_healthy(self) -> bool:
        async with httpx.AsyncClient(timeout=2.0) as client:
            # /health can hang on some opencode builds, so use a practical endpoint
            # that must return quickly when the server is actually ready.
            try:
                resp = await client.get(f"{self.base_url}/config/providers")
                if resp.status_code == 200:
                    return True
            except Exception:
                pass

            try:
                resp = await client.get(f"{self.base_url}/config")
                return resp.status_code == 200
            except Exception:
                return False

    async def ensure_running(self) -> None:
        if await self.is_healthy():
            return

        if not self.can_spawn_local:
            raise RuntimeError(
                "OpenCode server is not reachable and base URL is non-local. "
                "Set OPENCODE_BASE_URL to a reachable server."
            )

        cmd = [
            "opencode",
            "serve",
            "--hostname",
            self.host,
            "--port",
            str(self.port),
        ]
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        for _ in range(300):
            if await self.is_healthy():
                return
            await asyncio.sleep(0.1)

        await self.stop()
        raise RuntimeError("Failed to start OpenCode server for the demo.")

    async def stop(self) -> None:
        if not self.process:
            return
        if self.process.poll() is not None:
            self.process = None
            return

        self.process.terminate()
        try:
            await asyncio.to_thread(self.process.wait, 3)
        except Exception:
            self.process.kill()
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self.process.wait, 2)
        finally:
            self.process = None


SERVER = OpenCodeServerManager(OPENCODE_BASE_URL)


async def _fetch_mcp_prefixes() -> list[str]:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{OPENCODE_BASE_URL}/mcp")
            if resp.status_code != 200:
                return []
            payload = resp.json()
            if not isinstance(payload, dict):
                return []
            return [_sanitize_mcp_prefix(name) for name in payload.keys()]
    except Exception:
        return []


def _is_mcp_tool(tool_name: str, mcp_prefixes: list[str]) -> bool:
    return any(tool_name.startswith(prefix + "_") for prefix in mcp_prefixes)


async def _iter_sse_payloads(response: Any, stop_event: asyncio.Event):
    data_lines: list[str] = []
    async for line in response.iter_lines():
        if stop_event.is_set():
            break

        if line == "":
            if not data_lines:
                continue
            raw = "\n".join(data_lines)
            data_lines.clear()
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue
            continue

        if line.startswith(":"):
            continue

        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())


async def _list_github_copilot_models(
    client: AsyncOpencode,
) -> tuple[list[dict[str, Any]], str | None]:
    providers = await client.app.providers()
    provider = next((p for p in providers.providers if p.id == "github-copilot"), None)
    if not provider:
        return [], None

    models = sorted(provider.models.values(), key=lambda m: (m.name or m.id).lower())
    default_model = providers.default.get("github-copilot")
    out = [
        {
            "id": model.id,
            "name": model.name,
            "reasoning": model.reasoning,
            "tool_call": model.tool_call,
        }
        for model in models
    ]
    return out, default_model


async def _list_sessions(
    client: AsyncOpencode,
    limit: int = 5,
) -> list[dict[str, Any]]:
    # SDK typed model drops some fields on this endpoint in current alpha.
    # Use raw JSON to reliably capture session IDs/timestamps.
    response = await client.session.with_raw_response.list()
    payload = await response.json()

    if not isinstance(payload, list):
        return []

    normalized: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        session_id = row.get("id")
        if not isinstance(session_id, str) or not session_id:
            continue
        time_info = row.get("time") if isinstance(row.get("time"), dict) else {}
        time_updated = time_info.get("updated") if isinstance(time_info, dict) else None
        time_created = time_info.get("created") if isinstance(time_info, dict) else None
        updated = time_updated if isinstance(time_updated, (int, float)) else 0
        created = time_created if isinstance(time_created, (int, float)) else 0
        title = row.get("title") if isinstance(row.get("title"), str) else ""
        normalized.append(
            {
                "id": session_id,
                "title": title,
                "updated": updated,
                "created": created,
            }
        )

    ordered = sorted(normalized, key=lambda s: s["updated"], reverse=True)
    return ordered[:limit]


async def _stream_turn(
    websocket: WebSocket,
    client: AsyncOpencode,
    prompt: str,
    model_id: str,
    selected_session_id: str | None,
    mcp_prefixes: list[str],
) -> str:
    active_session_id = selected_session_id
    created_new = False

    if not active_session_id:
        # SDK quirk: POST /session requires an explicit JSON body even if empty.
        session = await client.session.create(extra_body={})
        active_session_id = session.id
        created_new = True

    await websocket.send_json(
        {
            "type": "turn_started",
            "session_id": active_session_id,
            "created_new": created_new,
            "provider_id": "github-copilot",
            "model_id": model_id,
        }
    )
    await websocket.send_json({"type": "phase", "value": "thinking"})

    stop_stream = asyncio.Event()
    idle_event = asyncio.Event()
    first_generation_seen = False

    part_types: dict[str, str] = {}
    reasoning_text: dict[str, str] = {}
    response_text: dict[str, str] = {}
    final_usage: dict[str, Any] = {}

    async def on_event(payload: dict[str, Any]) -> None:
        nonlocal first_generation_seen, final_usage

        envelope = payload
        maybe_inner = envelope.get("payload")
        inner: dict[str, Any] = (
            maybe_inner if isinstance(maybe_inner, dict) else envelope
        )

        event_type = inner.get("type")
        props = inner.get("properties") or {}

        if event_type in {"server.connected", "server.heartbeat"}:
            return

        await websocket.send_json({"type": "trace", "event_type": event_type})

        if event_type == "session.idle":
            if props.get("sessionID") == active_session_id:
                idle_event.set()
            return

        if event_type == "session.error":
            if props.get("sessionID") == active_session_id:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Session error received from OpenCode server.",
                    }
                )
                idle_event.set()
            return

        if event_type == "message.part.updated":
            part = props.get("part") or {}
            if part.get("sessionID") != active_session_id:
                return

            part_id = part.get("id")
            part_type = part.get("type")
            if isinstance(part_id, str) and isinstance(part_type, str):
                part_types[part_id] = part_type

            if part_type == "reasoning":
                text = part.get("text") or ""
                if isinstance(part_id, str):
                    reasoning_text[part_id] = text
                await websocket.send_json(
                    {
                        "type": "reasoning_update",
                        "part_id": part_id,
                        "text": text,
                    }
                )
                return

            if part_type == "text":
                text = part.get("text") or ""
                if isinstance(part_id, str):
                    response_text[part_id] = text
                if text and not first_generation_seen:
                    first_generation_seen = True
                    await websocket.send_json({"type": "phase", "value": "generating"})
                await websocket.send_json(
                    {
                        "type": "response_update",
                        "part_id": part_id,
                        "text": text,
                    }
                )
                return

            if part_type == "tool":
                state = part.get("state") or {}
                tool_name = part.get("tool") or "unknown"
                await websocket.send_json(
                    {
                        "type": "tool_update",
                        "call_id": part.get("callID") or part_id,
                        "tool": tool_name,
                        "status": state.get("status") or "pending",
                        "input": state.get("input"),
                        "metadata": state.get("metadata"),
                        "output": state.get("output"),
                        "error": state.get("error"),
                        "is_mcp": _is_mcp_tool(tool_name, mcp_prefixes),
                    }
                )
                return

            if part_type == "step-finish":
                final_usage = {
                    "tokens": part.get("tokens") or {},
                    "cost": part.get("cost") or 0,
                }
                await websocket.send_json(
                    {
                        "type": "usage",
                        "tokens": part.get("tokens") or {},
                        "cost": part.get("cost") or 0,
                    }
                )
                return

        if event_type == "message.part.delta":
            if props.get("sessionID") != active_session_id:
                return

            part_id = props.get("partID")
            if not isinstance(part_id, str):
                return

            field = props.get("field")
            delta = props.get("delta") or ""
            if field != "text" or not isinstance(delta, str):
                return

            part_type = part_types.get(part_id)
            if part_type == "reasoning":
                reasoning_text[part_id] = reasoning_text.get(part_id, "") + delta
                await websocket.send_json(
                    {
                        "type": "reasoning_delta",
                        "part_id": part_id,
                        "delta": delta,
                    }
                )
                return

            response_text[part_id] = response_text.get(part_id, "") + delta
            if delta and not first_generation_seen:
                first_generation_seen = True
                await websocket.send_json({"type": "phase", "value": "generating"})
            await websocket.send_json(
                {
                    "type": "response_delta",
                    "part_id": part_id,
                    "delta": delta,
                }
            )

    async def listen_events() -> None:
        try:
            async with client.event.with_streaming_response.list(
                timeout=None
            ) as response:
                async for payload in _iter_sse_payloads(response, stop_stream):
                    if stop_stream.is_set():
                        break
                    if isinstance(payload, dict):
                        await on_event(payload)
        except Exception:
            if not stop_stream.is_set():
                await websocket.send_json(
                    {
                        "type": "trace",
                        "event_type": "event.stream.error",
                    }
                )

    listener = asyncio.create_task(listen_events())
    chat_result: Any | None = None

    try:
        chat_result = await client.session.chat(
            id=active_session_id,
            model_id=model_id,
            provider_id="github-copilot",
            parts=[{"type": "text", "text": prompt}],
            tools={"*": True},
        )
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(idle_event.wait(), timeout=4)
    finally:
        stop_stream.set()
        listener.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await listener

    if chat_result is not None:
        model_extra = _as_dict(
            chat_result.model_extra if hasattr(chat_result, "model_extra") else {}
        )
        info = _as_dict(model_extra.get("info"))
        tokens = _as_dict(info.get("tokens"))
        cache = _as_dict(tokens.get("cache"))

        await websocket.send_json(
            {
                "type": "assistant_summary",
                "provider_id": chat_result.provider_id
                or info.get("providerID")
                or "github-copilot",
                "api_model_id": chat_result.api_model_id or info.get("modelID"),
                "cost": chat_result.cost
                if chat_result.cost is not None
                else info.get("cost", 0),
                "tokens": {
                    "input": chat_result.tokens.input
                    if chat_result.tokens
                    else tokens.get("input", 0),
                    "output": chat_result.tokens.output
                    if chat_result.tokens
                    else tokens.get("output", 0),
                    "reasoning": chat_result.tokens.reasoning
                    if chat_result.tokens
                    else tokens.get("reasoning", 0),
                    "cache_read": chat_result.tokens.cache.read
                    if chat_result.tokens
                    else cache.get("read", 0),
                    "cache_write": chat_result.tokens.cache.write
                    if chat_result.tokens
                    else cache.get("write", 0),
                },
            }
        )
    else:
        tokens = _as_dict(final_usage.get("tokens"))
        cache = _as_dict(tokens.get("cache"))
        await websocket.send_json(
            {
                "type": "assistant_summary",
                "provider_id": "github-copilot",
                "api_model_id": model_id,
                "cost": final_usage.get("cost", 0),
                "tokens": {
                    "input": tokens.get("input", 0),
                    "output": tokens.get("output", 0),
                    "reasoning": tokens.get("reasoning", 0),
                    "cache_read": cache.get("read", 0),
                    "cache_write": cache.get("write", 0),
                },
            }
        )

    await websocket.send_json({"type": "phase", "value": "done"})
    await websocket.send_json({"type": "turn_complete"})
    return active_session_id


async def _get_models_payload() -> dict[str, Any]:
    await SERVER.ensure_running()
    auth_info = _local_auth_diagnostics()
    async with AsyncOpencode(base_url=OPENCODE_BASE_URL) as client:
        models, default_model = await _list_github_copilot_models(client)
        sessions = await _list_sessions(client)
    return {
        "provider": "github-copilot",
        "base_url": OPENCODE_BASE_URL,
        "models": models,
        "default_model": default_model,
        "sessions": sessions,
        "current_session_id": None,
        "diagnostics": auth_info,
    }


async def _lifespan_startup() -> None:
    await SERVER.ensure_running()


async def _lifespan_shutdown() -> None:
    await SERVER.stop()


app = FastAPI(on_startup=[_lifespan_startup], on_shutdown=[_lifespan_shutdown])
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/models")
async def models() -> dict[str, Any]:
    return await _get_models_payload()


@app.websocket("/ws/stream")
async def stream(websocket: WebSocket) -> None:
    await websocket.accept()
    await SERVER.ensure_running()

    async with AsyncOpencode(base_url=OPENCODE_BASE_URL) as client:
        models, default_model = await _list_github_copilot_models(client)
        sessions = await _list_sessions(client)
        auth_info = _local_auth_diagnostics()
        mcp_prefixes = await _fetch_mcp_prefixes()

        await websocket.send_json(
            {
                "type": "ready",
                "provider": "github-copilot",
                "models": models,
                "default_model": default_model,
                "sessions": sessions,
                "current_session_id": None,
                "diagnostics": auth_info,
                "base_url": OPENCODE_BASE_URL,
            }
        )

        try:
            while True:
                message = await websocket.receive_json()
                msg_type = message.get("type")

                if msg_type == "refresh_models":
                    (
                        refreshed_models,
                        refreshed_default,
                    ) = await _list_github_copilot_models(client)
                    refreshed_sessions = await _list_sessions(client)
                    await websocket.send_json(
                        {
                            "type": "ready",
                            "provider": "github-copilot",
                            "models": refreshed_models,
                            "default_model": refreshed_default,
                            "sessions": refreshed_sessions,
                            "current_session_id": None,
                            "diagnostics": auth_info,
                            "base_url": OPENCODE_BASE_URL,
                        }
                    )
                    continue

                if msg_type != "submit":
                    await websocket.send_json(
                        {"type": "error", "message": "Unknown message type."}
                    )
                    continue

                prompt = str(message.get("prompt") or "").strip()
                model_id = str(message.get("model_id") or "").strip()
                selected_session_id = str(message.get("session_id") or "").strip()
                if selected_session_id.lower() in {"new", "__new__"}:
                    selected_session_id = ""

                if not prompt:
                    await websocket.send_json(
                        {"type": "error", "message": "Prompt cannot be empty."}
                    )
                    await websocket.send_json({"type": "turn_complete"})
                    continue

                if not model_id:
                    await websocket.send_json(
                        {"type": "error", "message": "Select a model first."}
                    )
                    await websocket.send_json({"type": "turn_complete"})
                    continue

                try:
                    active_session_id = await _stream_turn(
                        websocket,
                        client,
                        prompt,
                        model_id,
                        selected_session_id or None,
                        mcp_prefixes,
                    )
                    sessions = await _list_sessions(client)
                    await websocket.send_json(
                        {
                            "type": "sessions_update",
                            "sessions": sessions,
                            "current_session_id": active_session_id,
                        }
                    )
                except WebSocketDisconnect:
                    return
                except Exception:
                    try:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "Failed to complete this turn. Check OpenCode server and auth state.",
                            }
                        )
                        await websocket.send_json({"type": "phase", "value": "error"})
                        await websocket.send_json({"type": "turn_complete"})
                    except WebSocketDisconnect:
                        return

        except WebSocketDisconnect:
            return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8020)
