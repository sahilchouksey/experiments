"""
Shared Parakeet + VAD utilities for Parakeet engines.
"""

from __future__ import annotations

import contextlib
import gc
import threading
import tempfile

import numpy as np
import soundfile as sf
import torch

# ── Lazy model cache by model name ────────────────────────────────────────────
_model_cache: dict[str, object] = {}
_model_device: dict[str, str] = {}
_model_cache_lock = threading.Lock()
_model_infer_locks: dict[str, threading.Lock] = {}

# ── Shared Silero VAD base model ──────────────────────────────────────────────
_vad_model = None
_vad_lock = threading.Lock()


def extract_text(output) -> str:
    if output is None:
        return ""

    if isinstance(output, list):
        if not output:
            return ""
        first = output[0]
        if isinstance(first, str):
            return first.strip()
        text = getattr(first, "text", None)
        if isinstance(text, str):
            return text.strip()
        if isinstance(first, dict):
            for key in ("text", "pred_text", "transcript"):
                val = first.get(key)
                if isinstance(val, str):
                    return val.strip()

    text = getattr(output, "text", None)
    if isinstance(text, str):
        return text.strip()
    return str(output).strip()


def _is_cuda_issue(err: Exception) -> bool:
    msg = str(err).lower()
    return (
        "cuda out of memory" in msg
        or "out of memory" in msg
        and "cuda" in msg
        or "driver" in msg
        and "cuda" in msg
        or "cuda error" in msg
    )


def _load_model_instance(model_name: str, requested_device: str):
    import nemo.collections.asr as nemo_asr

    device = requested_device.lower().strip()
    if device not in {"cuda", "cpu"}:
        device = "cuda"

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    if device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
    else:
        model = model.to("cpu")
        device = "cpu"

    model.eval()
    return model, device


def get_parakeet_model(model_name: str, preferred_device: str):
    with _model_cache_lock:
        if model_name in _model_cache:
            return _model_cache[model_name], _model_device[model_name]

        requested = preferred_device.lower().strip()
        print(f"[parakeet] loading {model_name} on {requested} ...")

        model = None
        try:
            model, loaded_device = _load_model_instance(model_name, requested)
        except Exception as e:
            if requested == "cuda" and _is_cuda_issue(e):
                print(
                    f"[parakeet] CUDA load failed for {model_name} ({e}). "
                    "Falling back to CPU."
                )
                with contextlib.suppress(Exception):
                    del model
                gc.collect()
                with contextlib.suppress(Exception):
                    torch.cuda.empty_cache()
                model, loaded_device = _load_model_instance(model_name, "cpu")
            else:
                raise

        _model_cache[model_name] = model
        _model_infer_locks[model_name] = threading.Lock()
        _model_device[model_name] = loaded_device
        print(f"[parakeet] model loaded: {model_name} ({loaded_device})")
        return model, loaded_device


def transcribe_with_parakeet(model_name: str, model, audio_np: np.ndarray) -> str:
    lock = _model_infer_locks[model_name]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, audio_np, 16000, subtype="PCM_16")
        with lock:
            output = model.transcribe([tmp.name], batch_size=1)
    return extract_text(output)


def get_vad_base():
    global _vad_model
    if _vad_model is None:
        with _vad_lock:
            if _vad_model is None:
                print("[vad] loading silero-vad for parakeet engines ...")
                from silero_vad import load_silero_vad

                _vad_model = load_silero_vad(onnx=False)
                print("[vad] loaded for parakeet engines")
    return _vad_model
