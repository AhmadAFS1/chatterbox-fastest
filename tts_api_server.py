#!/usr/bin/env python3

import hashlib
import io
import os
import random
import re
import tempfile
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

import numpy as np
import torch
import torchaudio as ta
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from chatterbox_vllm.text_utils import SUPPORTED_LANGUAGES
from chatterbox_vllm.tts import ChatterboxTTS


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value is not None else default


def split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= 1:
        return [text.strip()]
    return sentences


def maybe_split_text(text: str, split_sentences: bool) -> list[str]:
    text = text.strip()
    if not split_sentences:
        return [text]
    return split_into_sentences(text)


def set_seed(seed: int) -> int:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


MODEL_VARIANT = os.environ.get("CHATTERBOX_MODEL_VARIANT", "multilingual").strip().lower()
MAX_BATCH_SIZE = _env_int("CHATTERBOX_MAX_BATCH_SIZE", 10)
MAX_MODEL_LEN = _env_int("CHATTERBOX_MAX_MODEL_LEN", 500)
GPU_MEMORY_UTILIZATION = _env_float("CHATTERBOX_GPU_MEMORY_UTILIZATION", 0.5)
ENABLE_SPLIT_SENTENCES_DEFAULT = _env_bool("CHATTERBOX_SPLIT_SENTENCES_DEFAULT", True)
COMPILE = _env_bool("CHATTERBOX_COMPILE", False)
CONDS_CACHE_SIZE = _env_int("CHATTERBOX_CONDS_CACHE_SIZE", 32)

if MODEL_VARIANT not in {"english", "multilingual"}:
    raise ValueError("CHATTERBOX_MODEL_VARIANT must be either 'english' or 'multilingual'.")

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

app = FastAPI(title="Chatterbox TTS API", version="0.1.0")

global_model: Optional[ChatterboxTTS] = None
cond_cache: OrderedDict[str, tuple[dict[str, Any], torch.Tensor]] = OrderedDict()
cond_cache_lock = threading.Lock()
generation_lock = threading.Lock()


def load_model() -> ChatterboxTTS:
    global global_model
    if global_model is not None:
        return global_model

    print(f"Loading model variant={MODEL_VARIANT}...")
    if MODEL_VARIANT == "multilingual":
        global_model = ChatterboxTTS.from_pretrained_multilingual(
            max_batch_size=MAX_BATCH_SIZE,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            compile=COMPILE,
        )
    else:
        global_model = ChatterboxTTS.from_pretrained(
            max_batch_size=MAX_BATCH_SIZE,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            compile=COMPILE,
        )
    return global_model


@app.on_event("startup")
def _startup() -> None:
    load_model()


def _get_conds_from_uploaded_audio(audio_bytes: bytes, filename: Optional[str]) -> tuple[dict[str, Any], torch.Tensor]:
    if global_model is None:
        raise RuntimeError("Model is not loaded.")

    digest = hashlib.sha256(audio_bytes).hexdigest()

    with cond_cache_lock:
        cached = cond_cache.get(digest)
        if cached is not None:
            cond_cache.move_to_end(digest)
            return cached

    suffix = ""
    if filename:
        _, ext = os.path.splitext(filename)
        suffix = ext

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        s3gen_ref, cond_emb = global_model.get_audio_conditionals(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    with cond_cache_lock:
        cond_cache[digest] = (s3gen_ref, cond_emb)
        cond_cache.move_to_end(digest)
        while len(cond_cache) > CONDS_CACHE_SIZE:
            cond_cache.popitem(last=False)

    return s3gen_ref, cond_emb


def _get_conds(audio_prompt: Optional[UploadFile]) -> tuple[dict[str, Any], torch.Tensor]:
    if global_model is None:
        raise RuntimeError("Model is not loaded.")

    if audio_prompt is None:
        return global_model.get_audio_conditionals(None)

    audio_bytes = audio_prompt.file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio_prompt is empty.")
    return _get_conds_from_uploaded_audio(audio_bytes, audio_prompt.filename)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/languages")
def languages() -> dict[str, str]:
    if MODEL_VARIANT == "multilingual":
        return SUPPORTED_LANGUAGES
    return {"en": "English"}


@app.post("/v1/tts")
def tts(
    text: str = Form(...),
    audio_prompt: Optional[UploadFile] = File(default=None),
    language_id: str = Form("en"),
    split_sentences: bool = Form(ENABLE_SPLIT_SENTENCES_DEFAULT),
    exaggeration: float = Form(0.5),
    temperature: float = Form(0.8),
    diffusion_steps: int = Form(4),
    min_p: float = Form(0.05),
    top_p: float = Form(1.0),
    repetition_penalty: float = Form(1.2),
    seed: int = Form(0),
) -> StreamingResponse:
    model = load_model()

    if not text.strip():
        raise HTTPException(status_code=400, detail="`text` must not be empty.")

    if model.variant != "multilingual":
        language_id = "en"
    elif language_id.lower() not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported `language_id`: {language_id}")

    if seed != 0:
        seed_value = set_seed(int(seed))
    else:
        seed_value = None

    s3gen_ref, cond_emb = _get_conds(audio_prompt)
    cond_emb = model.update_exaggeration(cond_emb, exaggeration=exaggeration)
    prompts = maybe_split_text(text, split_sentences=split_sentences)

    t0 = time.perf_counter()
    with generation_lock:
        wavs = model.generate_with_conds(
            prompts,
            s3gen_ref=s3gen_ref,
            cond_emb=cond_emb,
            language_id=language_id.lower(),
            temperature=temperature,
            diffusion_steps=diffusion_steps,
            min_p=min_p,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed_value,
            clear_cuda_cache=False,
        )

    combined = torch.cat([w.squeeze(0) for w in wavs], dim=-1)
    elapsed = time.perf_counter() - t0
    audio_seconds = combined.shape[-1] / model.sr
    print(
        f"[API] Generated {audio_seconds:.2f}s audio in {elapsed:.2f}s "
        f"({audio_seconds / elapsed:.2f}x realtime), chunks={len(prompts)}"
    )

    waveform = combined.unsqueeze(0).cpu()
    buffer = io.BytesIO()
    ta.save(buffer, waveform, model.sr, format="wav")
    buffer.seek(0)

    headers = {
        "X-Generation-Seconds": f"{elapsed:.4f}",
        "X-Audio-Seconds": f"{audio_seconds:.4f}",
        "X-Realtime-Factor": f"{(audio_seconds / elapsed):.4f}",
        "X-Chunks": str(len(prompts)),
    }
    return StreamingResponse(buffer, media_type="audio/wav", headers=headers)


@app.exception_handler(Exception)
def _exception_handler(_, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error": str(exc)})


if __name__ == "__main__":
    host = os.environ.get("CHATTERBOX_API_HOST", "0.0.0.0")
    port = _env_int("CHATTERBOX_API_PORT", 8000)
    uvicorn.run(app, host=host, port=port, reload=False, workers=1)
