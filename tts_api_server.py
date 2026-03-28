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
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torchaudio as ta
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from chatterbox_vllm.text_utils import SUPPORTED_LANGUAGES
from chatterbox_vllm.tts import BatchGenerationRequest, ChatterboxTTS


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
S3GEN_USE_FP16 = _env_bool("CHATTERBOX_S3GEN_USE_FP16", False)
API_BATCH_COLLECT_SECONDS = _env_float("CHATTERBOX_API_BATCH_COLLECT_MS", 10.0) / 1000.0
API_MAX_BATCH_REQUESTS = _env_int("CHATTERBOX_API_MAX_BATCH_REQUESTS", MAX_BATCH_SIZE)
API_MAX_BATCH_PROMPTS = _env_int("CHATTERBOX_API_MAX_BATCH_PROMPTS", MAX_BATCH_SIZE)

if MODEL_VARIANT not in {"english", "multilingual"}:
    raise ValueError("CHATTERBOX_MODEL_VARIANT must be either 'english' or 'multilingual'.")

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

app = FastAPI(title="Chatterbox TTS API", version="0.1.0")

global_model: Optional[ChatterboxTTS] = None
cond_cache: OrderedDict[str, tuple[dict[str, Any], torch.Tensor]] = OrderedDict()
cond_cache_lock = threading.Lock()
generation_batcher: Optional["GenerationBatcher"] = None


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float
    diffusion_steps: int
    min_p: float
    top_p: float
    repetition_penalty: float
    seed: Optional[int]


@dataclass
class PendingGeneration:
    prompts: list[str]
    s3gen_ref: dict[str, Any]
    cond_emb: torch.Tensor
    language_id: str
    config: GenerationConfig
    received_at: float
    conditioning_seconds: float
    enqueued_at: float = field(default_factory=time.perf_counter)
    done: threading.Event = field(default_factory=threading.Event)
    wavs: Optional[list[torch.Tensor]] = None
    timings: dict[str, float] = field(default_factory=dict)
    error: Optional[Exception] = None

    @property
    def prompt_count(self) -> int:
        return max(1, len(self.prompts))


class GenerationBatcher:
    def __init__(
        self,
        collect_seconds: float,
        max_batch_requests: int,
        max_batch_prompts: int,
    ):
        self.collect_seconds = max(0.0, collect_seconds)
        self.max_batch_requests = max(1, max_batch_requests)
        self.max_batch_prompts = max(1, max_batch_prompts)
        self._pending: list[PendingGeneration] = []
        self._cv = threading.Condition()
        self._closed = False
        self._worker = threading.Thread(target=self._run, name="tts-generation-batcher", daemon=True)
        self._worker.start()

    def submit(self, job: PendingGeneration) -> list[torch.Tensor]:
        with self._cv:
            if self._closed:
                raise RuntimeError("Generation batcher is closed.")
            self._pending.append(job)
            self._cv.notify()

        job.done.wait()
        if job.error is not None:
            raise job.error
        return job.wavs or []

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        self._worker.join(timeout=5)

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._pending and not self._closed:
                    self._cv.wait()

                if self._closed and not self._pending:
                    return

                batch_deadline = time.perf_counter() + self.collect_seconds
                while not self._closed:
                    if len(self._pending) >= self.max_batch_requests:
                        break
                    remaining = batch_deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    self._cv.wait(timeout=remaining)

                jobs = self._take_next_batch_locked()

            self._process_batch(jobs)

    def _take_next_batch_locked(self) -> list[PendingGeneration]:
        anchor = self._pending[0]
        selected: list[PendingGeneration] = []
        selected_indices: list[int] = []
        total_prompts = 0

        for idx, job in enumerate(self._pending):
            if job.config != anchor.config:
                continue

            prompt_count = job.prompt_count
            if selected and (
                len(selected) >= self.max_batch_requests
                or total_prompts + prompt_count > self.max_batch_prompts
            ):
                continue

            selected.append(job)
            selected_indices.append(idx)
            total_prompts += prompt_count

            if len(selected) >= self.max_batch_requests or total_prompts >= self.max_batch_prompts:
                break

        if not selected:
            return [self._pending.pop(0)]

        for idx in reversed(selected_indices):
            self._pending.pop(idx)
        return selected

    def _process_batch(self, jobs: list[PendingGeneration]) -> None:
        model = load_model()
        batch_started_at = time.perf_counter()
        config = jobs[0].config

        if config.seed is not None:
            set_seed(config.seed)

        try:
            batched_requests = [
                BatchGenerationRequest(
                    prompts=job.prompts,
                    s3gen_ref=job.s3gen_ref,
                    cond_emb=job.cond_emb,
                    language_id=job.language_id,
                )
                for job in jobs
            ]
            wav_batches, stage_timings = model.generate_batched_with_conds(
                batched_requests,
                temperature=config.temperature,
                diffusion_steps=config.diffusion_steps,
                min_p=config.min_p,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                seed=config.seed,
                clear_cuda_cache=False,
            )
            finished_at = time.perf_counter()
        except Exception as exc:
            for job in jobs:
                job.error = exc
                job.done.set()
            return

        generation_seconds = finished_at - batch_started_at
        batch_request_count = float(len(jobs))
        batch_prompt_count = float(sum(job.prompt_count for job in jobs))

        for job, wavs in zip(jobs, wav_batches):
            job.wavs = wavs
            job.timings = {
                "queue_wait_seconds": batch_started_at - job.enqueued_at,
                "generation_seconds": generation_seconds,
                "t3_seconds": stage_timings["t3_seconds"],
                "s3gen_seconds": stage_timings["s3gen_seconds"],
                "batch_requests": batch_request_count,
                "batch_prompts": batch_prompt_count,
            }
            job.done.set()


def load_model() -> ChatterboxTTS:
    global global_model
    if global_model is not None:
        return global_model

    print(
        f"Loading model variant={MODEL_VARIANT} "
        f"s3gen_fp16={S3GEN_USE_FP16}..."
    )
    if MODEL_VARIANT == "multilingual":
        global_model = ChatterboxTTS.from_pretrained_multilingual(
            max_batch_size=MAX_BATCH_SIZE,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            compile=COMPILE,
            s3gen_use_fp16=S3GEN_USE_FP16,
        )
    else:
        global_model = ChatterboxTTS.from_pretrained(
            max_batch_size=MAX_BATCH_SIZE,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            compile=COMPILE,
            s3gen_use_fp16=S3GEN_USE_FP16,
        )
    return global_model


def _get_generation_batcher() -> GenerationBatcher:
    global generation_batcher
    if generation_batcher is None:
        generation_batcher = GenerationBatcher(
            collect_seconds=API_BATCH_COLLECT_SECONDS,
            max_batch_requests=API_MAX_BATCH_REQUESTS,
            max_batch_prompts=API_MAX_BATCH_PROMPTS,
        )
    return generation_batcher


@app.on_event("startup")
def _startup() -> None:
    load_model()
    _get_generation_batcher()


@app.on_event("shutdown")
def _shutdown() -> None:
    global generation_batcher
    if generation_batcher is not None:
        generation_batcher.close()
        generation_batcher = None


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
    request_started_at = time.perf_counter()

    if not text.strip():
        raise HTTPException(status_code=400, detail="`text` must not be empty.")

    if model.variant != "multilingual":
        language_id = "en"
    elif language_id.lower() not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported `language_id`: {language_id}")

    seed_value = int(seed) if seed != 0 else None

    conds_started_at = time.perf_counter()
    s3gen_ref, cond_emb = _get_conds(audio_prompt)
    cond_emb = model.update_exaggeration(cond_emb, exaggeration=exaggeration)
    conditioning_seconds = time.perf_counter() - conds_started_at
    prompts = maybe_split_text(text, split_sentences=split_sentences)

    job = PendingGeneration(
        prompts=prompts,
        s3gen_ref=s3gen_ref,
        cond_emb=cond_emb,
        language_id=language_id.lower(),
        config=GenerationConfig(
            temperature=temperature,
            diffusion_steps=diffusion_steps,
            min_p=min_p,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed_value,
        ),
        received_at=request_started_at,
        conditioning_seconds=conditioning_seconds,
    )
    wavs = _get_generation_batcher().submit(job)

    wav_encode_started_at = time.perf_counter()
    combined = torch.cat([w.squeeze(0) for w in wavs], dim=-1)
    waveform = combined.unsqueeze(0).cpu()
    buffer = io.BytesIO()
    ta.save(buffer, waveform, model.sr, format="wav")
    buffer.seek(0)
    wav_encode_seconds = time.perf_counter() - wav_encode_started_at

    generation_seconds = job.timings["generation_seconds"]
    queue_wait_seconds = job.timings["queue_wait_seconds"]
    t3_seconds = job.timings["t3_seconds"]
    s3gen_seconds = job.timings["s3gen_seconds"]
    batch_requests = int(job.timings["batch_requests"])
    batch_prompts = int(job.timings["batch_prompts"])
    audio_seconds = combined.shape[-1] / model.sr
    end_to_end_seconds = time.perf_counter() - request_started_at

    print(
        f"[API] batch_requests={batch_requests} batch_prompts={batch_prompts} "
        f"queue={queue_wait_seconds:.2f}s conds={conditioning_seconds:.2f}s "
        f"t3={t3_seconds:.2f}s s3gen={s3gen_seconds:.2f}s wav={wav_encode_seconds:.2f}s "
        f"audio={audio_seconds:.2f}s gen={generation_seconds:.2f}s "
        f"rtf={audio_seconds / generation_seconds:.2f}x chunks={len(prompts)}"
    )

    headers = {
        "X-Conditioning-Seconds": f"{conditioning_seconds:.4f}",
        "X-Queue-Wait-Seconds": f"{queue_wait_seconds:.4f}",
        "X-Generation-Seconds": f"{generation_seconds:.4f}",
        "X-T3-Seconds": f"{t3_seconds:.4f}",
        "X-S3Gen-Seconds": f"{s3gen_seconds:.4f}",
        "X-Wav-Encode-Seconds": f"{wav_encode_seconds:.4f}",
        "X-End-To-End-Seconds": f"{end_to_end_seconds:.4f}",
        "X-Audio-Seconds": f"{audio_seconds:.4f}",
        "X-Realtime-Factor": f"{(audio_seconds / generation_seconds):.4f}",
        "X-Chunks": str(len(prompts)),
        "X-Batch-Requests": str(batch_requests),
        "X-Batch-Prompts": str(batch_prompts),
    }
    return StreamingResponse(buffer, media_type="audio/wav", headers=headers)


@app.exception_handler(Exception)
def _exception_handler(_, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error": str(exc)})


if __name__ == "__main__":
    host = os.environ.get("CHATTERBOX_API_HOST", "0.0.0.0")
    port = _env_int("CHATTERBOX_API_PORT", 8000)
    uvicorn.run(app, host=host, port=port, reload=False, workers=1)
