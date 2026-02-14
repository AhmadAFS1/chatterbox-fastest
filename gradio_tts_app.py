import random
import os
import re
import time
from functools import lru_cache

import numpy as np
import torch
import gradio as gr
import torchaudio as ta

from chatterbox_vllm.tts import ChatterboxTTS

DEVICE = "cuda"

config_seed = None
global_model = None

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    global config_seed
    config_seed = seed


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences at natural boundaries."""
    # Split on sentence-ending punctuation followed by space
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter empty strings and whitespace-only
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # If only 1 sentence or very short text, return as-is
    if len(sentences) <= 1:
        return [text.strip()]
    
    return sentences

def maybe_split_text(text: str, split_sentences: bool) -> list[str]:
    text = text.strip()
    if not split_sentences:
        return [text]
    return split_into_sentences(text)


def load_model():
    print("Loading model...")
    global global_model
    global_model = ChatterboxTTS.from_pretrained(
        max_batch_size = 10,   # Allow batching up to 10 sentences
        gpu_memory_utilization = 0.8,
        max_model_len = 500,
        # enforce_eager removed — CUDA graphs enabled
    )
    return global_model

# Cache audio conditioning per reference file
_cond_cache = {}

def get_cached_conds(audio_prompt_path):
    if audio_prompt_path not in _cond_cache:
        s3gen_ref, cond_emb = global_model.get_audio_conditionals(audio_prompt_path)
        _cond_cache[audio_prompt_path] = (s3gen_ref, cond_emb)
    return _cond_cache[audio_prompt_path]

def generate(text, audio_prompt_path, exaggeration, temperature, seed_num,
             split_sentences: bool,
             diffusion_steps,
             min_p, top_p, repetition_penalty):
    if seed_num != 0:
        set_seed(int(seed_num))
    else:
        global config_seed
        config_seed = None

    s3gen_ref, cond_emb = get_cached_conds(audio_prompt_path)
    cond_emb = global_model.update_exaggeration(cond_emb, exaggeration=exaggeration)

    # Splitting improves vLLM batching, but S3Gen runs per-chunk; for short clips
    # it's often faster to keep a single chunk.
    sentences = maybe_split_text(text, split_sentences=split_sentences)
    
    t0 = time.perf_counter()
    print(f"[GENERATE] Split into {len(sentences)} sentences: {[len(s) for s in sentences]}")
    for i, s in enumerate(sentences):
        print(f"  [{i}] {s[:60]}...")

    wavs = global_model.generate_with_conds(
        sentences,  # ALL sentences processed in parallel by VLLM
        s3gen_ref=s3gen_ref,
        cond_emb=cond_emb,
        temperature=temperature,
        diffusion_steps=diffusion_steps,
        min_p=min_p,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        seed=config_seed,
        clear_cuda_cache=False,
    )

    # Concatenate all audio chunks in order
    combined = torch.cat([w.squeeze(0) for w in wavs], dim=-1)
    
    elapsed = time.perf_counter() - t0
    audio_duration = combined.shape[-1] / global_model.sr
    print(f"[GENERATE] {elapsed:.2f}s for {audio_duration:.1f}s audio ({audio_duration/elapsed:.1f}x realtime) | {len(sentences)} sentences batched")

    return (global_model.sr, combined.numpy())


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Hey! How's it going? I'm glad to see you here. Let's get to know each other shall we? There are so many things I need to talk to you about!",
                label="Text to synthesize (max chars 300)",
                max_lines=5
            )
            split_sentences = gr.Checkbox(
                value=True,
                label="Batch by sentence (usually faster for short clips)",
            )
            ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File", value=None)
            exaggeration = gr.Slider(0.25, 2, step=.05, label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=.5)

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                diffusion_steps = gr.Slider(1, 15, step=1, label="Diffusion Steps (more = slower and higher quality)", value=4)
                temp = gr.Slider(0.05, 5, step=.05, label="temperature", value=.8)
                min_p = gr.Slider(0.00, 1.00, step=0.01, label="min_p", value=0.05)
                top_p = gr.Slider(0.00, 1.00, step=0.01, label="top_p", value=1.00)
                repetition_penalty = gr.Slider(1.00, 2.00, step=0.1, label="repetition_penalty", value=1.2)

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")

    run_btn.click(
        fn=generate,
        inputs=[
            text,
            ref_wav,
            exaggeration,
            temp,
            seed_num,
            split_sentences,
            diffusion_steps,
            min_p,
            top_p,
            repetition_penalty,
        ],
        outputs=audio_output,
    )

if __name__ == "__main__":
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    load_model()

    print("Starting Gradio app...")
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(share=True)
