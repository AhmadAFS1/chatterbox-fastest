# Increase Throughput Plan

## Goal

Improve end-to-end throughput, with special focus on `S3Gen`, while avoiding meaningful quality loss.

This plan is based on a full pass through the repo, especially:

- `tts_api_server.py`
- `src/chatterbox_vllm/tts.py`
- `src/chatterbox_vllm/models/s3gen/`
- `load_test_tts.sh`
- `README.md`

## Executive Summary

The current bottlenecks are a mix of real model cost and server-side serialization:

1. The API currently serializes all generation behind a global lock.
2. T3 uses batching through vLLM, but `S3Gen` still runs one output at a time.
3. The load test currently blends queue time and generation time, which makes tuning harder.
4. The HiFiGAN and F0 predictor path still has obvious inference-time overheads.
5. Server defaults are currently over-reserving memory for T3 on low-VRAM GPUs.

For long-form generation, the repo's own benchmark already shows that `S3Gen` dominates wall time. For short API requests, T3 is still a noticeable slice, but queueing becomes the biggest issue once concurrency rises.

## Current Bottlenecks

### 1. API request serialization

The biggest end-to-end throughput limiter is not raw S3Gen math. It is the server-level lock in `tts_api_server.py`.

- `generation_lock` is defined in `tts_api_server.py`
- Every request waits on `with generation_lock:`
- Uvicorn also runs with `workers=1`

Impact:

- Concurrent requests are accepted, but only one generation runs at a time.
- Later requests spend most of their time waiting in line.
- This is why API realtime factor degrades sharply under concurrent load.

Quality impact of fixing this: none.

### 2. T3 batches, S3Gen does not

The pipeline currently looks like this:

1. `tts.py` batches prompts through vLLM T3.
2. Then it loops over each T3 output.
3. Each output is passed through `self.s3gen.inference(...)` individually.

This means the main batching advantage stops after T3.

Relevant files:

- `src/chatterbox_vllm/tts.py`
- `src/chatterbox_vllm/models/s3gen/s3gen.py`
- `src/chatterbox_vllm/models/s3gen/flow.py`

There are still explicit batch-size-1 assumptions in the S3Gen path.

Impact:

- T3 gets faster with batching.
- S3Gen remains serialized per chunk/output.
- For multi-sentence and multi-request loads, S3Gen becomes the structural bottleneck.

Quality impact of fixing this: none if batching is implemented correctly.

### 3. Load-test timing is misleading under concurrency

The API timer starts before the generation lock is acquired, and the load test treats the returned header as model time.

Impact:

- `avg_model_gen` from `load_test_tts.sh` is not pure model time.
- It includes queue wait under concurrent load.
- This makes it harder to know whether a change improved actual inference or just reduced queueing.

Quality impact of fixing this: none.

### 4. HiFiGAN / vocoder path still carries inference overhead

In `src/chatterbox_vllm/models/s3gen/hifigan.py`:

- `weight_norm` is used heavily in conv layers and residual blocks.
- STFT and ISTFT are performed in the decode path.
- The source-generation path does extra tensor creation and transforms.

There is a `remove_weight_norm()` helper, but it is not currently used in inference and is not fully wired correctly yet.

Impact:

- Extra overhead in the final waveform generation stage.
- Prevents an easy inference-only speedup that should not change output quality.

Quality impact of fixing this: expected to be none if done correctly after load.

### 5. F0 predictor is also weight-norm heavy

In `src/chatterbox_vllm/models/s3gen/f0_predictor.py`, the F0 predictor uses stacked weight-normalized convolutions.

Impact:

- Adds inference-time overhead on every S3Gen pass.

Quality impact of fixing this: expected none if weight norm is removed only for inference after weights are loaded.

### 6. Diffusion cost scales directly with step count

The S3Gen diffusion path exposes `diffusion_steps`, and the generator comments already note:

- `10` is the original quality-oriented setting
- `5` can often be close in quality
- `2-3` degrade quality substantially

Current API defaults already run at `4`, which is speed-oriented.

Impact:

- Lower step counts improve speed.
- But this is one of the few levers that can change quality directly.

Quality impact: medium to high depending on step count.

### 7. Flow and cache infrastructure are underused

The code contains flow-cache and HiFiGAN cache concepts, but the normal API path largely ignores them.

Examples:

- HiFiGAN cache is explicitly ignored in `s3gen.py`
- The causal flow path returns `None` for cache in normal inference

Impact:

- Long-form or multi-chunk text recomputes more than it needs to.

Quality impact of improving this: none if cache behavior is correct.

### 8. Current server defaults over-allocate for T3 on low VRAM

`easy_start.sh` currently defaults to:

- `CHATTERBOX_GPU_MEMORY_UTILIZATION=0.8`
- `CHATTERBOX_MAX_BATCH_SIZE=10`
- `CHATTERBOX_MAX_MODEL_LEN=10000`

The vLLM heuristic in `src/chatterbox_vllm/tts.py` scales directly with `max_batch_size * max_model_len`.

Impact:

- On a 12 GB card, this likely gives too much headroom to T3/KV cache for short API requests.
- That can reduce flexibility for the rest of the pipeline without helping real request throughput.

Quality impact of fixing this: none, as long as the new `max_model_len` still covers the actual request shapes you care about.

### 9. Voice-clone conditioning is secondary, but still worth separating

The voice-cloning setup path uses:

- `librosa.load`
- `librosa.resample`
- S3 tokenizer work
- voice encoder embedding
- temp-file handling for uploaded audio

Caching is already present and helps, so this is not the main throughput limiter for repeated requests. It matters more for first-use latency and unique uploaded reference audio.

Quality impact of optimizing this: none.

## Prioritized Plan

### Priority 0: Fix observability first

Before tuning more, fix measurement so we can trust the data.

Changes:

1. Split API timing into:
   - `queue_wait_seconds`
   - `generation_seconds`
   - `t3_seconds`
   - `s3gen_seconds`
2. Move the current generation timer to start after lock acquisition.
3. Update `load_test_tts.sh` to print both:
   - end-to-end latency
   - pure generation time
4. Add optional response headers for stage timing.

Expected impact:

- Better decisions.
- Easier A/B testing.

Quality impact: none.

### Priority 1: Right-size server defaults for low-VRAM deployment

Current defaults are too aggressive for T3 memory reservation on a 12 GB GPU.

Recommended server defaults to test first:

- `CHATTERBOX_MAX_MODEL_LEN=500` or `1000`
- keep `CHATTERBOX_MAX_BATCH_SIZE=10` initially
- keep `CHATTERBOX_GPU_MEMORY_UTILIZATION` conservative

Notes:

- Most API requests in this repo are short.
- A `max_model_len` of `10000` is likely unnecessary for the common case.

Expected impact:

- Better GPU memory balance.
- Reduced memory pressure.
- Potentially better practical throughput on small GPUs.

Quality impact: none if request sizes still fit.

### Priority 2: Enable safe inference-time weight norm removal

Implement and call weight norm removal once after model load for:

- HiFiGAN generator
- HiFiGAN residual blocks
- source residual blocks
- F0 predictor conv stack

Also fix the current removal helper so it only targets modules that actually have weight norm applied.

Expected impact:

- Lower per-call inference overhead in waveform generation.

Quality impact: expected none.

### Priority 3: Add true S3Gen batching

This is the biggest structural improvement for S3Gen itself.

Work items:

1. Add `S3Gen.inference_batch(...)`
2. Remove batch-size-1 assumptions from:
   - token handling
   - flow inference
   - mel conditioning
   - vocoder inference
3. Batch multiple T3 outputs through S3Gen together when shapes permit
4. Keep output order stable

Why this matters:

- T3 already benefits from batching.
- S3Gen currently throws that advantage away.

Expected impact:

- Largest likely S3Gen throughput gain.
- Especially important for multi-sentence requests and queue backlogs.

Quality impact: none if batching is numerically equivalent.

### Priority 4: Reuse flow and vocoder caches across chunks

For sentence-split or long-form generation:

1. Use flow cache instead of recomputing overlap/prompt context each time
2. Reuse HiFiGAN cache where intended
3. Avoid resetting chunk-local state on every sentence when generating a continuous stream

Expected impact:

- Better long-form throughput.
- Lower redundant compute.

Quality impact: none if cache boundaries are validated carefully.

### Priority 5: Keep mixed precision selective

Current mixed precision works, but HiFiGAN still hits STFT/ISTFT behavior that is not clearly a win.

Recommended direction:

1. Keep token-to-mel / diffusion side as the first place to test AMP
2. Keep STFT-heavy vocoder sections in fp32 if needed
3. Compare:
   - full autocast
   - flow-only autocast
   - pure fp32

Expected impact:

- Possible moderate speedup.
- Less instability than blanket fp16 conversion.

Quality impact: low if selective and validated.

### Priority 6: Reduce Python and tensor-allocation overhead in the diffusion loop

In `flow_matching.py`, the Euler solver allocates and maintains CFG buffers per inference call.

Potential improvements:

1. Reuse work buffers instead of allocating them every call
2. Avoid unnecessary `.float()` conversions at the end of the solver when possible
3. Reduce dtype churn between stages
4. Profile whether CFG duplication is now a meaningful share of S3Gen time on short clips

Expected impact:

- Moderate gain on many short requests.

Quality impact: none if numerics are preserved.

## What To Avoid

### Do not rewrite the project in TensorFlow

That would be a major rewrite with low probability of practical throughput gains.

### Do not optimize against queued metrics alone

`avg_model_gen` from the current load test is not a clean proxy for pure model speed under concurrency.

### Do not push diffusion steps too low if quality matters

The repo already warns that very low steps degrade quality noticeably.

### Do not assume more VRAM automatically means much faster single-request generation

More VRAM helps with fit, headroom, and batching. It is not a direct linear speed multiplier by itself.

## Suggested Validation Order

### Phase 1: Measurement and config cleanup

1. Add clean stage timing
2. Lower `CHATTERBOX_MAX_MODEL_LEN`
3. Re-run:
   - single request
   - `10 total / 10 concurrent`
   - `100 total / 10 concurrent`

### Phase 2: Low-risk inference optimizations

1. Weight norm removal
2. Compare fp32 vs selective AMP
3. Verify waveform output quality manually

### Phase 3: Structural S3Gen improvements

1. Implement batched S3Gen
2. Add cache reuse across chunks
3. Re-benchmark long-form and short-form

## Benchmark Matrix To Use

Use the same text and reference audio for all A/B tests.

Recommended matrix:

1. `diffusion_steps`: `4`, `5`, `10`
2. `max_model_len`: `500`, `1000`, current large value
3. AMP mode:
   - disabled
   - selective flow-side AMP
   - broader autocast if stable
4. weight norm:
   - on
   - removed
5. request pattern:
   - single short request
   - 10 concurrent short requests
   - long-form benchmark

Track:

- queue wait
- T3 time
- S3Gen time
- total generation time
- end-to-end latency
- audio duration
- realtime factor

## Concrete Next Steps

Recommended implementation order:

1. Fix API and load-test timing so queue time and generation time are separated.
2. Lower `CHATTERBOX_MAX_MODEL_LEN` to a realistic deployment default.
3. Implement safe inference-time weight norm removal.
4. Re-benchmark on the current GPU.
5. If S3Gen is still the clear dominant cost, build a batched S3Gen inference path.
6. After batching, revisit cache reuse for long-form generation.

## Expected Best Wins

Highest confidence, lowest quality risk:

- better timing and measurement
- lower `max_model_len` for the API server
- weight norm removal at inference

Highest upside:

- true S3Gen batching
- chunk/cache reuse across long-form generation

Most quality-sensitive lever:

- diffusion step count

## Bottom Line

The biggest improvements are not likely to come from changing frameworks or blaming VRAM alone.

The current repo already has a fast T3 path. The main work now is:

1. stop over-measuring queue time as model time
2. stop serializing all useful work behind one path
3. make S3Gen more inference-optimized
4. batch S3Gen the same way T3 is already batched

That is the most credible path to better throughput without sacrificing output quality.
