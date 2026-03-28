# Increase Throughput Plan

## Goal

Improve end-to-end throughput, with special focus on `S3Gen`, while avoiding meaningful quality loss.

This plan is based on a full pass through the repo, especially:

- `tts_api_server.py`
- `src/chatterbox_vllm/tts.py`
- `src/chatterbox_vllm/models/s3gen/`
- `load_test_tts.sh`
- `README.md`

Historical note: sections below describe the original bottlenecks found before the first implementation pass. The current state now includes API micro-batching, stage timing headers, safe inference-time weight norm removal, and WAV validation in the load test.

## Executive Summary

The original bottlenecks were a mix of real model cost and server-side serialization. After the first implementation pass, the biggest remaining issues are different:

1. Compatible requests now batch together, but a late arrival still waits for the current GPU batch to finish.
2. T3 uses batching through vLLM, but `S3Gen` still does not batch internally.
3. The load test now separates queue time and generation time and validates returned WAV bodies.
4. HiFiGAN and the F0 predictor now remove weight norm safely at inference, so the easiest low-risk overhead reduction is already in place.
5. Server defaults still need empirical tuning on 12 GB GPUs because vLLM reservation interacts with the rest of the pipeline.

For long-form generation, the repo's own benchmark already shows that `S3Gen` dominates wall time. For short API requests, T3 is still a noticeable slice, but queueing becomes the biggest issue once concurrency rises.

## Findings As Of 2026-03-28

The first pass is implemented and measured.

What is now true:

* The old full-request `generation_lock` bottleneck is gone. The API batches compatible concurrent requests together before generation starts.
* `load_test_tts.sh` now validates that each response is a real WAV and that decoded duration matches `X-Audio-Seconds`.
* Batch timing is now broken out into queue wait, conditioning, T3, S3Gen, WAV encode, and end-to-end headers.

Key benchmark findings on an RTX 3080 Ti 12 GB:

* `10` total / `10` concurrent multilingual requests now batch correctly into one batch, with `valid_wavs=10/10`.
* `CHATTERBOX_S3GEN_USE_FP16=true` was slower than `false` in the tested multilingual runs.
* Lowering `diffusion_steps` remains the biggest request-level speed lever.
* Real React Native app traffic with batch size `1` and varying text lengths was already comfortably realtime on the same RTX 3080 Ti, with observed realtime factors from `1.33x` to `1.85x`.

Single-process burst results with `diffusion_steps=3`:

| Setting | avg_generation | avg_t3 | avg_s3gen | throughput |
| --- | ---: | ---: | ---: | ---: |
| `gpu_memory_utilization=0.25` | `4.971s` | `1.606s` | `3.365s` | `1.434 req/s` |
| `gpu_memory_utilization=0.50` | `5.197s` | `2.107s` | `3.090s` | `1.365 req/s` |

Staggered-arrival findings:

* A request sent `1.0s` after another request did not join the in-flight batch. It waited for the current batch to finish.
* At `gpu_memory_utilization=0.25`, that second request saw about `0.42s` queue wait.
* At `gpu_memory_utilization=0.50`, that second request saw about `0.82s` queue wait.
* With a `3.0s` gap, both settings behaved like independent single-request runs with queue wait around `0.01s`.

Multi-worker findings:

* Two independent model processes on the same 12 GB GPU were viable, but only with careful settings and startup order.
* The stable multilingual two-worker setup used `gpu_memory_utilization=0.25` and `max_model_len=300`, and the workers had to be started sequentially. Starting both at once was less reliable because model load briefly spikes GPU memory usage.
* In the main `2 workers` burst test, `10/10` requests were sent to each worker at the same time. All `20/20` requests succeeded, all returned valid WAVs, combined wall time was `7.716s`, and combined throughput was `2.592 req/s`.
* Per worker in that same test, `avg_generation` was about `4.06s` to `4.10s` for about `2.712s` of audio, `avg_t3` was about `1.87s` to `1.91s`, `avg_s3gen=2.186s`, `avg_queue_wait` was about `1.43s`, and `avg_batch_requests=5.8`.
* So the two-worker setup increased aggregate throughput, but it still did not deliver realtime per request under that `10 + 10 simultaneous` burst. The effective per-request realtime factor was about `0.66x`.
* Lowering `max_model_len` to `200` was also stable, but it made each worker slower per request: `avg_generation` rose to about `5.59s` to `5.65s` for about `2.648s` of audio, even though combined throughput nudged up to `2.656 req/s`.
* This makes separate worker processes behind a reverse proxy a promising deployment option for short-request burst traffic when total capacity matters more than single-request latency.
* It is not the same as blindly raising `uvicorn workers`, because each worker duplicates the full model stack and needs explicit VRAM budgeting.

Observed real app traces on the same RTX 3080 Ti:

| Pattern | avg_audio | avg_generation | avg_t3 | avg_s3gen | avg_rtf |
| --- | ---: | ---: | ---: | ---: | ---: |
| React Native single-request traffic | `4.208s` | `2.510s` | `2.022s` | `0.488s` | `1.658x` |

Interpretation:

* The repo is already good enough for realtime full-audio responses for single-user or lightly staggered app traffic on an RTX 3080 Ti.
* The main remaining gap is burst handling, not one-request latency.

## Current Bottlenecks

### 1. Single-GPU batch scheduling

The old server-level serialization has been replaced by a request queue plus micro-batching. The remaining scheduling limitation is that one GPU batch still runs at a time.

- Compatible pending requests can batch together before generation starts.
- Requests that arrive after a batch has already started wait for the next batch.
- The server still runs one model process per port by default.

Impact:

- Concurrent requests no longer serialize one-by-one.
- Near-simultaneous arrivals benefit from batching.
- Staggered arrivals still queue behind the in-flight batch.

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

### 3. Observability is fixed; keep using it

This item is no longer a blocker. The API now returns separate timing headers, and the load test validates the returned WAV body.

Keep relying on:

- `avg_queue_wait`
- `avg_t3`
- `avg_s3gen`
- `valid_wavs`
- `duration_mismatches`

These now make A/B tuning decisions much safer.

### 4. HiFiGAN / vocoder path still carries inference overhead

In `src/chatterbox_vllm/models/s3gen/hifigan.py`:

- `weight_norm` is used heavily in conv layers and residual blocks.
- STFT and ISTFT are performed in the decode path.
- The source-generation path does extra tensor creation and transforms.

Inference-time weight norm removal is now wired in safely after load, but the vocoder path is still a major share of the remaining `S3Gen` time.

Impact:

- Extra overhead in the final waveform generation stage.
- Prevents an easy inference-only speedup that should not change output quality.

Quality impact of fixing this: expected to be none if done correctly after load.

### 5. F0 predictor still contributes to S3Gen cost

In `src/chatterbox_vllm/models/s3gen/f0_predictor.py`, the F0 predictor uses stacked weight-normalized convolutions.

Weight norm removal is now applied safely for inference. The remaining concern is total S3Gen compute cost, not the old missing optimization.

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
