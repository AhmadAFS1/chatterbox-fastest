#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./load_test_tts.sh [total_requests] [concurrency] [url]
#
# Examples:
#   ./load_test_tts.sh
#   ./load_test_tts.sh 100 10
#   ./load_test_tts.sh 1000 100 http://127.0.0.1:8000/v1/tts
#
# Optional env overrides:
#   TEXT, LANGUAGE_ID, SPLIT_SENTENCES, EXAGGERATION, TEMPERATURE,
#   DIFFUSION_STEPS, MIN_P, TOP_P, REPETITION_PENALTY, SEED, TIMEOUT_SECONDS,
#   VALIDATE_WAV, AUDIO_SECONDS_TOLERANCE, SAVE_RESPONSES_DIR, SAVE_RESPONSES_LIMIT

TOTAL_REQUESTS="${1:-100}"
CONCURRENCY="${2:-10}"
URL="${3:-http://127.0.0.1:8000/v1/tts}"

PYTHON_BIN=""
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "No python interpreter found (.venv/bin/python, python3, or python)." >&2
  exit 1
fi

TOTAL_REQUESTS="$TOTAL_REQUESTS" \
CONCURRENCY="$CONCURRENCY" \
URL="$URL" \
"$PYTHON_BIN" - <<'PY'
import concurrent.futures
import io
import os
import statistics
import time
import urllib.parse
import urllib.request
import wave

url = os.environ["URL"]
n = int(os.environ["TOTAL_REQUESTS"])
c = int(os.environ["CONCURRENCY"])

timeout = float(os.environ.get("TIMEOUT_SECONDS", "180"))
validate_wav = os.environ.get("VALIDATE_WAV", "true").lower() not in {"0", "false", "no"}
audio_tolerance = float(os.environ.get("AUDIO_SECONDS_TOLERANCE", "0.08"))
save_responses_dir = os.environ.get("SAVE_RESPONSES_DIR")
save_responses_limit = int(os.environ.get("SAVE_RESPONSES_LIMIT", "0"))

if save_responses_dir:
    os.makedirs(save_responses_dir, exist_ok=True)

form = {
    "text": os.environ.get("TEXT", "Hello this is a load test request."),
    "language_id": os.environ.get("LANGUAGE_ID", "en"),
    "split_sentences": os.environ.get("SPLIT_SENTENCES", "true"),
    "exaggeration": os.environ.get("EXAGGERATION", "0.5"),
    "temperature": os.environ.get("TEMPERATURE", "0.8"),
    "diffusion_steps": os.environ.get("DIFFUSION_STEPS", "4"),
    "min_p": os.environ.get("MIN_P", "0.05"),
    "top_p": os.environ.get("TOP_P", "1.0"),
    "repetition_penalty": os.environ.get("REPETITION_PENALTY", "1.2"),
    "seed": os.environ.get("SEED", "0"),
}
payload = urllib.parse.urlencode(form).encode()

latencies = []
conditioning_times = []
queue_wait_times = []
gen_times = []
t3_times = []
s3gen_times = []
end_to_end_times = []
audio_header_seconds = []
audio_decoded_seconds = []
batch_request_sizes = []
batch_prompt_sizes = []
status_counts = {}
errors = 0
valid_wavs = 0
invalid_wavs = 0
duration_mismatches = 0
saved_responses = 0


def maybe_save_response(idx: int, body: bytes):
    global saved_responses
    if not save_responses_dir or saved_responses >= save_responses_limit:
        return
    path = os.path.join(save_responses_dir, f"response_{idx:04d}.wav")
    with open(path, "wb") as fh:
        fh.write(body)
    saved_responses += 1


def validate_wav_response(body: bytes, expected_audio_seconds: float | None):
    if len(body) < 44 or body[:4] != b"RIFF" or body[8:12] != b"WAVE":
        return False, None, "missing_riff_wave_header"

    try:
        with wave.open(io.BytesIO(body), "rb") as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            decoded_seconds = frames / sample_rate if sample_rate else 0.0
    except wave.Error as exc:
        return False, None, f"wave_decode_error:{exc}"

    if expected_audio_seconds is not None and abs(decoded_seconds - expected_audio_seconds) > audio_tolerance:
        return False, decoded_seconds, (
            f"duration_mismatch:header={expected_audio_seconds:.4f}s decoded={decoded_seconds:.4f}s"
        )

    return True, decoded_seconds, None

def one_request(_idx: int):
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            elapsed = time.perf_counter() - t0
            def parse_header(name: str):
                value = resp.headers.get(name)
                return float(value) if value is not None else None

            audio_seconds = parse_header("X-Audio-Seconds")
            validation_error = None
            decoded_seconds = None
            if validate_wav:
                ok, decoded_seconds, validation_error = validate_wav_response(body, audio_seconds)
                maybe_save_response(_idx, body)
            elif save_responses_dir:
                maybe_save_response(_idx, body)

            return {
                "elapsed": elapsed,
                "status": resp.status,
                "conditioning": parse_header("X-Conditioning-Seconds"),
                "queue_wait": parse_header("X-Queue-Wait-Seconds"),
                "generation": parse_header("X-Generation-Seconds"),
                "t3": parse_header("X-T3-Seconds"),
                "s3gen": parse_header("X-S3Gen-Seconds"),
                "end_to_end": parse_header("X-End-To-End-Seconds"),
                "audio_seconds": audio_seconds,
                "decoded_audio_seconds": decoded_seconds,
                "batch_requests": parse_header("X-Batch-Requests"),
                "batch_prompts": parse_header("X-Batch-Prompts"),
                "response_bytes": len(body),
                "validation_error": validation_error,
            }, None
    except Exception as exc:
        return None, str(exc)

def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    idx = int((p / 100.0) * (len(sorted_vals) - 1))
    return sorted_vals[idx]

print(f"Running load test: URL={url} N={n} C={c}")
start = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
    for result, err in pool.map(one_request, range(n)):
        if err:
            errors += 1
            continue
        latencies.append(result["elapsed"])
        status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
        if result["conditioning"] is not None:
            conditioning_times.append(result["conditioning"])
        if result["queue_wait"] is not None:
            queue_wait_times.append(result["queue_wait"])
        if result["generation"] is not None:
            gen_times.append(result["generation"])
        if result["t3"] is not None:
            t3_times.append(result["t3"])
        if result["s3gen"] is not None:
            s3gen_times.append(result["s3gen"])
        if result["end_to_end"] is not None:
            end_to_end_times.append(result["end_to_end"])
        if result["audio_seconds"] is not None:
            audio_header_seconds.append(result["audio_seconds"])
        if result["decoded_audio_seconds"] is not None:
            audio_decoded_seconds.append(result["decoded_audio_seconds"])
        if result["batch_requests"] is not None:
            batch_request_sizes.append(result["batch_requests"])
        if result["batch_prompts"] is not None:
            batch_prompt_sizes.append(result["batch_prompts"])
        if result["validation_error"] is None:
            valid_wavs += 1
        else:
            invalid_wavs += 1
            if result["validation_error"].startswith("duration_mismatch:"):
                duration_mismatches += 1
            print(f"invalid_wav: {result['validation_error']}")
wall = time.perf_counter() - start

ok = len(latencies)
if ok == 0:
    print("No successful requests.")
    print(f"errors={errors}")
    raise SystemExit(1)

latencies_sorted = sorted(latencies)
p50 = percentile(latencies_sorted, 50)
p95 = percentile(latencies_sorted, 95)
p99 = percentile(latencies_sorted, 99)
avg = statistics.mean(latencies)
throughput = ok / wall if wall > 0 else 0.0

print()
print("=== Results ===")
print(f"success={ok}/{n} errors={errors}")
print(f"status_counts={status_counts}")
print(f"wall_time={wall:.3f}s throughput={throughput:.3f} req/s")
print(f"latency_avg={avg:.3f}s p50={p50:.3f}s p95={p95:.3f}s p99={p99:.3f}s")

if conditioning_times:
    print(f"avg_conditioning={statistics.mean(conditioning_times):.3f}s")
if queue_wait_times:
    print(f"avg_queue_wait={statistics.mean(queue_wait_times):.3f}s")
if gen_times:
    print(f"avg_generation={statistics.mean(gen_times):.3f}s")
if t3_times:
    print(f"avg_t3={statistics.mean(t3_times):.3f}s")
if s3gen_times:
    print(f"avg_s3gen={statistics.mean(s3gen_times):.3f}s")
if end_to_end_times:
    print(f"avg_end_to_end_header={statistics.mean(end_to_end_times):.3f}s")
if audio_header_seconds:
    print(f"avg_audio_header={statistics.mean(audio_header_seconds):.3f}s")
if audio_decoded_seconds:
    print(f"avg_audio_decoded={statistics.mean(audio_decoded_seconds):.3f}s")
if batch_request_sizes:
    print(f"avg_batch_requests={statistics.mean(batch_request_sizes):.2f}")
if batch_prompt_sizes:
    print(f"avg_batch_prompts={statistics.mean(batch_prompt_sizes):.2f}")
if validate_wav:
    print(f"valid_wavs={valid_wavs}/{ok} invalid_wavs={invalid_wavs} duration_mismatches={duration_mismatches}")
if save_responses_dir:
    print(f"saved_responses={saved_responses} dir={save_responses_dir}")

print()
print("Tip: compare avg_queue_wait, avg_t3, and avg_s3gen as you raise concurrency.")
PY
