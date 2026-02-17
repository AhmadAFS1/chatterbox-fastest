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
#   DIFFUSION_STEPS, MIN_P, TOP_P, REPETITION_PENALTY, SEED, TIMEOUT_SECONDS

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
import os
import statistics
import time
import urllib.parse
import urllib.request

url = os.environ["URL"]
n = int(os.environ["TOTAL_REQUESTS"])
c = int(os.environ["CONCURRENCY"])

timeout = float(os.environ.get("TIMEOUT_SECONDS", "180"))

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
gen_times = []
status_counts = {}
errors = 0

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
            _ = resp.read()
            elapsed = time.perf_counter() - t0
            gen_header = resp.headers.get("X-Generation-Seconds")
            gen = float(gen_header) if gen_header else None
            return elapsed, resp.status, gen, None
    except Exception as exc:
        return None, None, None, str(exc)

def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    idx = int((p / 100.0) * (len(sorted_vals) - 1))
    return sorted_vals[idx]

print(f"Running load test: URL={url} N={n} C={c}")
start = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
    for elapsed, status, gen, err in pool.map(one_request, range(n)):
        if err:
            errors += 1
            continue
        latencies.append(elapsed)
        status_counts[status] = status_counts.get(status, 0) + 1
        if gen is not None:
            gen_times.append(gen)
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

if gen_times:
    avg_gen = statistics.mean(gen_times)
    queue_overhead = max(0.0, avg - avg_gen)
    print(f"avg_model_gen={avg_gen:.3f}s est_queue_plus_overhead={queue_overhead:.3f}s")

print()
print("Tip: run again with higher C (e.g. 100) to see queueing impact.")
PY
