#!/usr/bin/env bash
set -euo pipefail

# Kill the Gradio app process started via: python gradio_tts_app.py
pkill -f "gradio_tts_app.py" || true
sleep 0.2

# Best-effort cleanup of orphaned vLLM EngineCore workers (V1 multiprocessing).
# These can linger if the parent process is terminated abruptly, and will keep
# VRAM allocated, preventing the next startup.
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
venv_python="${repo_dir}/.venv/bin/python"

orphan_pids="$(
  ps -eo pid=,ppid=,args= \
    | awk -v venv_python="${venv_python}" '$3 == venv_python && $0 ~ /multiprocessing\\.(spawn|resource_tracker)/ { print $1 }'
)"
if [[ -n "${orphan_pids}" ]]; then
  kill ${orphan_pids} || true
  sleep 0.2
  # Some stuck workers ignore SIGTERM; force-kill anything still alive.
  kill -9 ${orphan_pids} 2>/dev/null || true
fi
