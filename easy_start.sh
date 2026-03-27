source .venv/bin/activate && \
CHATTERBOX_MODEL_VARIANT=multilingual \
CHATTERBOX_GPU_MEMORY_UTILIZATION=0.3 \
python tts_api_server.py