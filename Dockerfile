FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POCKET_TTS_REPO=/app/pocket-tts \
    POCKET_TTS_CHECKPOINT=/models \
    POCKET_TTS_BASE_CONFIG=/app/pocket-tts/logs/pirula_tts/base_config.yaml \
    POCKET_TTS_TOKENIZER=/app/pocket-tts/logs/pirula_tts/tokenizer.model \
    POCKET_TTS_CACHE_DIR=/var/cache/pocket-tts \
    POCKET_TTS_VOICE_PROMPT=/app/reference.wav \
    POCKET_TTS_DEVICE=cpu \
    API_HOST=0.0.0.0 \
    API_PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates lame libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install the CPU wheel explicitly so the default image does not pull CUDA libraries.
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.5,<3"

WORKDIR /app
COPY requirements-pocket-tts.txt ./
RUN python -m pip install -r requirements-pocket-tts.txt

COPY main.py ./
COPY reference.wav ./reference.wav
COPY tts_api ./tts_api
COPY scripts/download_checkpoint.py ./scripts/download_checkpoint.py
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint

RUN mkdir -p /models /var/cache/pocket-tts \
    && useradd --create-home --uid 10001 app \
    && chown -R app:app /models /var/cache/pocket-tts \
    && chmod 755 /usr/local/bin/docker-entrypoint

USER app
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/docker-entrypoint"]
CMD ["python", "main.py", "server"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
