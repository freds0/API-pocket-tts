"""Environment configuration; defaults match the local Douglas training run."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).lower()
    if value not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
        raise ValueError(f"{name} must be a boolean")
    return value in {"true", "1", "yes", "on"}


class ConfigManager:
    SERVER_HOST = os.getenv("API_HOST", "0.0.0.0")
    SERVER_PORT = int(os.getenv("API_PORT", "8000"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
    POCKET_TTS_REPO = Path(os.getenv("POCKET_TTS_REPO", "/home/fred/Projetos/pocket-tts")).expanduser()
    POCKET_TTS_CHECKPOINT_URL = os.getenv(
        "POCKET_TTS_CHECKPOINT_URL",
        "https://drive.google.com/file/d/17lRzpUDzQh0FlMcJ4kXfrIXyjOtDSwlJ/view?usp=sharing",
    ).strip()
    POCKET_TTS_CHECKPOINT_FILENAME = os.getenv("POCKET_TTS_CHECKPOINT_FILENAME", "step500-fm0.4251.ckpt")
    POCKET_TTS_CHECKPOINT = Path(os.getenv(
        "POCKET_TTS_CHECKPOINT",
        str(ROOT / "checkpoints"),
    )).expanduser()
    POCKET_TTS_BASE_CONFIG = Path(os.getenv(
        "POCKET_TTS_BASE_CONFIG", str(POCKET_TTS_REPO / "logs/pirula_tts/base_config.yaml")
    )).expanduser()
    POCKET_TTS_TOKENIZER = Path(os.getenv(
        "POCKET_TTS_TOKENIZER", str(POCKET_TTS_REPO / "logs/pirula_tts/tokenizer.model")
    )).expanduser()
    POCKET_TTS_CACHE_DIR = Path(os.getenv("POCKET_TTS_CACHE_DIR", str(ROOT / ".cache/model"))).expanduser()
    POCKET_TTS_DEVICE = os.getenv("POCKET_TTS_DEVICE", "auto")
    POCKET_TTS_VOICE_PROMPT = os.getenv(
        "POCKET_TTS_VOICE_PROMPT", str(ROOT / "reference.wav")
    ).strip()
    DEFAULT_VOICE = "douglas"
    POCKET_TTS_TEMPERATURE = float(os.getenv("POCKET_TTS_TEMPERATURE", "0.7"))
    POCKET_TTS_LSD_DECODE_STEPS = int(os.getenv("POCKET_TTS_LSD_DECODE_STEPS", "1"))
    POCKET_TTS_EOS_THRESHOLD = float(os.getenv("POCKET_TTS_EOS_THRESHOLD", "-4.0"))
    POCKET_TTS_NOISE_CLAMP = (
        float(os.environ["POCKET_TTS_NOISE_CLAMP"])
        if os.getenv("POCKET_TTS_NOISE_CLAMP", "").strip() else None
    )
    POCKET_TTS_NUM_THREADS = int(os.getenv("POCKET_TTS_NUM_THREADS", "1"))
    POCKET_TTS_MAX_TEXT_LENGTH = int(os.getenv("POCKET_TTS_MAX_TEXT_LENGTH", "2000"))
    POCKET_TTS_WARMUP = env_bool("POCKET_TTS_WARMUP", True)
    POCKET_TTS_WARMUP_TEXT = os.getenv("POCKET_TTS_WARMUP_TEXT", "Olá, este é um teste de voz.")
    STREAM_SEND_TIMEOUT = float(os.getenv("STREAM_SEND_TIMEOUT", "60"))
    MAX_BATCH_ITEMS = int(os.getenv("MAX_BATCH_ITEMS", "100"))
