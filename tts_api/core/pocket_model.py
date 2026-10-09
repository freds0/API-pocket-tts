"""Load the local Pocket-TTS model once and serialize full/streaming inference."""

import logging
import math
import queue
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from tts_api.config import ConfigManager
from tts_api.core.checkpoint import prepare_config
from tts_api.core.gdrive import ensure_checkpoint

logger = logging.getLogger(__name__)


class PocketTTSEngine:
    def __init__(self, config=ConfigManager):
        repo = config.POCKET_TTS_REPO.resolve()
        if not (repo / "pocket_tts" / "__init__.py").is_file():
            raise FileNotFoundError(f"Pocket-TTS source not found in {repo}")
        # Use this checkout, including its 24-layer Portuguese architecture.
        sys.path.insert(0, str(repo))
        import torch
        from pocket_tts import TTSModel
        from pocket_tts.modules.stateful_module import init_states

        self.config = config
        self._lock = threading.Lock()
        self.checkpoint_path = ensure_checkpoint(
            config.POCKET_TTS_CHECKPOINT,
            config.POCKET_TTS_CHECKPOINT_URL,
            config.POCKET_TTS_CHECKPOINT_FILENAME,
        )
        self.model_id = str(self.checkpoint_path)
        self.finetuned = True
        self.optimize_requested = False
        self.device = config.POCKET_TTS_DEVICE
        if self.device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use POCKET_TTS_DEVICE=cpu")
        if config.POCKET_TTS_NUM_THREADS < 1:
            raise ValueError("POCKET_TTS_NUM_THREADS must be positive")
        torch.set_num_threads(config.POCKET_TTS_NUM_THREADS)
        config_file = prepare_config(
            self.checkpoint_path,
            config.POCKET_TTS_BASE_CONFIG,
            config.POCKET_TTS_TOKENIZER,
            config.POCKET_TTS_CACHE_DIR,
        )
        self.model = TTSModel.load_model(
            config=str(config_file),
            temp=config.POCKET_TTS_TEMPERATURE,
            lsd_decode_steps=config.POCKET_TTS_LSD_DECODE_STEPS,
            noise_clamp=config.POCKET_TTS_NOISE_CLAMP,
            eos_threshold=config.POCKET_TTS_EOS_THRESHOLD,
        ).to(self.device).eval()
        self.sample_rate = self.model.sample_rate
        self.execution_providers = [f"PocketTTS:{self.device}"]
        self.voice_prompt = config.POCKET_TTS_VOICE_PROMPT
        with torch.no_grad():
            if self.voice_prompt:
                path = Path(self.voice_prompt).expanduser().resolve(strict=True)
                if path.suffix == ".safetensors":
                    self.voice_state = self.model.get_state_for_audio_prompt(path)
                else:
                    # Handle PCM16, PCM24 and float WAV correctly.
                    import soundfile as sf
                    from scipy.signal import resample_poly

                    samples, rate = sf.read(path, dtype="float32", always_2d=True)
                    samples = samples[:30 * rate].mean(axis=1)
                    if samples.size == 0:
                        raise ValueError("Voice reference contains no audio")
                    if rate != self.sample_rate:
                        divisor = math.gcd(rate, self.sample_rate)
                        samples = resample_poly(
                            samples, self.sample_rate // divisor, rate // divisor
                        ).astype(np.float32)
                    prompt = torch.from_numpy(samples.copy()).unsqueeze(0)
                    self.voice_state = self.model.get_state_for_audio_prompt(prompt)
            else:
                # Training drops the voice prompt in 50% of examples.
                self.voice_state = init_states(
                    self.model.flow_lm, batch_size=1, sequence_length=1
                )
                logger.warning("No voice prompt configured; using unconditional finetuned voice")
        logger.info("Pocket-TTS loaded: %s, device=%s, rate=%s", self.model_id, self.device, self.sample_rate)

    def generate_streaming(
        self, text: str, temperature=None, lsd_decode_steps=None,
        noise_clamp=None, eos_threshold=None, seed=None,
    ):
        """Own all Torch contexts in one worker, with bounded delivery buffering.

        Closing a Pocket-TTS iterator early does not join its internal workers.
        On disconnect, consume remaining chunks without sending them, then release
        the model lock. Another request must not change sampling parameters or RNG
        while those workers are still using the model.
        """
        import torch

        results = queue.Queue(maxsize=4)
        stop = threading.Event()

        def emit(kind, value):
            while not stop.is_set():
                try:
                    results.put((kind, value), timeout=0.1)
                    return
                except queue.Full:
                    continue

        def produce():
            try:
                with self._lock:
                    names = ("temp", "lsd_decode_steps", "noise_clamp", "eos_threshold")
                    previous = {name: getattr(self.model, name) for name in names}
                    requested = dict(zip(names, (temperature, lsd_decode_steps, noise_clamp, eos_threshold)))
                    devices = [torch.device(self.device).index or 0] if self.device.startswith("cuda") else []
                    rng = torch.random.fork_rng(devices=devices) if seed is not None else nullcontext()
                    try:
                        for name, value in requested.items():
                            if value is not None:
                                setattr(self.model, name, value)
                        with rng, torch.no_grad():
                            if seed is not None:
                                torch.random.default_generator.manual_seed(seed)
                                for device in devices:
                                    with torch.cuda.device(device):
                                        torch.cuda.manual_seed(seed)
                            chunks = self.model.generate_audio_stream(
                                model_state=self.voice_state,
                                text_to_generate=text,
                                copy_state=True,
                            )
                            conversion_error = None
                            for chunk in chunks:
                                if not stop.is_set() and conversion_error is None:
                                    try:
                                        data = chunk.detach().float().cpu().numpy().reshape(-1)
                                        if not np.isfinite(data).all():
                                            raise RuntimeError("Pocket-TTS generated non-finite audio")
                                        if data.size:
                                            emit("chunk", data)
                                    except Exception as exc:
                                        # Drain the original iterator before releasing its workers.
                                        conversion_error = exc
                            if conversion_error is not None:
                                raise conversion_error
                    finally:
                        for name, value in previous.items():
                            setattr(self.model, name, value)
            except Exception as exc:
                emit("error", exc)
            finally:
                emit("done", None)

        worker = threading.Thread(target=produce, name="pocket-tts-inference", daemon=True)
        worker.start()
        try:
            while True:
                kind, value = results.get()
                if kind == "done":
                    return
                if kind == "error":
                    raise value
                yield value
        finally:
            stop.set()
            worker.join()

    def generate(self, text: str, **kwargs):
        started = time.perf_counter()
        chunks = list(self.generate_streaming(text, **kwargs))
        if not chunks:
            raise RuntimeError("Pocket-TTS generated empty audio")
        waveform = np.concatenate(chunks)
        elapsed = time.perf_counter() - started
        duration = waveform.size / self.sample_rate
        return waveform, {
            "inference_time": round(elapsed, 4),
            "audio_duration": round(duration, 4),
            "rtf": round(elapsed / duration, 4),
        }

    def warmup(self, text: str):
        self.generate(text, seed=0)
