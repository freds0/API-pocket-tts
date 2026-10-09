"""FastAPI HTTP and WebSocket API patterned after API-TTS-VoxCPM."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

import anyio
from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tts_api import audio
from tts_api.config import ConfigManager
from tts_api.core import PocketTTSEngine

logger = logging.getLogger(__name__)


class TTSRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    text: str
    voice: str | None = None
    language: str | None = None
    format: str = "wav"
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    temperature: float | None = Field(default=None, ge=0, le=5)
    lsd_decode_steps: int | None = Field(default=None, ge=1, le=64)
    inference_timesteps: int | None = Field(default=None, ge=1, le=64)
    noise_clamp: float | None = Field(default=None, gt=0)
    eos_threshold: float | None = None
    # Accepted in the schema for an explicit migration error, never silently ignored.
    cfg_value: float | None = None


class HealthResponse(BaseModel):
    status: str
    model: str
    available_models: list[str]
    voices: dict[str, str]
    default_voice: str
    sample_rate: int
    execution_providers: list[str]
    ready: bool
    warmup_completed: bool
    optimize_requested: bool
    finetuned: bool
    voice_conditioning: str


def validate_request(request: TTSRequest, config):
    if not request.text.strip():
        raise HTTPException(400, "'text' cannot be empty")
    if len(request.text) > config.POCKET_TTS_MAX_TEXT_LENGTH:
        raise HTTPException(400, f"'text' exceeds {config.POCKET_TTS_MAX_TEXT_LENGTH} characters")
    if request.voice not in (None, config.DEFAULT_VOICE):
        raise HTTPException(400, f"'voice' must be '{config.DEFAULT_VOICE}'")
    if request.language and request.language.lower() not in {"pt", "pt-br", "pt_br", "portuguese", "portuguese_24l"}:
        raise HTTPException(400, "This checkpoint supports Portuguese; use language='pt-BR'")
    if request.format.lower() not in {"wav", "mp3"}:
        raise HTTPException(400, "'format' must be one of: mp3, wav")
    if request.cfg_value is not None:
        raise HTTPException(400, "Pocket-TTS has no cfg_value parameter; use temperature")
    if (
        request.lsd_decode_steps is not None
        and request.inference_timesteps is not None
        and request.lsd_decode_steps != request.inference_timesteps
    ):
        raise HTTPException(400, "lsd_decode_steps and inference_timesteps must agree")


def generation_kwargs(request: TTSRequest) -> dict:
    return {
        "text": request.text.strip(),
        "seed": request.seed,
        "temperature": request.temperature,
        "lsd_decode_steps": request.lsd_decode_steps or request.inference_timesteps,
        "noise_clamp": request.noise_clamp,
        "eos_threshold": request.eos_threshold,
    }


def synthesis_error(exc: Exception) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, ValueError):
        return HTTPException(400, str(exc))
    if "out of memory" in str(exc).lower():
        return HTTPException(503, "Out of memory; retry later or use a shorter text")
    logger.error("Pocket-TTS synthesis failed", exc_info=(type(exc), exc, exc.__traceback__))
    return HTTPException(500, "Speech synthesis failed; inspect the server log")


@dataclass
class RuntimeState:
    engine: object = None
    semaphore: object = None
    ready: bool = False
    warmup_completed: bool = False


class PCMStream:
    """Keep the synthesis gate until the model and its worker have stopped."""

    def __init__(self, engine, semaphore):
        self.engine = engine
        self.semaphore = semaphore
        self.iterator = None
        self.acquired = False
        self.closed = False
        self.first = None
        self.info = {}

    async def open(self, request):
        started = time.perf_counter()
        await self.semaphore.acquire()
        self.acquired = True
        queue_ms = (time.perf_counter() - started) * 1000
        try:
            self.iterator = self.engine.generate_streaming(**generation_kwargs(request))
            model_started = time.perf_counter()
            chunk = await anyio.to_thread.run_sync(next, self.iterator, None)
            model_ttfc_ms = (time.perf_counter() - model_started) * 1000
            if chunk is None:
                raise RuntimeError("Pocket-TTS generated empty audio")
            pcm_started = time.perf_counter()
            self.first = audio.pcm16(chunk)
            pcm_ms = (time.perf_counter() - pcm_started) * 1000
            self.info = {
                "sample_rate": self.engine.sample_rate,
                "queue_ms": queue_ms,
                "model_ttfc_ms": model_ttfc_ms,
                "pcm_ms": pcm_ms,
                "server_ttfp_ms": (time.perf_counter() - started) * 1000,
                "first_chunk_samples": len(chunk),
            }
        except BaseException:
            await self.close()
            raise
        return self

    async def chunks(self):
        try:
            yield self.first
            while True:
                chunk = await anyio.to_thread.run_sync(next, self.iterator, None)
                if chunk is None:
                    break
                yield audio.pcm16(chunk)
        finally:
            await self.close()

    async def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.iterator is not None:
                with anyio.CancelScope(shield=True):
                    await anyio.to_thread.run_sync(self.iterator.close)
        finally:
            if self.acquired:
                self.semaphore.release()
                self.acquired = False

    def headers(self):
        info = self.info
        return {
            "X-Queue-Time-Ms": f"{info['queue_ms']:.2f}",
            "X-Model-TTFC-Ms": f"{info['model_ttfc_ms']:.2f}",
            "X-Server-TTFP-Ms": f"{info['server_ttfp_ms']:.2f}",
            "X-First-Chunk-Samples": str(info["first_chunk_samples"]),
            "Server-Timing": (
                f"queue;dur={info['queue_ms']:.2f}, "
                f"model_ttfc;dur={info['model_ttfc_ms']:.2f}, pcm;dur={info['pcm_ms']:.2f}"
            ),
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        }


class AudioStreamingResponse(StreamingResponse):
    def __init__(self, body, stream, **kwargs):
        super().__init__(body, **kwargs)
        self.stream = stream

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await self.body_iterator.aclose()
                finally:
                    await self.stream.close()


def create_app(engine_factory=None, config=ConfigManager):
    @asynccontextmanager
    async def lifespan(app):
        state = app.state.tts
        factory = engine_factory or (lambda: PocketTTSEngine(config))
        state.semaphore = asyncio.Semaphore(1)
        state.engine = await anyio.to_thread.run_sync(factory)
        try:
            if config.POCKET_TTS_WARMUP:
                await anyio.to_thread.run_sync(state.engine.warmup, config.POCKET_TTS_WARMUP_TEXT)
                state.warmup_completed = True
            state.ready = True
            yield
        finally:
            state.ready = False
            state.warmup_completed = False
            state.engine = None

    app = FastAPI(
        title="Pocket-TTS API",
        description="Local Douglas checkpoint with HTTP and WebSocket audio streaming.",
        version="1.0.0", lifespan=lifespan,
    )
    app.state.tts = RuntimeState()

    def engine():
        state = app.state.tts
        if not state.ready or state.engine is None:
            raise HTTPException(503, "Pocket-TTS backend is not ready")
        return state.engine

    async def open_stream(request):
        validate_request(request, config)
        stream = PCMStream(engine(), app.state.tts.semaphore)
        try:
            return await stream.open(request)
        except Exception as exc:
            raise synthesis_error(exc) from exc

    @app.get("/health", response_model=HealthResponse)
    async def health():
        backend = engine()
        return HealthResponse(
            status="healthy", model=f"Pocket-TTS ({backend.model_id})",
            available_models=[backend.model_id],
            voices={config.DEFAULT_VOICE: backend.model_id},
            default_voice=config.DEFAULT_VOICE,
            sample_rate=backend.sample_rate,
            execution_providers=backend.execution_providers,
            ready=app.state.tts.ready,
            warmup_completed=app.state.tts.warmup_completed,
            optimize_requested=backend.optimize_requested,
            finetuned=backend.finetuned,
            voice_conditioning="reference" if backend.voice_prompt else "unconditional",
        )

    @app.get("/models")
    async def models():
        backend = engine()
        return {
            "models": [backend.model_id], "default": backend.model_id,
            "finetuned": backend.finetuned,
            "voices": {config.DEFAULT_VOICE: backend.model_id},
            "default_voice": config.DEFAULT_VOICE,
        }

    @app.post("/tts")
    async def tts(request: TTSRequest):
        started = time.perf_counter()
        validate_request(request, config)
        backend = engine()
        audio_format = request.format.lower()
        try:
            if audio_format == "mp3":
                audio.require_mp3_encoder()
            queued = time.perf_counter()
            async with app.state.tts.semaphore:
                queue_ms = (time.perf_counter() - queued) * 1000
                waveform, metrics = await anyio.to_thread.run_sync(
                    lambda: backend.generate(**generation_kwargs(request))
                )
            content = await anyio.to_thread.run_sync(
                audio.encode_audio, waveform, backend.sample_rate, audio_format
            )
        except Exception as exc:
            raise synthesis_error(exc) from exc
        return Response(
            content, media_type="audio/wav" if audio_format == "wav" else "audio/mpeg",
            headers={
                "Content-Disposition": f'inline; filename="tts.{audio_format}"',
                "X-Audio-Format": audio_format,
                "X-Sample-Rate": str(backend.sample_rate if audio_format == "wav" else audio.MP3_SAMPLE_RATE),
                "X-Inference-Time": str(metrics["inference_time"]),
                "X-RTF": str(metrics["rtf"]),
                "X-Queue-Time-Ms": f"{queue_ms:.2f}",
                "X-Server-Total-Ms": f"{(time.perf_counter() - started) * 1000:.2f}",
                "Cache-Control": "no-store",
            },
        )

    @app.post("/tts/stream")
    async def tts_stream(request: TTSRequest):
        stream = await open_stream(request)
        return AudioStreamingResponse(
            stream.chunks(), stream,
            # Same legacy media type as VoxCPM; bytes are explicitly little endian.
            media_type=f"audio/L16; rate={stream.engine.sample_rate}; channels=1",
            headers={
                "X-Sample-Rate": str(stream.engine.sample_rate),
                "X-Audio-Format": "pcm_s16le", **stream.headers(),
            },
        )

    @app.get("/pocket-tts")
    @app.get("/douglas-tts")
    async def shortcut(text: str, format: str = "mp3"):
        request = TTSRequest(text=text, format=format)
        validate_request(request, config)
        if request.format.lower() == "mp3":
            try:
                audio.require_mp3_encoder()
            except RuntimeError as exc:
                raise HTTPException(503, str(exc)) from exc
        stream = await open_stream(request)
        audio_format = format.lower()
        if audio_format == "mp3":
            body = audio.mp3_stream(stream.chunks(), stream.engine.sample_rate)
            rate, media_type = audio.MP3_SAMPLE_RATE, "audio/mpeg"
        else:
            rate, media_type = 48000, "audio/wav"
            body = audio.wav_stream(
                audio.resample_pcm_stream(stream.chunks(), stream.engine.sample_rate, rate), rate
            )
        return AudioStreamingResponse(
            body, stream, media_type=media_type,
            headers={
                "Content-Disposition": f'inline; filename="douglas-tts.{audio_format}"',
                "X-Audio-Format": audio_format, "X-Sample-Rate": str(rate),
                **stream.headers(),
            },
        )

    async def websocket_item(websocket, request, metadata):
        started = time.perf_counter()
        stream = await open_stream(request)
        count = samples = 0
        source = stream.chunks()
        try:
            with anyio.fail_after(config.STREAM_SEND_TIMEOUT):
                await websocket.send_json({
                    "status": "metadata", "mode": "streaming",
                    "audio_format": "pcm_s16le", "channels": 1,
                    **stream.info, **metadata,
                })
            async for chunk in source:
                with anyio.fail_after(config.STREAM_SEND_TIMEOUT):
                    await websocket.send_bytes(chunk)
                count += 1
                samples += len(chunk) // 2
        finally:
            with anyio.CancelScope(shield=True):
                await source.aclose()
                await stream.close()
        return {
            "chunks": count, "total_samples": samples,
            "audio_duration": round(samples / stream.engine.sample_rate, 3),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    async def websocket_handler(websocket, batch=False):
        await websocket.accept()
        try:
            with anyio.fail_after(config.STREAM_SEND_TIMEOUT):
                message = await websocket.receive_json()
            if not isinstance(message, dict):
                raise ValueError("Request must be a JSON object")
            payload = dict(message)
            if batch:
                texts = payload.pop("texts", None)
                if not isinstance(texts, list) or not texts:
                    raise ValueError("'texts' must be a nonempty array")
                if len(texts) > config.MAX_BATCH_ITEMS:
                    raise ValueError(f"'texts' exceeds {config.MAX_BATCH_ITEMS} items")
                if any(not isinstance(text, str) for text in texts):
                    raise ValueError("Every item in 'texts' must be a string")
            else:
                texts = [payload.get("text", "")]
            completed = 0
            for index, text in enumerate(texts, 1):
                if batch and not text.strip():
                    continue
                try:
                    request = TTSRequest.model_validate({**payload, "text": text})
                    metrics = await websocket_item(websocket, request, {
                        **({"index": index} if batch else {}),
                        "text": text, "requested_format": request.format.lower(),
                    })
                    completed += 1
                    if batch:
                        await websocket.send_json({"status": "item_complete", "index": index, **metrics})
                    else:
                        await websocket.send_json({"status": "complete", **metrics})
                except (ValidationError, HTTPException, ValueError, RuntimeError) as exc:
                    detail = str(exc.detail) if isinstance(exc, HTTPException) else str(exc)
                    if batch:
                        await websocket.send_json({"status": "error", "index": index, "text": text, "error": detail})
                    else:
                        await websocket.send_text(f"Error: {detail}")
            if batch:
                await websocket.send_json({
                    "status": "complete", "total": len(texts),
                    "total_items": len(texts), "completed_items": completed,
                })
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.info("WebSocket closed: %s", exc)
            try:
                await websocket.send_text(f"Error: {exc}")
            except (RuntimeError, OSError):
                pass
        finally:
            try:
                await websocket.close()
            except (RuntimeError, OSError):
                pass

    @app.websocket("/tts")
    async def websocket_tts(websocket: WebSocket):
        await websocket_handler(websocket)

    @app.websocket("/tts/batch")
    async def websocket_batch_tts(websocket: WebSocket):
        await websocket_handler(websocket, batch=True)

    return app


app = create_app()
