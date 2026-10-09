import io
import wave
import shutil

import anyio
import numpy as np
import pytest
from fastapi.testclient import TestClient

from tts_api.config import ConfigManager
from tts_api.server import PCMStream, TTSRequest, create_app


class TestConfig(ConfigManager):
    __test__ = False
    POCKET_TTS_WARMUP = False


class FakeEngine:
    model_id = "/local/step500-fm0.4251.ckpt"
    sample_rate = 24000
    execution_providers = ["PocketTTS:cpu"]
    finetuned = True
    optimize_requested = False
    voice_prompt = ""

    def __init__(self):
        self.closed = 0
        self.calls = []
        self.failure = None
        self.chunks = [np.array([-1, 0, 1], dtype=np.float32), np.array([0.5], dtype=np.float32)]

    def generate_streaming(self, **kwargs):
        self.calls.append(kwargs)
        try:
            if self.failure:
                raise self.failure
            yield from self.chunks
        finally:
            self.closed += 1

    def generate(self, **kwargs):
        return np.concatenate(list(self.generate_streaming(**kwargs))), {
            "inference_time": 0.1, "rtf": 0.2,
        }


@pytest.fixture
def backend():
    return FakeEngine()


@pytest.fixture
def api(backend):
    app = create_app(lambda: backend, TestConfig)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_health_and_model_identity(api):
    response = api.get("/health")
    assert response.status_code == 200
    assert response.json()["ready"]
    assert response.json()["voice_conditioning"] == "unconditional"
    models = api.get("/models").json()
    assert models["default_voice"] == "douglas"
    assert models["voices"]["douglas"] == FakeEngine.model_id


def test_wav_contract(api, backend):
    response = api.post("/tts", json={"text": "Olá!", "seed": 42, "inference_timesteps": 2})
    assert response.status_code == 200
    assert response.headers["x-sample-rate"] == "24000"
    assert response.headers["x-audio-format"] == "wav"
    assert "x-rtf" in response.headers
    with wave.open(io.BytesIO(response.content)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 24000)
        assert np.frombuffer(wav.readframes(4), dtype="<i2").tolist() == [-32767, 0, 32767, 16383]
    assert backend.calls[0]["lsd_decode_steps"] == 2


def test_stream_is_little_endian_pcm_with_timing(api, backend):
    response = api.post("/tts/stream", json={"text": "Olá!"})
    assert response.status_code == 200
    assert response.headers["x-audio-format"] == "pcm_s16le"
    assert "x-model-ttfc-ms" in response.headers
    assert np.frombuffer(response.content, dtype="<i2").tolist() == [-32767, 0, 32767, 16383]
    assert backend.closed == 1


@pytest.mark.parametrize("route", ["/douglas-tts", "/pocket-tts"])
def test_shortcut_wav_is_streamed_at_48khz(api, route):
    response = api.get(route, params={"text": "Olá!", "format": "wav"})
    assert response.status_code == 200
    assert response.content.startswith(b"RIFF")
    assert response.headers["x-sample-rate"] == "48000"
    assert len(response.content[44:]) == 4 * 2 * 2


@pytest.mark.parametrize("payload,status", [
    ({"text": "  "}, 400),
    ({"text": "a" * 2001}, 400),
    ({"text": "Olá", "voice": "missing"}, 400),
    ({"text": "Olá", "format": "ogg"}, 400),
    ({"text": "Olá", "language": "en"}, 400),
    ({"text": "Olá", "cfg_value": 2}, 400),
    ({"text": "Olá", "lsd_decode_steps": 1, "inference_timesteps": 2}, 400),
    ({"text": "Olá", "temperature": -1}, 422),
    ({"text": "Olá", "lsd_decode_steps": 0}, 422),
    ({"text": "Olá", "unexpected": True}, 422),
])
def test_rejects_invalid_requests_without_inference(api, backend, payload, status):
    assert api.post("/tts", json=payload).status_code == status
    assert not backend.calls


@pytest.mark.parametrize("route", ["/tts", "/tts/stream"])
def test_startup_inference_failure_releases_gate(api, backend, route):
    backend.failure = RuntimeError("CUDA out of memory")
    assert api.post(route, json={"text": "Olá!"}).status_code == 503
    backend.failure = None
    assert api.post(route, json={"text": "Olá!"}).status_code == 200
    assert backend.closed == 2


def test_empty_stream_fails_before_http_headers(api, backend):
    backend.chunks = []
    assert api.post("/tts/stream", json={"text": "Olá!"}).status_code == 500
    assert backend.closed == 1


def test_websocket_contract(api):
    with api.websocket_connect("/tts") as ws:
        ws.send_json({"text": "Olá!", "seed": 42})
        metadata = ws.receive_json()
        assert metadata["status"] == "metadata"
        assert metadata["audio_format"] == "pcm_s16le"
        assert len(ws.receive_bytes()) == 6
        assert len(ws.receive_bytes()) == 2
        complete = ws.receive_json()
        assert complete["status"] == "complete"
        assert complete["total_samples"] == 4


def test_websocket_batch_boundaries_and_skipped_empty_items(api):
    with api.websocket_connect("/tts/batch") as ws:
        ws.send_json({"texts": ["Olá!", " ", "Tudo bem?"]})
        for index in (1, 3):
            assert ws.receive_json()["index"] == index
            ws.receive_bytes()
            ws.receive_bytes()
            assert ws.receive_json()["status"] == "item_complete"
        assert ws.receive_json() == {
            "status": "complete", "total": 3, "total_items": 3, "completed_items": 2,
        }


def test_websocket_rejects_non_array_batch(api):
    with api.websocket_connect("/tts/batch") as ws:
        ws.send_json({"texts": "Olá"})
        assert ws.receive_text().startswith("Error:")


@pytest.mark.anyio
async def test_stream_close_without_consuming_body_releases_gate(backend):
    semaphore = anyio.Semaphore(1)
    stream = await PCMStream(backend, semaphore).open(TTSRequest(text="Olá!"))
    assert semaphore.value == 0
    await stream.close()
    await stream.close()
    assert semaphore.value == 1
    assert backend.closed == 1


@pytest.mark.anyio
async def test_partial_consumer_releases_gate(backend):
    semaphore = anyio.Semaphore(1)
    stream = await PCMStream(backend, semaphore).open(TTSRequest(text="Olá!"))
    source = stream.chunks()
    assert await anext(source)
    await source.aclose()
    assert semaphore.value == 1
    assert backend.closed == 1


@pytest.mark.skipif(shutil.which("lame") is None, reason="LAME is not installed")
def test_mp3_full_and_streamed_shortcut(api):
    response = api.post("/tts", json={"text": "Olá!", "format": "mp3"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.headers["x-sample-rate"] == "22050"
    assert response.content
    shortcut = api.get("/douglas-tts", params={"text": "Olá!"})
    assert shortcut.status_code == 200
    assert shortcut.headers["x-audio-format"] == "mp3"
    assert shortcut.content
