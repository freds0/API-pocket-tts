"""Generate and validate real audio through an in-process API and its checkpoint."""

import io
import sys
import wave
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tts_api.server import app


def main():
    output = Path("outputs")
    output.mkdir(exist_ok=True)
    request = {"text": "Olá, este é um teste da voz em português.", "seed": 42}
    with TestClient(app) as client:
        health = client.get("/health")
        health.raise_for_status()
        print(health.json())
        response = client.post("/tts", json=request)
        response.raise_for_status()
        with wave.open(io.BytesIO(response.content)) as wav:
            assert wav.getframerate() == 24000
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
            assert len(samples) > 2400, "Generated less than 100 ms of audio"
            assert np.max(np.abs(samples.astype(np.int32))) > 0, "Silent audio"
        (output / "smoke.wav").write_bytes(response.content)
        stream = client.post("/tts/stream", json=request)
        stream.raise_for_status()
        assert stream.headers["x-audio-format"] == "pcm_s16le"
        assert len(stream.content) > 4800 and len(stream.content) % 2 == 0
        with wave.open(str(output / "smoke_stream.wav"), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(stream.content)
        print("Real checkpoint generated WAV and PCM successfully.")
        print("Listen to outputs/smoke.wav and outputs/smoke_stream.wav to assess speech quality.")


if __name__ == "__main__":
    main()
