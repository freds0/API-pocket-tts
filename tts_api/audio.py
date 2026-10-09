"""PCM16, WAV and MP3 encoding following the VoxCPM output conventions."""

import asyncio
import io
import shutil
import struct
import subprocess
import wave

import anyio
import numpy as np

MP3_SAMPLE_RATE = 22050
MP3_BIT_RATE = 48


def pcm16(waveform: np.ndarray) -> bytes:
    return (np.clip(waveform, -1, 1) * 32767).astype("<i2").tobytes()


def wav_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm16(waveform))
    return output.getvalue()


def require_mp3_encoder():
    if not shutil.which("lame"):
        raise RuntimeError("MP3 requires LAME. Install it with conda install -c conda-forge lame")


def encode_audio(waveform: np.ndarray, sample_rate: int, audio_format: str) -> bytes:
    wav = wav_bytes(waveform, sample_rate)
    if audio_format == "wav":
        return wav
    require_mp3_encoder()
    result = subprocess.run(
        ["lame", "--silent", "--cbr", "-b", str(MP3_BIT_RATE), "--resample", "22.05", "-", "-"],
        input=wav, capture_output=True, timeout=120, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"MP3 encoding failed: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def streaming_wav_header(sample_rate: int) -> bytes:
    return (
        b"RIFF" + bytes([255]) * 4 + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data" + bytes([255]) * 4
    )


async def wav_stream(source, sample_rate: int):
    try:
        yield streaming_wav_header(sample_rate)
        async for chunk in source:
            yield chunk
    finally:
        await source.aclose()


async def resample_pcm_stream(source, orig_sr: int, target_sr: int):
    """Continuous interpolation for the 48 kHz WAV shortcut contract."""
    pending = np.empty(0, dtype=np.float64)
    position = 0.0
    step = orig_sr / target_sr
    try:
        async for chunk in source:
            pending = np.concatenate([pending, np.frombuffer(chunk, dtype="<i2")])
            count = max(0, int(np.ceil((len(pending) - 1 - position) / step)))
            if not count:
                continue
            positions = position + np.arange(count) * step
            out = np.interp(positions, np.arange(len(pending)), pending)
            yield np.clip(out, -32768, 32767).astype("<i2").tobytes()
            position += count * step
            consumed = min(int(position), len(pending) - 1)
            pending = pending[consumed:]
            position -= consumed
        if len(pending):
            positions = np.arange(position, len(pending), step)
            if len(positions):
                out = np.interp(positions, np.arange(len(pending)), pending)
                yield np.clip(out, -32768, 32767).astype("<i2").tobytes()
    finally:
        await source.aclose()


async def mp3_stream(source, sample_rate: int):
    process = None
    feeder = None
    try:
        require_mp3_encoder()
        command = [
            "lame", "--silent", "-r", "-s", f"{sample_rate / 1000:g}",
            "--bitwidth", "16", "--signed", "--little-endian", "-m", "m",
            "--cbr", "-b", str(MP3_BIT_RATE), "--resample", "22.05", "-", "-",
        ]
        if shutil.which("stdbuf"):
            command = ["stdbuf", "-o0", *command]
        process = await asyncio.create_subprocess_exec(
            *command, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )

        async def feed():
            try:
                async for chunk in source:
                    process.stdin.write(chunk)
                    await process.stdin.drain()
            finally:
                process.stdin.close()

        feeder = asyncio.create_task(feed())
        while chunk := await process.stdout.read(4096):
            yield chunk
        await feeder
        code = await process.wait()
        if code:
            raise RuntimeError(f"LAME exited with status {code}")
    finally:
        with anyio.CancelScope(shield=True):
            if feeder is not None and not feeder.done():
                feeder.cancel()
            if process is not None and process.returncode is None:
                process.kill()
            if feeder is not None:
                await asyncio.gather(feeder, return_exceptions=True)
            if process is not None:
                await process.wait()
            await source.aclose()
