"""HTTP client: WAV/MP3, or incremental PCM saved to a valid WAV."""

import argparse
import time
import wave
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, default=Path("sentences.txt"))
    source.add_argument("--text", help="Use this text instead of reading the input file")
    parser.add_argument("--output", type=Path, default=Path("outputs/sample.wav"))
    parser.add_argument("--format", choices=["wav", "mp3"], default="wav")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    text = args.text if args.text is not None else args.input.read_text(encoding="utf-8").strip()
    if not text:
        parser.error(f"input text is empty: {args.input}")
    if args.stream and args.format != "wav":
        parser.error("--stream saves raw PCM as WAV; choose --format wav")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "text": text, "voice": "douglas", "format": args.format,
        "seed": args.seed, "temperature": args.temperature,
    }
    started = time.perf_counter()
    with httpx.Client(timeout=600) as client:
        if args.stream:
            with client.stream("POST", args.url.rstrip("/") + "/tts/stream", json=payload) as response:
                response.raise_for_status()
                rate = int(response.headers["X-Sample-Rate"])
                first = True
                with wave.open(str(args.output), "wb") as writer:
                    writer.setnchannels(1)
                    writer.setsampwidth(2)
                    writer.setframerate(rate)
                    for chunk in response.iter_bytes():
                        if first:
                            print(f"First PCM byte: {(time.perf_counter() - started) * 1000:.1f} ms")
                            first = False
                        writer.writeframesraw(chunk)
        else:
            response = client.post(args.url.rstrip("/") + "/tts", json=payload)
            response.raise_for_status()
            args.output.write_bytes(response.content)
        print(dict(response.headers))
    print(f"Saved {args.output} in {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
