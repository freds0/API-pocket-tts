"""Send one HTTP synthesis request per nonempty input line and save separate audio files."""

import argparse
import time
import wave
from pathlib import Path

import httpx


def output_for(base: Path, index: int, count: int, audio_format: str) -> Path:
    suffix = ".wav" if audio_format == "wav" else ".mp3"
    destination = base if base.suffix.lower() == suffix else base.with_suffix(suffix)
    if count > 1:
        destination = destination.with_name(f"{destination.stem}_{index:03d}{suffix}")
    return destination


def load_texts(args, parser) -> list[str]:
    if args.text is not None:
        texts = [args.text.strip()]
    else:
        try:
            texts = [
                line.strip()
                for line in args.input.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except OSError as exc:
            parser.error(f"cannot read input file {args.input}: {exc}")
    if not texts:
        parser.error("input contains no nonempty sentences")
    return texts


def synthesize(client: httpx.Client, args, text: str, output: Path, index: int, total: int):
    payload = {
        "text": text,
        "voice": "douglas",
        "format": args.format,
        "seed": args.seed,
        "temperature": args.temperature,
    }
    url = args.url.rstrip("/")
    started = time.perf_counter()
    first_audio_at = None
    print(f"[{index}/{total}] Sending request", flush=True)

    try:
        if args.stream:
            with client.stream("POST", url + "/tts/stream", json=payload) as response:
                response.raise_for_status()
                sample_rate = int(response.headers["X-Sample-Rate"])
                output.parent.mkdir(parents=True, exist_ok=True)
                with wave.open(str(output), "wb") as writer:
                    writer.setnchannels(1)
                    writer.setsampwidth(2)
                    writer.setframerate(sample_rate)
                    for chunk in response.iter_bytes():
                        if first_audio_at is None and chunk:
                            first_audio_at = time.perf_counter() - started
                            print(f"[{index}/{total}] First audio after {first_audio_at:.2f}s", flush=True)
                        writer.writeframesraw(chunk)
                status_code = response.status_code
                headers = response.headers
        else:
            response = client.post(url + "/tts", json=payload)
            response.raise_for_status()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(response.content)
            status_code = response.status_code
            headers = response.headers
    except httpx.HTTPStatusError as exc:
        elapsed = time.perf_counter() - started
        print(
            f"[{index}/{total}] HTTP {exc.response.status_code} after {elapsed:.2f}s: "
            f"{exc.response.text[:500]}", flush=True
        )
        raise
    except httpx.RequestError as exc:
        elapsed = time.perf_counter() - started
        print(f"[{index}/{total}] Request failed after {elapsed:.2f}s: {exc}", flush=True)
        raise

    elapsed = time.perf_counter() - started
    timing = f"response={elapsed:.2f}s"
    if first_audio_at is not None:
        timing += f", first_audio={first_audio_at:.2f}s"
    server_time = headers.get("X-Inference-Time")
    if server_time is not None:
        timing += f", inference={float(server_time):.2f}s"
    print(f"[{index}/{total}] HTTP {status_code} | {timing} | saved {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, default=Path("sentences.txt"))
    source.add_argument("--text", help="Synthesize one text instead of reading an input file")
    parser.add_argument("--output", type=Path, default=Path("outputs/sample.wav"))
    parser.add_argument("--format", choices=["wav", "mp3"], default="wav")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    if args.stream and args.format != "wav":
        parser.error("--stream saves raw PCM as WAV; choose --format wav")
    texts = load_texts(args, parser)

    with httpx.Client(timeout=600) as client:
        for index, text in enumerate(texts, start=1):
            output = output_for(args.output, index, len(texts), args.format)
            synthesize(client, args, text, output, index, len(texts))


if __name__ == "__main__":
    main()
