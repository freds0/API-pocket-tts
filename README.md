# Pocket-TTS API

FastAPI service for the Douglas/BRSpeech checkpoint in the local Pocket-TTS
checkout, following the structure and audio contracts of the VoxCPM API at
`/home/fred/Projetos/NativeVoice/CareBears/API-TTS-VoxCPM`.

The server uses the code in `/home/fred/Projetos/pocket-tts`. Its defaults are:

- Checkpoint: `checkpoints/step500-fm0.4251.ckpt`.
- Architecture: `logs/pirula_tts/base_config.yaml`, with 24 layers.
- Tokenizer: `logs/pirula_tts/tokenizer.model`.
- Audio: mono, 24,000 Hz; the API encodes output as PCM16.
- Logical voice: `douglas`.

## Installation and Conda environment

Run these commands from this API's directory:

```bash
bash scripts/create_conda_env.sh
conda activate /home/fred/Projetos/API-pocket-tts/.conda/api-pocket-tts
python main.py server
```

The script creates an isolated environment inside the project with Python 3.12,
PyTorch, inference dependencies, FastAPI, LAME, and testing tools. You can also
choose a different prefix:

```bash
bash scripts/create_conda_env.sh /path/to/env/api-pocket-tts
```

Dependencies must be available in the cache or over the network. Pocket-TTS is
imported directly from `POCKET_TTS_REPO`; you do not need to install or modify
the model checkout. The script installs PyTorch from PyPI. To use a different
CUDA or CPU build, install the appropriate PyTorch version in the same
environment.

The API loads the checkpoint from `checkpoints/` at startup. The Lightning
checkpoint is exported to `.cache/model/<identifier>/model.safetensors`. This export
contains only inference weights and uses the local architecture and tokenizer
files, without downloading base weights. Allow about 1.3 GB for the cache, in
addition to the environment dependencies, and enough memory to read the
approximately 3.7 GB checkpoint during export.

You can export the checkpoint before starting the server:

```bash
python main.py export
```

If `last.ckpt` is missing, the server selects the `.ckpt` file with the highest
numeric step. To pin a checkpoint, set the full path to the file. A change to
the checkpoint, configuration, or tokenizer creates a new cache entry.

## Docker

Build and start the CPU image with Docker Compose:

```bash
docker compose up --build
```

Compose mounts the local Pocket-TTS checkout at `/home/fred/Projetos/pocket-tts`
by default. Set `POCKET_TTS_SOURCE_PATH` in `.env` if the checkout is elsewhere.
It also mounts this project's `checkpoints` directory at `/models`, so put
`step500-fm0.4251.ckpt` there before starting the container. The separate
`pocket-tts-cache` volume holds the exported model weights.

The provided image uses CPU-only PyTorch. To use a GPU, build an image with the
matching PyTorch CUDA distribution and set `POCKET_TTS_DEVICE=cuda:0`. The
checkpoint is about 3.7 GB; allow additional memory and storage for its 1.3 GB
inference export.

## Configuration

Copy `.env.example` to `.env` and edit the values as needed. Variables already
set in the shell take precedence.

| Variable | Default / purpose |
| --- | --- |
| `API_HOST`, `API_PORT` | `0.0.0.0`, `8000` |
| `POCKET_TTS_REPO` | Local Pocket-TTS checkout |
| `POCKET_TTS_CHECKPOINT` | Checkpoint directory or file; defaults to `checkpoints/` |
| `POCKET_TTS_BASE_CONFIG` | Training architecture YAML |
| `POCKET_TTS_TOKENIZER` | Tokenizer used during training |
| `POCKET_TTS_CACHE_DIR` | `.cache/model` inside this API project |
| `POCKET_TTS_DEVICE` | `auto`: use CUDA if available, otherwise CPU |
| `POCKET_TTS_VOICE_PROMPT` | `reference.wav` in the project root; accepts local audio or a `.safetensors` state |
| `POCKET_TTS_TEMPERATURE` | `0.7` |
| `POCKET_TTS_LSD_DECODE_STEPS` | `1` |
| `POCKET_TTS_EOS_THRESHOLD` | `-4.0` |
| `POCKET_TTS_NOISE_CLAMP` | Empty: no clamp |
| `POCKET_TTS_NUM_THREADS` | `1` PyTorch operation thread |
| `POCKET_TTS_MAX_TEXT_LENGTH` | `2000` characters per text |
| `POCKET_TTS_WARMUP` | `true` |
| `STREAM_SEND_TIMEOUT` | `60` seconds for WebSocket receive/send operations |
| `MAX_BATCH_ITEMS` | `100` texts per batch |

To require a GPU, set `POCKET_TTS_DEVICE=cuda:0`. Startup fails if CUDA is
unavailable. For CPU testing, set `POCKET_TTS_DEVICE=cpu`.

By default, synthesis uses `reference.wav` from the project root. Docker copies
this file into the image and mounts the local file at `/app/reference.wav`. To
use a different reference, set `POCKET_TTS_VOICE_PROMPT` to its path (and, with
Docker Compose, update the mounted file path as needed):

```dotenv
POCKET_TTS_VOICE_PROMPT=/path/to/douglas_reference.wav
```

The reference audio is converted to mono at 24 kHz and limited to its first
30 seconds. `GET /health` reports `voice_conditioning=reference` or
`unconditional`.

## Endpoints

Interactive API documentation is available at <http://localhost:8000/docs>.

| Method | Path | Response |
| --- | --- | --- |
| GET | `/health` | Readiness, model, device, and voice mode |
| GET | `/models` | Available checkpoint and voice |
| POST | `/tts` | Complete 24 kHz WAV or MP3 |
| POST | `/tts/stream` | Incremental 24 kHz mono PCM16, little-endian |
| GET | `/douglas-tts?text=...&format=mp3` | Incremental MP3; MP3 is the default |
| GET | `/pocket-tts?text=...&format=wav` | Same shortcut; incremental 48 kHz WAV |
| WebSocket | `/tts` | JSON metadata, binary PCM frames, and completion JSON |
| WebSocket | `/tts/batch` | Sequence of items with explicit boundaries |

MP3 matches the VoxCPM API settings: LAME, 48 kbps CBR, 22.05 kHz. The GET
shortcut WAV uses 48 kHz and an open-ended size header; the POST WAV has an
accurate size and retains the native 24 kHz rate.

Example request:

```bash
curl http://localhost:8000/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello, this is Portuguese speech.","voice":"douglas","seed":42,"format":"wav"}' \
  --output speech.wav
```

Optional parameters: `temperature`, `lsd_decode_steps`, `noise_clamp`,
`eos_threshold`, `seed`, and `language` (`pt` or `pt-BR`).
`inference_timesteps` is an alias for `lsd_decode_steps`; conflicting values
are rejected. VoxCPM's `cfg_value` has no equivalent in Pocket-TTS and returns an
explanatory error. The VoxCPM voices `dj_kb` and `cheer_bear` are replaced by
the `douglas` voice.

HTTP and WebSocket streaming always send raw PCM, regardless of `format`.
Clients should use `X-Sample-Rate` or the JSON metadata. For compatibility
with VoxCPM, the HTTP media type remains `audio/L16`; the actual encoding is
**little-endian**, as indicated by `X-Audio-Format: pcm_s16le`.

`POST /tts` returns `X-Inference-Time` (seconds), `X-RTF`,
`X-Queue-Time-Ms`, and `X-Server-Total-Ms`. Streams return
`X-Model-TTFC-Ms`, `X-Server-TTFP-Ms`, `X-First-Chunk-Samples`, and
`Server-Timing`. The server obtains the model's first chunk before sending
headers, so early failures can be returned as HTTP errors. Failures after the
first chunk terminate the stream.

## WebSocket

Send `{"text":"Hello!","seed":42}` to `ws://localhost:8000/tts`. The response
contains:

1. JSON with `status=metadata`, sample rate, channels, format, and timings.
2. Binary frames containing little-endian PCM16 samples.
3. JSON with `status=complete`, chunk count, sample count, and duration.

For `/tts/batch`, send `{"texts":["Hello!","How are you?"],"seed":42}`. Each
item has an `index` starting at 1, metadata, binary frames, and
`status=item_complete`. Empty texts are skipped; per-item errors use
`status=error`. The final JSON reports `completed_items`.

## Clients and testing

With the environment activated:

```bash
python scripts/inference.py --input sentences.txt --output outputs/speech.wav
python scripts/inference.py --text "Hello, how are you?" --output outputs/speech.wav
python scripts/inference.py --stream --output outputs/stream.wav
python scripts/inference.py --format mp3 --output outputs/speech.mp3

# Contract, export, and stream shutdown tests; these do not load the real checkpoint.
python -m pytest -q

# Loads the real checkpoint and checks WAV/PCM through the application.
python scripts/smoke_test.py
```

The smoke test saves `outputs/smoke.wav` and `outputs/smoke_stream.wav`. It
checks the format, minimum duration, and presence of a signal. Listen to the
files to assess speech quality.

Run one Uvicorn worker per instance. Pocket-TTS generation is not thread-safe
for concurrent calls on the same model. The API serializes inference and uses
a delivery queue limited to four chunks. The original generator does not stop
its internal threads when its iterator is closed, so after a disconnect the API
discards the remaining chunks and waits for synthesis to finish before releasing
the model for the next request.
