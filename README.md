# Speech Data Collector

Internal tool for turning video, podcast, and local audio into clean timestamped markdown or TTS/STT training datasets. Local engines keep audio on-device; optional remote STT profiles send audio only to the API you configure.

**Copyright © 2026 Rohan Ahmed / Modern Intelligent Solutions. All rights reserved.** See [COPYRIGHT](COPYRIGHT) and [LICENSE](LICENSE).

## How it works

```text
YouTube / podcast / local file
        │
        ▼
   captions available? ──yes──► fetch captions (no media download)
        │ no / --transcribe / --api
        ▼
   transcribe locally (Whisper / Parakeet) or via remote STT API
        │
        ├──────────────► markdown (+ optional JSON sidecar)
        │                  paragraphs, timestamps, chapters
        │
        └──────────────► dataset mode (always word-timed ASR)
                           short clips + LJSpeech / NeMo / HF indexes
```

| Mode | Command | Output |
| --- | --- | --- |
| Markdown | `speechcollector SOURCE` | `.md` (and optional `.json`) |
| Dataset | `speechcollector dataset SOURCE` | `wavs/` + manifests + `dataset_card.md` |
| Web UI | `speechcollector web` | Browser at `http://localhost:8756` |
| MCP | `speechcollector mcp` | Stdio tools for agents |

## Quick start (local)

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/), [ffmpeg](https://ffmpeg.org/) on `PATH`.

```bash
# 1. Install deps into .venv
uv sync

# 2. Sanity check
uv run speechcollector --help

# 3. Try it
uv run speechcollector "https://youtu.be/VIDEO_ID"
uv run speechcollector web
```

Optional extras:

```bash
uv sync --extra parakeet   # fast Apple-Silicon transcription (macOS arm64)
uv sync --extra diarize    # speaker diarization for single-voice TTS datasets
uv sync --extra mcp        # MCP server for agent integration
```

Full local-dev notes (Windows tips, tests, env vars): **[docs/local-development.md](docs/local-development.md)**.

### ffmpeg

| OS | Install |
| --- | --- |
| Windows | `winget install Gyan.FFmpeg` |
| macOS | `brew install ffmpeg` |
| Debian/Ubuntu | `sudo apt install ffmpeg` |

## Usage

```bash
# YouTube / podcast / local file → markdown
uv run speechcollector "https://youtu.be/VIDEO_ID"
uv run speechcollector talk.mp3 -o talk.md --json

# Force a local model
uv run speechcollector talk.mp3 --transcribe --model small

# Build a TTS/STT dataset
uv run speechcollector dataset "https://youtu.be/VIDEO_ID" --out ./data
uv run speechcollector dataset talk.mp3 --sample-rate 16000 --format hf --out ./asr-data

# Local web UI
uv run speechcollector web
uv run speechcollector web --port 9000
```

### Markdown mode

Fetches captions when available; otherwise transcribes with Whisper, Parakeet (Apple Silicon), or a remote STT API. Output is readable markdown with timestamps and an optional JSON sidecar (`--json`) matching [`docs/schema.json`](docs/schema.json).

### Dataset mode

Always runs word-timestamped transcription, then slices audio on word boundaries into training clips. Exports LJSpeech, NeMo JSONL, and HuggingFace audiofolder layouts.

```bash
uv run speechcollector dataset SOURCE --out ./voice-data
uv run speechcollector dataset SOURCE --sample-rate 16000 --segment-min 2 --segment-max 12 --format hf
```

### Remote STT APIs

Any HTTP speech-to-text API can be used via a JSON profile (endpoint, headers, upload style, response field mapping). Secrets use `${ENV_VAR}` placeholders.

```bash
# Install a profile (see examples/apis/)
mkdir -p ~/.speechcollector/apis          # Windows: %USERPROFILE%\.speechcollector\apis
cp examples/apis/openai-compatible.json ~/.speechcollector/apis/openai.json
export OPENAI_API_KEY=sk-...

uv run speechcollector talk.mp3 --api openai
uv run speechcollector dataset talk.mp3 --api openai --out ./data
uv run speechcollector talk.mp3 --api-config ./examples/apis/openai-compatible.json
```

Profiles are searched in `$SPEECHCOLLECTOR_API_DIR`, `~/.speechcollector/apis/`, and `%APPDATA%\speechcollector\apis\`. Dataset mode needs a `words_path` in the profile. Details: [`examples/apis/README.md`](examples/apis/README.md).

## CLI reference

```text
speechcollector <SOURCE> [options]

  -o, --output PATH       Output file (default ./<id>.md)
  --output-dir PATH       Batch output directory (default ./speechcollector-out)
  --lang CODE             Language (captions default English; transcription auto-detects)
  --transcribe            Force transcription (skip captions)
  --model MODEL           auto | parakeet | parakeet-en | tiny … large-v3
  --api NAME              Remote STT profile name (overrides --model)
  --api-config PATH       Remote STT profile JSON path (overrides --model)
  --json                  Also write a .json sidecar
  --all [--limit N]       Batch ingest from playlist or feed

speechcollector dataset <SOURCE> [options]

  --out PATH              Dataset directory (default ./speechcollector-dataset)
  --format FMT            ljspeech | jsonl | hf
  --sample-rate HZ        Output WAV rate (default 22050)
  --segment-min/max S     Clip length bounds (default 1–15 s)
  --normalize             EBU R128 loudness normalization
  --no-filter             Skip quality filters
  --api / --api-config    Same as ingest (remote STT; needs word timestamps)

speechcollector web       Local web UI (--host, --port)
speechcollector mcp       MCP stdio server
```

## MCP server

```bash
uv sync --extra mcp
uv run speechcollector mcp
```

Environment variables: `SPEECHCOLLECTOR_MODEL` (including `api:<name>`), `SPEECHCOLLECTOR_LANG`, `SPEECHCOLLECTOR_VAD`, `SPEECHCOLLECTOR_PARAKEET_MODEL`, `SPEECHCOLLECTOR_API_DIR`.

## Project layout

```text
src/speechcollector/   Python package (CLI, pipeline, ASR, dataset, web UI)
tests/                 Offline pytest suite
examples/              Sample outputs + API profiles
docs/                  Schema, local-dev guide, design notes
CLAUDE.md              Guidance for coding agents working in this repo
```

## Development

```bash
uv sync
uv run pytest
uv run ruff check src tests
uv run mypy
```

Agent-oriented context (architecture, conventions, gotchas): **[CLAUDE.md](CLAUDE.md)**.  
Step-by-step local setup: **[docs/local-development.md](docs/local-development.md)**.
