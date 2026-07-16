# Local development

## Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** package manager
- **ffmpeg** on your `PATH` (dataset slicing + some yt-dlp merges)

| OS | Install ffmpeg |
| --- | --- |
| Windows | `winget install Gyan.FFmpeg` |
| macOS | `brew install ffmpeg` |
| Debian / Ubuntu | `sudo apt install ffmpeg` |

## First-time setup

From the repo root:

```bash
uv sync
uv run speechcollector --help
```

Optional extras:

```bash
uv sync --extra mcp        # MCP stdio server
uv sync --extra diarize    # pyannote speaker diarization (~GB of deps)
uv sync --extra parakeet   # Apple Silicon only (macOS arm64)
```

On Windows, if the console errors on Unicode, set:

```powershell
$env:PYTHONUTF8 = "1"
```

## Run the tool

Always prefer `uv run` so the project venv is used:

```bash
# Markdown from YouTube (captions when available)
uv run speechcollector "https://youtu.be/VIDEO_ID"

# Force transcription
uv run speechcollector "https://youtu.be/VIDEO_ID" --transcribe --model small

# Local file
uv run speechcollector path\to\talk.mp3 -o talk.md --json

# Dataset export
uv run speechcollector dataset path\to\talk.mp3 --out .\data

# Web UI → http://localhost:8756
uv run speechcollector web
uv run speechcollector web --port 9000
```

### Remote STT API (optional)

```bash
# Copy a profile (Windows example)
mkdir $env:USERPROFILE\.speechcollector\apis
copy examples\apis\openai-compatible.json $env:USERPROFILE\.speechcollector\apis\openai.json
$env:OPENAI_API_KEY = "sk-..."

uv run speechcollector talk.mp3 --api openai
```

See [examples/apis/README.md](../examples/apis/README.md).

## Tests and lint

```bash
uv run pytest
uv run pytest tests/test_remote.py -q
uv run ruff check src tests
uv run mypy
```

Tests are designed to stay **offline** (network and real ASR are monkeypatched). Integration tests that need a Whisper/Parakeet model will skip if the model is not already cached.

## Project layout

```text
src/speechcollector/     # library + CLI
  cli.py                 # Typer entry (ingest / dataset / web / mcp)
  pipeline.py            # captions → document, or download + transcribe
  transcribe.py          # whisper / parakeet / remote (api:…)
  remote.py              # configurable HTTP STT profiles
  webui.py               # local browser UI
  dataset/               # training-dataset export
tests/                   # pytest suite
examples/                # sample markdown, dataset, API profiles
docs/                    # schema + design notes
```

## Environment variables

| Variable | Purpose |
| --- | --- |
| `SPEECHCOLLECTOR_MODEL` | Default model for MCP (`auto`, Whisper size, or `api:<name>`) |
| `SPEECHCOLLECTOR_LANG` | Default language |
| `SPEECHCOLLECTOR_VAD` | `1` (default) or `0` for music |
| `SPEECHCOLLECTOR_PARAKEET_MODEL` | Override Parakeet HF repo id |
| `SPEECHCOLLECTOR_API_DIR` | Directory of remote STT profile JSON files |
| `HF_TOKEN` | Hugging Face token for diarization models |
| `OPENAI_API_KEY` (etc.) | Referenced from API profiles as `${ENV_VAR}` |
