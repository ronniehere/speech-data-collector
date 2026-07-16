# CLAUDE.md — Speech Data Collector

Private Python toolkit (`speechcollector`) that turns YouTube videos, podcast feeds, and local audio/video into (1) timestamped markdown or (2) TTS/STT training datasets. Prefer `uv` for all package and run commands.

## Commands

```bash
# Setup (Python 3.11+, ffmpeg on PATH)
uv sync
uv sync --extra mcp        # optional MCP server
uv sync --extra diarize    # optional speaker diarization (heavy)
# uv sync --extra parakeet # macOS arm64 only

# Run
uv run speechcollector --help
uv run speechcollector "https://youtu.be/VIDEO_ID"
uv run speechcollector dataset talk.mp3 --out ./data
uv run speechcollector web                 # http://localhost:8756
uv run speechcollector mcp                 # needs mcp extra

# Quality
uv run pytest
uv run pytest tests/test_remote.py -q
uv run ruff check src tests
uv run mypy
```

Windows console: set `PYTHONUTF8=1` (or `PYTHONIOENCODING=utf-8`) if you hit `charmap` encode errors. Do not print Unicode arrows (`→`) to stdout on Windows without that.

## Architecture

```
Source (URL / file / feed)
  → pipeline.py          captions-first, else transcribe
  → grouping + sectioning → Document
  → render.py            markdown (+ optional JSON sidecar)

Dataset mode (always transcribes with word_timestamps=True):
  → dataset/build.py → segment_words → filters → slice WAVs → formats
```

| Area | Path | Notes |
| --- | --- | --- |
| CLI entry | `src/speechcollector/cli.py` | Typer; default command is `ingest` |
| Pipeline | `pipeline.py` | Injectable fetchers/transcriber for offline tests |
| Transcription | `transcribe.py` | Engines: `whisper`, `parakeet`, `remote` (`api:…`) |
| Remote STT | `remote.py` | JSON profiles; stdlib `urllib` only |
| Dataset | `dataset/` | build, segmentation, filters, formats, diarize, audio |
| Web UI | `webui.py` | Single-file HTML/CSS/JS; no extra deps |
| Models | `models.py` | Pydantic; `Transcript` schema → `docs/schema.json` |
| Errors | `errors.py` | `CollectorError` + subclasses; CLI prints hint, no traceback |

**Model string** flows everywhere as `model_size`:

- Local: `auto` \| `parakeet` \| `parakeet-en` \| Whisper sizes
- Remote: `api:<name>` or `api:/abs/path.json` (CLI: `--api` / `--api-config`)

Profiles: `$SPEECHCOLLECTOR_API_DIR`, `~/.speechcollector/apis/`, `%APPDATA%\speechcollector\apis\`.

## Conventions

- Package lives under `src/speechcollector/` (src layout). Console script: `speechcollector`.
- Prefer stdlib for HTTP where possible (`feeds.py`, `remote.py`, `webui.py`).
- User-facing failures → `CollectorError` / `TranscriptionError` with a `hint`.
- Tests stay offline: monkeypatch network/transcription; fixtures in `tests/fixtures/`.
- Dataset builds need word timestamps; remote profiles without `words_path` must error clearly.
- Line length 100 (ruff); `webui.py` ignores E501 (embedded page).
- Do not reintroduce open-source branding, license files, or compliance/legal walls of text in dataset cards.

## Key docs

- [README.md](README.md) — install, usage, CLI
- [examples/apis/README.md](examples/apis/README.md) — remote STT profile schema
- [docs/dataset-mode-design.md](docs/dataset-mode-design.md) — dataset design notes
- [docs/schema.json](docs/schema.json) — Transcript JSON schema (keep in sync via `scripts/export_schema.py`)

## Gotchas

- `uv sync` can stall on large wheels (`av`, `ctranslate2`); one `uv` process at a time.
- Captions path does **not** provide word timings; dataset mode always re-transcribes.
- Web UI does not support playlists/feeds — CLI only.
- Diarization needs `speechcollector[diarize]` + HF token + accepted model terms.
- Changing `Transcript` fields: update model, run `uv run python scripts/export_schema.py`, fix `tests/test_transcript.py`.
