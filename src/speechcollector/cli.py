"""Command-line interface for speechcollector.

A Typer command group with a default command (``ingest``), so
``speechcollector <SOURCE> [options]`` works without typing ``ingest`` while
``speechcollector mcp`` runs the MCP server. SOURCE may be a YouTube video/playlist URL,
a podcast RSS feed, or a local audio/video file. Single sources write one
markdown file; batch sources (playlists/feeds) list their items, or ingest a
selection into an output directory.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeElapsedColumn
from rich.table import Table
from typer.core import TyperGroup

from speechcollector import __version__
from speechcollector.batch import BatchItem, ensure_unique_slugs, run_batch, select, slugify
from speechcollector.errors import CollectorError, OutputWriteError
from speechcollector.feeds import Episode, fetch_feed
from speechcollector.models import Document, Transcript
from speechcollector.pipeline import (
    NoCaptionsError,
    ingest_episode,
    ingest_file,
    ingest_youtube,
    ingest_youtube_transcribe,
)
from speechcollector.render import render_markdown
from speechcollector.timefmt import format_timestamp
from speechcollector.transcribe import DEFAULT_MODEL
from speechcollector.youtube import (
    PlaylistEntry,
    extract_playlist_id,
    extract_video_id,
    fetch_playlist,
)

PITCH = (
    "Collect speech data from video, podcasts, and recordings — "
    "timestamped markdown or TTS/STT training datasets."
)


class ModelSize(StrEnum):
    """Transcription model: ``auto``, Parakeet (MLX), or a Whisper size.

    ``auto`` picks Parakeet on Apple Silicon (~3x faster) and falls back to
    ``small`` Whisper elsewhere. Whisper sizes run smallest (fastest) to largest
    (most accurate) on CPU.
    """

    auto = "auto"
    parakeet = "parakeet"
    parakeet_en = "parakeet-en"
    tiny = "tiny"
    base = "base"
    small = "small"
    medium = "medium"
    large_v3 = "large-v3"


class DatasetFormat(StrEnum):
    """A dataset index format. ``ljspeech`` + ``jsonl`` are the default pair."""

    ljspeech = "ljspeech"
    jsonl = "jsonl"
    hf = "hf"


_DEFAULT_MODEL = ModelSize(DEFAULT_MODEL)
_DEFAULT_OUTPUT_DIR = Path("./speechcollector-out")
_DEFAULT_DATASET_DIR = Path("./speechcollector-dataset")


def _resolve_model_choice(
    model: ModelSize,
    api: str | None,
    api_config: Path | None,
) -> str:
    """Pick the effective model string; ``--api`` / ``--api-config`` override ``--model``."""
    if api is not None and api_config is not None:
        raise typer.BadParameter("Use either --api or --api-config, not both.")
    if api_config is not None:
        path = api_config.expanduser()
        if not path.is_file():
            raise typer.BadParameter(f"API config not found: {path}")
        return f"api:{path.resolve()}"
    if api is not None:
        name = api.strip()
        if not name:
            raise typer.BadParameter("--api requires a non-empty profile name.")
        return f"api:{name}"
    return model.value


class DefaultCommandGroup(TyperGroup):
    """A command group whose default command is ``ingest``.

    Lets ``speechcollector <SOURCE> [options]`` work without typing ``ingest`` while
    still supporting the ``mcp`` subcommand. If the first argument is neither a
    known command nor an option, ``ingest`` is prepended.
    """

    default_command = "ingest"
    # Options handled by the group itself (not by the default command).
    _group_options = frozenset({"--version", "--help", "-h"})

    def parse_args(self, ctx, args):
        # Prepend `ingest` unless the first token is a known subcommand or a
        # group-level option, so both `speechcollector <SOURCE> [options]` and
        # `speechcollector [options] <SOURCE>` reach ingestion, while `speechcollector mcp`,
        # `speechcollector --version`, and `speechcollector --help` still work.
        if args and args[0] not in self.commands and args[0] not in self._group_options:
            args = [self.default_command, *args]
        return super().parse_args(ctx, args)


app = typer.Typer(
    cls=DefaultCommandGroup,
    add_completion=False,
    no_args_is_help=True,
    help=PITCH,
)
console = Console()
err_console = Console(stderr=True)


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"speechcollector {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_show_version, is_eager=True, help="Show the version and exit."
        ),
    ] = False,
) -> None:
    """Collect speech data — timestamped markdown or TTS/STT training datasets."""


@app.command("mcp", help="Run the MCP stdio server (give your AI agent ears).")
def mcp_command() -> None:
    """Start the speechcollector MCP server on stdio."""
    from speechcollector.mcp_server import run_server

    try:
        run_server()
    except KeyboardInterrupt:
        raise typer.Exit(code=130) from None
    except CollectorError as exc:
        _print_error(exc)
        raise typer.Exit(code=1) from exc


@app.command("web", help="Run a simple local web UI in your browser.")
def web_command(
    host: Annotated[str, typer.Option("--host", help="Address to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Port to listen on.")] = 8756,
) -> None:
    """Start the speechcollector web UI (paste a URL or upload a file in the browser)."""
    from speechcollector.webui import run_server

    try:
        run_server(host=host, port=port)
    except KeyboardInterrupt:
        raise typer.Exit(code=130) from None
    except OSError as exc:
        err_console.print(f"[red]Could not start the web server:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command("dataset", help="Build a TTS/STT training dataset from media (clips + manifests).")
def dataset(
    source: Annotated[
        str,
        typer.Argument(
            metavar="SOURCE",
            help="YouTube video/playlist URL, podcast RSS feed, or local file.",
            show_default=False,
        ),
    ],
    out: Annotated[
        Path, typer.Option("--out", help="Dataset output directory.")
    ] = _DEFAULT_DATASET_DIR,
    fmt: Annotated[
        list[DatasetFormat] | None,
        typer.Option(
            "--format",
            help="Index format(s), repeatable. Default: ljspeech + jsonl.",
            show_default=False,
        ),
    ] = None,
    sample_rate: Annotated[
        int, typer.Option("--sample-rate", help="Output WAV rate (Hz): 22050 TTS, 16000 ASR.")
    ] = 22050,
    segment_min: Annotated[
        float, typer.Option("--segment-min", help="Minimum clip length (seconds).")
    ] = 1.0,
    segment_max: Annotated[
        float, typer.Option("--segment-max", help="Maximum clip length (seconds).")
    ] = 15.0,
    language: Annotated[
        str | None,
        typer.Option(
            "--lang", help="Target language (transcription + filter).", show_default=False
        ),
    ] = None,
    model: Annotated[
        ModelSize, typer.Option("--model", help="Transcription model.")
    ] = _DEFAULT_MODEL,
    api: Annotated[
        str | None,
        typer.Option(
            "--api",
            help="Remote STT API profile name (overrides --model). Looks up <name>.json.",
            show_default=False,
        ),
    ] = None,
    api_config: Annotated[
        Path | None,
        typer.Option(
            "--api-config",
            help="Path to a remote STT API profile JSON (overrides --model).",
            show_default=False,
        ),
    ] = None,
    vad: Annotated[
        bool, typer.Option("--vad/--no-vad", help="Voice-activity filter (Whisper).")
    ] = True,
    normalize: Annotated[
        bool, typer.Option("--normalize", help="EBU R128 loudness-normalize each clip.")
    ] = False,
    pad: Annotated[
        float,
        typer.Option("--pad", help="Edge padding (seconds) added to each side of a clip."),
    ] = 0.1,
    no_filter: Annotated[
        bool, typer.Option("--no-filter", help="Disable the quality filters (keep every clip).")
    ] = False,
    diarize: Annotated[
        bool,
        typer.Option("--diarize", help="Label speakers (needs the diarize extra + an HF token)."),
    ] = False,
    per_speaker: Annotated[
        bool, typer.Option("--per-speaker", help="Diarize and also emit a per-speaker index.")
    ] = False,
    dominant_speaker: Annotated[
        bool,
        typer.Option("--dominant-speaker", help="Diarize and keep only the most-spoken speaker."),
    ] = False,
    hf_token: Annotated[
        str | None,
        typer.Option(
            "--hf-token", help="HF token for diarization (else HF_TOKEN).", show_default=False
        ),
    ] = None,
    min_speakers: Annotated[
        int | None, typer.Option("--min-speakers", help="Diarization hint.", show_default=False)
    ] = None,
    max_speakers: Annotated[
        int | None, typer.Option("--max-speakers", help="Diarization hint.", show_default=False)
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Batch: cap items from a playlist/feed.", show_default=False),
    ] = None,
    no_resume: Annotated[
        bool, typer.Option("--no-resume", help="Batch: don't reuse a previous run's clips.")
    ] = False,
) -> None:
    """Build a TTS/STT dataset from SOURCE into --out."""
    from speechcollector.dataset.build import (
        build_dataset_from_feed,
        build_dataset_from_file,
        build_dataset_from_playlist,
        build_dataset_from_youtube,
    )
    from speechcollector.dataset.models import DatasetConfig, DiarizeConfig, FilterConfig

    language = language or None  # treat --lang "" like an omitted flag (auto-detect)
    mode = "per_speaker" if per_speaker else "dominant" if dominant_speaker else "tag"
    config_kwargs: dict = {} if not fmt else {"formats": [f.value for f in fmt]}
    config = DatasetConfig(
        out_dir=out,
        sample_rate=sample_rate,
        segment_min_s=segment_min,
        segment_max_s=segment_max,
        normalize=normalize,
        edge_pad_s=pad,
        filters=FilterConfig(enabled=not no_filter, target_language=language or "en"),
        diarize=DiarizeConfig(
            enabled=diarize or per_speaker or dominant_speaker,
            mode=mode,
            hf_token=hf_token,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        ),
        **config_kwargs,
    )
    m = _resolve_model_choice(model, api, api_config)
    try:
        path = Path(source).expanduser()
        if path.is_file():
            report: object = _with_status(
                f"Building dataset from {path.name}",
                lambda: build_dataset_from_file(
                    path, config=config, model_size=m, language=language, vad_filter=vad
                ),
            )
        elif extract_playlist_id(source) is not None:
            report = build_dataset_from_playlist(
                source,
                config=config,
                model_size=m,
                language=language,
                vad_filter=vad,
                limit=limit,
                on_item=_announce_source,
                resume=not no_resume,
            )
        elif extract_video_id(source) is not None:
            report = _with_status(
                "Building dataset",
                lambda: build_dataset_from_youtube(
                    source, config=config, model_size=m, language=language, vad_filter=vad
                ),
            )
        elif source.startswith(("http://", "https://")):
            report = build_dataset_from_feed(
                source,
                config=config,
                model_size=m,
                language=language,
                vad_filter=vad,
                limit=limit,
                on_item=_announce_source,
                resume=not no_resume,
            )
        else:
            report = _with_status(
                f"Building dataset from {path.name}",
                lambda: build_dataset_from_file(
                    path, config=config, model_size=m, language=language, vad_filter=vad
                ),
            )
        _print_dataset_summary(report)
    except KeyboardInterrupt:
        err_console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(code=130) from None
    except CollectorError as exc:
        _print_error(exc)
        raise typer.Exit(code=1) from exc


def _with_status(label: str, func: Callable[[], object]) -> object:
    """Run ``func`` under a rich status spinner (single-source dataset builds)."""
    with console.status(f"[bold]{label}…", spinner="dots"):
        return func()


def _announce_source(index: int, total: int, src: object) -> None:
    label = getattr(src, "label", "")
    console.print(f"[bold blue]\\[{index}/{total}][/bold blue] {label}")


def _print_dataset_summary(report: object) -> None:
    """Print a dataset build summary (works for single and combined reports)."""
    clip_count = report.clip_count  # type: ignore[attr-defined]
    duration = format_timestamp(report.total_duration_s)  # type: ignore[attr-defined]
    dropped = report.dropped_count  # type: ignore[attr-defined]
    body = (
        f"[bold green]✓[/bold green] {clip_count} clips · {duration} · "
        f"dropped {dropped}\n→ [bold]{report.out_dir}/[/bold]"  # type: ignore[attr-defined]
    )
    sources = getattr(report, "sources", None)
    if sources is not None:
        ok = sum(1 for s in sources if s.ok)
        body += f"\n[dim]{ok}/{len(sources)} sources[/dim]"
    console.print(Panel.fit(body, title="speechcollector dataset", border_style="green"))
    if report.drops_by_reason:  # type: ignore[attr-defined]
        console.print(f"[dim]dropped by: {report.drops_by_reason}[/dim]")  # type: ignore[attr-defined]
    for warning in report.warnings:  # type: ignore[attr-defined]
        console.print(f"[yellow]![/yellow] {escape(warning)}")  # may contain "speechcollector[diarize]"


@app.command("ingest", help=PITCH)
def ingest(
    source: Annotated[
        str,
        typer.Argument(
            metavar="SOURCE",
            help="YouTube video/playlist URL, podcast RSS feed, or local file.",
            show_default=False,
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--output",
            help="Output file for a single source. Default ./<id>.md",
            show_default=False,
        ),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Output directory for batch (playlist/feed) ingestion."),
    ] = _DEFAULT_OUTPUT_DIR,
    language: Annotated[
        str | None,
        typer.Option(
            "--lang",
            help="Language code. Captions default to English; transcription auto-detects.",
            show_default=False,
        ),
    ] = None,
    transcribe: Annotated[
        bool,
        typer.Option(
            "--transcribe", help="Force local whisper transcription even when captions exist."
        ),
    ] = False,
    model: Annotated[
        ModelSize,
        typer.Option(
            "--model",
            help="Transcription model: auto (fastest available), parakeet, or a whisper size.",
        ),
    ] = _DEFAULT_MODEL,
    api: Annotated[
        str | None,
        typer.Option(
            "--api",
            help="Remote STT API profile name (overrides --model). Looks up <name>.json.",
            show_default=False,
        ),
    ] = None,
    api_config: Annotated[
        Path | None,
        typer.Option(
            "--api-config",
            help="Path to a remote STT API profile JSON (overrides --model).",
            show_default=False,
        ),
    ] = None,
    vad: Annotated[
        bool,
        typer.Option(
            "--vad/--no-vad",
            help="Voice-activity filter: on for speech (default), --no-vad for music/songs.",
        ),
    ] = True,
    write_json: Annotated[
        bool,
        typer.Option("--json", help="Also write a .json sidecar matching the Transcript schema."),
    ] = False,
    latest: Annotated[
        bool,
        typer.Option("--latest", help="Batch: ingest only the most recent item."),
    ] = False,
    episode: Annotated[
        int | None,
        typer.Option(
            "--episode", help="Batch: ingest only item number N (1-indexed).", show_default=False
        ),
    ] = None,
    all_: Annotated[
        bool,
        typer.Option("--all", help="Batch: ingest every item (cap with --limit)."),
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit", help="Batch: cap the number ingested with --all.", show_default=False
        ),
    ] = None,
) -> None:
    """Ingest SOURCE into timestamped, LLM-ready markdown."""
    opts = _Options(
        output=output,
        output_dir=output_dir,
        language=language,
        # Remote APIs always transcribe; force the transcription path.
        transcribe=transcribe or api is not None or api_config is not None,
        model=_resolve_model_choice(model, api, api_config),
        vad=vad,
        write_json=write_json,
        latest=latest,
        episode=episode,
        all_=all_,
        limit=limit,
    )
    try:
        path = Path(source).expanduser()
        video_id = extract_video_id(source)
        if path.is_file():
            _run_single(_ingest_file(path, opts), path.stem, opts)
        elif extract_playlist_id(source) is not None:
            _run_playlist(source, opts)
        elif video_id is not None:
            _run_single(_ingest_youtube_source(source, opts), video_id, opts)
        elif source.startswith(("http://", "https://")):
            _run_feed(source, opts)
        else:
            _run_single(_ingest_file(path, opts), path.stem, opts)
    except KeyboardInterrupt:
        err_console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(code=130) from None
    except CollectorError as exc:
        _print_error(exc)
        raise typer.Exit(code=1) from exc


@dataclass(frozen=True)
class _Options:
    """Resolved CLI options, passed explicitly so a typo can't become a KeyError."""

    output: Path | None
    output_dir: Path
    language: str | None
    transcribe: bool
    model: str
    vad: bool
    write_json: bool
    latest: bool
    episode: int | None
    all_: bool
    limit: int | None


# --- Single-source ingestion ---------------------------------------------


def _run_single(document: Document, default_name: str, opts: _Options) -> None:
    destination = opts.output if opts.output is not None else Path(f"./{default_name}.md")
    written = _write_document(document, destination, write_json=opts.write_json)
    _print_success(document, written)


def _ingest_file(path: Path, opts: _Options) -> Document:
    with _progress(f"Transcribing {path.name} with '{opts.model}'") as cb:
        return ingest_file(
            path,
            model_size=opts.model,
            language=opts.language,
            vad_filter=opts.vad,
            on_progress=cb,
        )


def _ingest_youtube_source(url: str, opts: _Options) -> Document:
    """Captions path, or whisper — forced by --transcribe or auto on no captions."""
    if opts.transcribe:
        return _transcribe_youtube(url, opts, forced=True)
    video_id = extract_video_id(url) or url
    try:
        with console.status(f"[bold]Fetching captions for {video_id}…", spinner="dots"):
            return ingest_youtube(url, language=opts.language or "en")
    except NoCaptionsError:
        console.print("[yellow]No captions found.[/yellow] Falling back to local transcription.")
        return _transcribe_youtube(url, opts, forced=False)


def _transcribe_youtube(url: str, opts: _Options, *, forced: bool) -> Document:
    why = "Transcribing" if forced else "Downloading audio, then transcribing"
    console.print(f"[dim]{why} locally with '{opts.model}'. This can take a while.[/dim]")
    with _progress("Transcribing audio") as cb:
        return ingest_youtube_transcribe(
            url,
            model_size=opts.model,
            language=opts.language,
            vad_filter=opts.vad,
            on_progress=cb,
        )


# --- Batch ingestion (playlists & feeds) ----------------------------------


def _run_playlist(url: str, opts: _Options) -> None:
    with console.status("[bold]Listing playlist…", spinner="dots"):
        title, entries = fetch_playlist(url)
    items = [
        BatchItem(title=e.title, slug=e.video_id, ingest=_youtube_entry_ingester(e, opts))
        for e in entries
    ]
    _run_batch_or_list(items, title, "video", opts, episodes=None)


def _run_feed(url: str, opts: _Options) -> None:
    with console.status("[bold]Fetching feed…", spinner="dots"):
        feed = fetch_feed(url)
    items = [
        BatchItem(
            title=ep.title,
            slug=slugify(ep.title, fallback=f"episode-{i}"),
            ingest=_episode_ingester(ep, feed.title, opts),
        )
        for i, ep in enumerate(feed.episodes, start=1)
    ]
    _run_batch_or_list(items, feed.title, "episode", opts, episodes=feed.episodes)


def _youtube_entry_ingester(entry: PlaylistEntry, opts: _Options) -> Callable[[], Document]:
    def _ingest() -> Document:
        if opts.transcribe:
            return ingest_youtube_transcribe(
                entry.url, model_size=opts.model, language=opts.language, vad_filter=opts.vad
            )
        try:
            return ingest_youtube(entry.url, language=opts.language or "en")
        except NoCaptionsError:
            return ingest_youtube_transcribe(
                entry.url, model_size=opts.model, language=opts.language, vad_filter=opts.vad
            )

    return _ingest


def _episode_ingester(episode: Episode, show: str, opts: _Options) -> Callable[[], Document]:
    def _ingest() -> Document:
        return ingest_episode(
            episode, show, model_size=opts.model, language=opts.language, vad_filter=opts.vad
        )

    return _ingest


def _run_batch_or_list(
    items: list[BatchItem],
    source_title: str,
    noun: str,
    opts: _Options,
    *,
    episodes: list[Episode] | None,
) -> None:
    ensure_unique_slugs(items)  # distinct items must not overwrite each other's output
    selected = select(
        items, latest=opts.latest, episode=opts.episode, all_=opts.all_, limit=opts.limit
    )
    if not selected:
        _print_listing(items, source_title, noun, episodes=episodes)
        return

    _make_output_dir(opts.output_dir)

    def write(document: Document, slug: str) -> Path:
        return _write_document(document, opts.output_dir / f"{slug}.md", write_json=opts.write_json)

    def announce(index: int, total: int, item: BatchItem) -> None:
        console.print(f"[bold blue]\\[{index}/{total}][/bold blue] {item.title}")

    results = run_batch(selected, write, on_item=announce)
    _print_summary(results, opts.output_dir)


def _make_output_dir(output_dir: Path) -> None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputWriteError(
            f"Could not create output directory {output_dir}: {exc.strerror or exc}",
            hint="Pick a writable --output-dir whose path is not an existing file.",
        ) from exc


# --- Output writing -------------------------------------------------------


def _write_document(document: Document, destination: Path, *, write_json: bool) -> Path:
    """Write the markdown (and optional JSON sidecar); return the markdown path."""
    _write_text(destination, render_markdown(document))
    if write_json:
        sidecar = destination.with_suffix(".json")
        if sidecar == destination:  # destination already ends in .json — don't clobber it
            sidecar = destination.with_name(destination.name + ".json")
        transcript = Transcript.from_document(document)
        _write_text(sidecar, transcript.model_dump_json(indent=2) + "\n")
    return destination


def _write_text(destination: Path, text: str) -> None:
    try:
        destination.write_text(text, encoding="utf-8")
    except IsADirectoryError as exc:
        raise OutputWriteError(
            f"The output path is a directory, not a file: {destination}",
            hint="Pass a file path to -o/--output, e.g. -o transcript.md",
        ) from exc
    except OSError as exc:
        raise OutputWriteError(
            f"Could not write to {destination}: {exc.strerror or exc}",
            hint="Check that the parent directory exists and is writable, then try again.",
        ) from exc


# --- Presentation ---------------------------------------------------------


@contextmanager
def _progress(label: str) -> Iterator:
    """A rich progress bar driven by an (processed_s, total_s) callback."""
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    with progress:
        task: TaskID = progress.add_task(label, total=None)

        def on_progress(processed_s: float, total_s: float) -> None:
            if total_s > 0:
                progress.update(task, total=total_s, completed=processed_s)

        yield on_progress


def _print_listing(
    items: list[BatchItem],
    source_title: str,
    noun: str,
    *,
    episodes: list[Episode] | None,
) -> None:
    table = Table(title=source_title, title_style="bold")
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column(noun.capitalize())
    if episodes is not None:
        table.add_column("Duration", justify="right", style="dim")
    for index, item in enumerate(items, start=1):
        if episodes is not None:
            dur = (
                format_timestamp(episodes[index - 1].duration_s)
                if episodes[index - 1].duration_s
                else "—"
            )
            table.add_row(str(index), item.title, dur)
        else:
            table.add_row(str(index), item.title)
    console.print(table)
    console.print(
        f"[dim]{len(items)} {noun}(s). Ingest with "
        "--latest, --episode N, or --all [--limit N].[/dim]"
    )


def _print_summary(results: list, output_dir: Path) -> None:
    table = Table(title="speechcollector batch", title_style="bold")
    table.add_column("Status", no_wrap=True)
    table.add_column("Item")
    table.add_column("Output / error", overflow="fold")
    ok_count = 0
    for result in results:
        if result.ok:
            ok_count += 1
            table.add_row("[green]✓[/green]", result.title, str(result.output))
        else:
            table.add_row("[red]✗[/red]", result.title, f"[red]{result.error}[/red]")
    console.print(table)
    failed = len(results) - ok_count
    summary = f"[bold]{ok_count} succeeded[/bold]"
    if failed:
        summary += f", [red]{failed} failed[/red]"
    console.print(f"{summary} · → [bold]{output_dir}/[/bold]")


def _print_success(document: Document, destination: Path) -> None:
    paragraphs = sum(len(section.paragraphs) for section in document.sections)
    console.print(
        Panel.fit(
            f"[bold green]✓[/bold green] {document.meta.title}\n"
            f"[dim]{len(document.sections)} sections · {paragraphs} paragraphs · "
            f"method: {document.method}[/dim]\n"
            f"→ [bold]{destination}[/bold]",
            title="speechcollector",
            border_style="green",
        )
    )


def _print_error(exc: CollectorError) -> None:
    # Escape dynamic text: messages/hints can contain brackets (e.g. "speechcollector[diarize]")
    # that Rich would otherwise parse as markup.
    body = f"[red]{escape(exc.message)}[/red]"
    if exc.hint:
        body += f"\n\n[bold]Try:[/bold] {escape(exc.hint)}"
    err_console.print(Panel.fit(body, title="speechcollector error", border_style="red"))
