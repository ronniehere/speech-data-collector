"""A polished, dependency-free web UI for speechcollector.

One static HTML page (embedded below) talks to two JSON endpoints that wrap the
existing pipeline. The design is Swiss/International style: white, Helvetica, a
strict grid, hairline rules, generous whitespace, monochrome with one restrained
accent. A sidebar holds history, the top bar holds the model selector, and a
composer takes a YouTube URL or an uploaded audio/video file, returning clean
timestamped markdown with a live preview.

Built on the standard-library HTTP server so it adds no runtime dependencies.
File uploads are sent as the raw request body (filename in a query param),
sidestepping multipart parsing (and ``cgi``, which is gone in Python 3.13).
History lives entirely in the browser (localStorage); the server is stateless.
"""

import base64
import io
import json
import tempfile
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from speechcollector.errors import CollectorError, InvalidSourceError
from speechcollector.models import Document
from speechcollector.pipeline import (
    NoCaptionsError,
    ingest_file,
    ingest_youtube,
    ingest_youtube_transcribe,
)
from speechcollector.render import render_markdown
from speechcollector.timefmt import format_timestamp
from speechcollector.transcribe import DEFAULT_MODEL
from speechcollector.youtube import extract_playlist_id, extract_video_id

# Upload ceiling so a stray huge request can't exhaust memory (raw body is read
# fully into RAM). 1 GiB comfortably covers multi-hour audio.
_MAX_UPLOAD_BYTES = 1024 * 1024 * 1024


# --- Pipeline glue (network/transcription happens downstream) --------------


def process_url(
    url: str,
    *,
    transcribe: bool = False,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
    vad: bool = True,
) -> Document:
    """Ingest a single YouTube video URL (captions, or whisper/parakeet).

    Mirrors the CLI's single-video behaviour: captions first, falling back to
    local transcription when there are none (or when ``transcribe`` forces it).
    """
    url = url.strip()
    if not url:
        raise InvalidSourceError("No URL provided.", hint="Paste a YouTube video link.")
    if extract_playlist_id(url) is not None:
        raise InvalidSourceError(
            "Playlists and podcast feeds aren't supported in the web UI yet.",
            hint="Use the CLI for batch sources, e.g. speechcollector <url> --all.",
        )
    if extract_video_id(url) is None:
        raise InvalidSourceError(
            f"Not a recognized YouTube video URL: {url}",
            hint="Paste a single YouTube video link, or upload a file instead.",
        )
    # Remote API profiles always go through the transcription path.
    force = transcribe or model.startswith("api:")
    if force:
        return ingest_youtube_transcribe(url, model_size=model, language=language, vad_filter=vad)
    try:
        return ingest_youtube(url, language=language or "en")
    except NoCaptionsError:
        return ingest_youtube_transcribe(url, model_size=model, language=language, vad_filter=vad)


def process_file(
    name: str,
    data: bytes,
    *,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
    vad: bool = True,
) -> Document:
    """Transcribe uploaded file bytes, preserving the original name for the title."""
    safe_name = Path(name).name or "upload"
    with tempfile.TemporaryDirectory(prefix="speechcollector-web-") as tmp:
        path = Path(tmp) / safe_name
        path.write_bytes(data)
        return ingest_file(path, model_size=model, language=language, vad_filter=vad)


def _document_payload(doc: Document) -> dict:
    """Shape a Document into the JSON the page renders."""
    return {
        "ok": True,
        "markdown": render_markdown(doc),
        "title": doc.meta.title,
        "method": doc.method,
        "language": doc.language,
        "duration": format_timestamp(doc.meta.duration_s),
        "sections": len(doc.sections),
    }


# --- Dataset mode (single source -> a downloadable zip) --------------------

_DEFAULT_DATASET_FORMATS = ["ljspeech", "jsonl"]


def _zip_dir(root: Path) -> bytes:
    """Zip a directory tree into in-memory bytes (relative paths)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(root).as_posix())
    return buf.getvalue()


def _dataset_payload(out_dir: Path, report: object) -> dict:
    """Shape a build report + the zipped dataset into the JSON the page downloads."""
    return {
        "ok": True,
        "dataset": True,
        "clips": report.clip_count,  # type: ignore[attr-defined]
        "duration": format_timestamp(report.total_duration_s),  # type: ignore[attr-defined]
        "dropped": report.dropped_count,  # type: ignore[attr-defined]
        "warnings": report.warnings,  # type: ignore[attr-defined]
        "zip_name": f"{out_dir.name}.zip",
        "zip_b64": base64.b64encode(_zip_dir(out_dir)).decode("ascii"),
    }


def _dataset_config(out_dir: Path, opts: dict):
    from speechcollector.dataset.models import DatasetConfig

    formats = opts.get("formats") or _DEFAULT_DATASET_FORMATS
    try:
        return DatasetConfig(
            out_dir=out_dir,
            formats=list(formats),
            sample_rate=int(opts.get("sample_rate") or 22050),
            segment_min_s=float(opts.get("segment_min") or 1.0),
            segment_max_s=float(opts.get("segment_max") or 15.0),
        )
    except (ValueError, TypeError) as exc:  # bad numeric option -> a 400, not an opaque 500
        raise InvalidSourceError(
            f"Invalid dataset option: {exc}",
            hint="sample-rate must be a positive integer and segment bounds positive numbers.",
        ) from exc


def build_url_dataset(url: str, *, opts: dict) -> dict:
    """Build a dataset from one YouTube video URL and return a zip payload."""
    from speechcollector.dataset.build import build_dataset_from_youtube

    url = url.strip()
    if not url:
        raise InvalidSourceError("No URL provided.", hint="Paste a YouTube video link.")
    if extract_playlist_id(url) is not None:
        raise InvalidSourceError(
            "Playlists aren't supported in the web UI's dataset mode.",
            hint="Use the CLI for batch sources, e.g. speechcollector dataset <playlist-url>.",
        )
    if extract_video_id(url) is None:
        raise InvalidSourceError(
            f"Not a recognized YouTube video URL: {url}",
            hint="Paste a single YouTube video link, or upload a file instead.",
        )
    with tempfile.TemporaryDirectory(prefix="speechcollector-web-ds-") as tmp:
        out_dir = Path(tmp) / "speechcollector-dataset"
        report = build_dataset_from_youtube(
            url,
            config=_dataset_config(out_dir, opts),
            model_size=str(opts.get("model") or DEFAULT_MODEL),
            language=(opts.get("lang") or None),
            vad_filter=bool(opts.get("vad", True)),
        )
        return _dataset_payload(out_dir, report)


def build_file_dataset(name: str, data: bytes, *, opts: dict) -> dict:
    """Build a dataset from an uploaded file and return a zip payload."""
    from speechcollector.dataset.build import build_dataset_from_file

    safe_name = Path(name).name or "upload"
    with tempfile.TemporaryDirectory(prefix="speechcollector-web-ds-") as tmp:
        source = Path(tmp) / safe_name
        source.write_bytes(data)
        out_dir = Path(tmp) / "speechcollector-dataset"
        report = build_dataset_from_file(
            source,
            config=_dataset_config(out_dir, opts),
            model_size=str(opts.get("model") or DEFAULT_MODEL),
            language=(opts.get("lang") or None),
            vad_filter=bool(opts.get("vad", True)),
        )
        return _dataset_payload(out_dir, report)


# --- HTTP server ----------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    server_version = "speechcollector-webui"

    def log_message(self, *args: object) -> None:
        pass  # quiet by default — no request logging to stderr

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send_html(render_page())
        elif path == "/health":
            self._send_json({"ok": True})
        else:
            self._send_json({"ok": False, "error": "Not found."}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/url":
                payload = _document_payload(self._handle_url())
            elif parsed.path == "/api/file":
                payload = _document_payload(self._handle_file(parse_qs(parsed.query)))
            elif parsed.path == "/api/dataset":
                payload = self._handle_dataset_url()
            elif parsed.path == "/api/dataset-file":
                payload = self._handle_dataset_file(parse_qs(parsed.query))
            else:
                self._send_json({"ok": False, "error": "Not found."}, status=404)
                return
        except CollectorError as exc:
            self._send_json({"ok": False, "error": exc.message, "hint": exc.hint}, status=400)
        except Exception as exc:  # surface anything unexpected as a 500
            self._send_json({"ok": False, "error": str(exc)}, status=500)
        else:
            self._send_json(payload)

    # -- request helpers --

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length > _MAX_UPLOAD_BYTES:
            raise InvalidSourceError(
                "Upload is too large.",
                hint="The web UI accepts files up to 1 GiB; use the CLI for bigger ones.",
            )
        return self.rfile.read(length) if length else b""

    def _handle_url(self) -> Document:
        body = json.loads(self._read_body() or b"{}")
        return process_url(
            str(body.get("url", "")),
            transcribe=bool(body.get("transcribe", False)),
            model=str(body.get("model") or DEFAULT_MODEL),
            language=(body.get("lang") or None),
            vad=bool(body.get("vad", True)),
        )

    def _handle_file(self, query: dict[str, list[str]]) -> Document:
        data = self._read_body()
        if not data:
            raise InvalidSourceError("No file received.", hint="Choose an audio/video file.")
        name = (query.get("name") or ["upload"])[0]
        model = (query.get("model") or [DEFAULT_MODEL])[0]
        lang = (query.get("lang") or [""])[0] or None
        vad = (query.get("vad") or ["1"])[0] != "0"
        return process_file(name, data, model=model, language=lang, vad=vad)

    def _handle_dataset_url(self) -> dict:
        body = json.loads(self._read_body() or b"{}")
        return build_url_dataset(str(body.get("url", "")), opts=body)

    def _handle_dataset_file(self, query: dict[str, list[str]]) -> dict:
        data = self._read_body()
        if not data:
            raise InvalidSourceError("No file received.", hint="Choose an audio/video file.")
        name = (query.get("name") or ["upload"])[0]
        opts: dict = {
            "model": (query.get("model") or [DEFAULT_MODEL])[0],
            "lang": (query.get("lang") or [""])[0] or None,
            "vad": (query.get("vad") or ["1"])[0] != "0",
            "sample_rate": (query.get("sample_rate") or ["22050"])[0],
            "segment_min": (query.get("segment_min") or ["1.0"])[0],
            "segment_max": (query.get("segment_max") or ["15.0"])[0],
        }
        return build_file_dataset(name, data, opts=opts)

    # -- response helpers --

    def _send_json(self, payload: dict, *, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def make_server(host: str, port: int) -> ThreadingHTTPServer:
    """Create (but do not start) the threaded speechcollector web server."""
    return ThreadingHTTPServer((host, port), _Handler)


def run_server(host: str = "127.0.0.1", port: int = 8756) -> None:
    """Serve the web UI until interrupted."""
    httpd = make_server(host, port)
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    print(f"speechcollector web UI -> http://{shown}:{port}  (Ctrl-C to stop)", flush=True)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()


# --- The page (single file: HTML + CSS + JS, no external assets) -----------

_API_OPTIONS_MARKER = "<!--API_OPTIONS-->"


def render_page() -> str:
    """HTML page with discovered remote API profiles injected into the model select."""
    from html import escape

    from speechcollector.remote import list_api_profiles

    options = "".join(
        f'<option value="api:{escape(name)}">API: {escape(name)}</option>\n'
        for name in list_api_profiles()
    )
    return _PAGE.replace(_API_OPTIONS_MARKER, options)


_PAGE = r"""<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Speech Data Collector</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Outfit:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#eef2f0; --bg2:#e4ebe7; --panel:#f7faf8; --fg:#14201c; --fg2:#2f3f39;
    --muted:#6a7a73; --line:#c9d4ce; --line2:#1a2e27; --hover:#e2ebe6;
    --accent:#0f7a66; --accent2:#d97706; --surface:#ffffffcc;
    --maxw:780px; --radius:14px; --font:"Outfit",system-ui,sans-serif;
    --serif:"Source Serif 4",Georgia,serif; --mono:"IBM Plex Mono",ui-monospace,monospace;
  }
  html[data-theme="dark"]{
    --bg:#0f1614; --bg2:#15201c; --panel:#121b18; --fg:#e8f0ec; --fg2:#c5d4cd;
    --muted:#7f948a; --line:#24332d; --line2:#d7e6df; --hover:#1a2621;
    --accent:#2dd4a8; --accent2:#f0b429; --surface:#15201ccc;
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{margin:0;color:var(--fg);font:14.5px/1.55 var(--font);
    background:
      radial-gradient(1200px 600px at 12% -10%, color-mix(in srgb, var(--accent) 18%, transparent), transparent 55%),
      radial-gradient(900px 500px at 100% 0%, color-mix(in srgb, var(--accent2) 14%, transparent), transparent 50%),
      linear-gradient(165deg, var(--bg) 0%, var(--bg2) 100%);
    -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
  body::before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.35;z-index:0;
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='120' height='120' filter='url(%23n)' opacity='.35'/%3E%3C/svg%3E");
    mix-blend-mode:soft-light}
  button{font:inherit;cursor:pointer;color:inherit;border:0;background:none}
  svg{display:block}
  .lbl{font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:500}
  ::-webkit-scrollbar{width:10px;height:10px}
  ::-webkit-scrollbar-thumb{background:var(--line);border:3px solid transparent;background-clip:padding-box;border-radius:8px}
  ::selection{background:var(--accent);color:#fff}

  .app{position:relative;z-index:1;display:flex;height:100vh;overflow:hidden}

  /* sidebar */
  .sidebar{width:260px;flex:0 0 260px;display:flex;flex-direction:column;
    background:color-mix(in srgb, var(--panel) 88%, transparent);
    backdrop-filter:blur(16px);border-right:1px solid var(--line);
    transition:margin-left .22s cubic-bezier(.2,.8,.2,1)}
  .app.collapsed .sidebar{margin-left:-261px}
  .side-top{padding:22px 18px 16px;display:flex;flex-direction:column;gap:16px}
  .brand{display:flex;align-items:center;gap:10px;font-weight:700;font-size:15px;letter-spacing:-.03em}
  .brand .mark{width:18px;height:18px;border-radius:5px;background:
    linear-gradient(135deg, var(--accent), color-mix(in srgb, var(--accent) 40%, var(--accent2)));
    box-shadow:0 0 0 3px color-mix(in srgb, var(--accent) 18%, transparent);
    animation:pulseMark 2.8s ease-in-out infinite}
  @keyframes pulseMark{0%,100%{transform:scale(1)}50%{transform:scale(1.06)}}
  .new-btn{display:flex;align-items:center;justify-content:center;gap:8px;padding:11px 12px;
    border-radius:10px;background:var(--fg);color:var(--bg);font-size:12px;letter-spacing:.06em;
    font-weight:600;transition:transform .14s ease, background .14s ease}
  .new-btn:hover{background:var(--accent);transform:translateY(-1px)}
  .new-btn svg{width:13px;height:13px}
  .hist{flex:1;overflow:auto;padding:4px 12px 14px}
  .hist .lbl{padding:12px 8px 8px;display:block}
  .hist-item{position:relative;padding:11px 28px 11px 12px;cursor:pointer;border-radius:10px;
    margin-bottom:4px;transition:background .12s ease, transform .12s ease}
  .hist-item:hover{background:var(--hover)}
  .hist-item.active{background:color-mix(in srgb, var(--accent) 12%, var(--hover));
    box-shadow:inset 3px 0 0 var(--accent)}
  .hist-item .t{font-size:13px;font-weight:500;color:var(--fg);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .hist-item .s{font-size:11px;color:var(--muted);margin-top:2px;font-family:var(--mono)}
  .hist-item .del{position:absolute;right:8px;top:50%;transform:translateY(-50%);opacity:0;color:var(--muted);
    font-size:15px;line-height:1;padding:3px 5px;transition:.12s}
  .hist-item:hover .del{opacity:1}
  .hist-item .del:hover{color:var(--accent2)}
  .hist-empty{color:var(--muted);font-size:12.5px;padding:16px 10px;line-height:1.65}
  .side-foot{padding:14px 18px;border-top:1px solid var(--line);font-size:10.5px;color:var(--muted);
    display:flex;justify-content:space-between;letter-spacing:.1em;font-family:var(--mono)}

  /* main */
  .main{flex:1;display:flex;flex-direction:column;min-width:0;position:relative}
  .topbar{height:60px;flex:0 0 60px;display:flex;align-items:center;gap:14px;padding:0 22px;
    border-bottom:1px solid color-mix(in srgb, var(--line) 80%, transparent);
    background:color-mix(in srgb, var(--panel) 55%, transparent);backdrop-filter:blur(12px)}
  .icon-btn{width:34px;height:34px;display:grid;place-items:center;color:var(--fg2);border-radius:9px;transition:.12s}
  .icon-btn:hover{color:var(--fg);background:var(--hover)}
  .icon-btn svg{width:18px;height:18px}
  .top-title{flex:1;min-width:0;font-size:13.5px;font-weight:500;letter-spacing:-.01em;color:var(--fg);
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .model{display:flex;align-items:center;gap:8px}
  .model .lbl{display:none}
  .model select{border:1px solid var(--line);background:var(--panel);color:var(--fg);font:inherit;font-size:12.5px;
    padding:8px 28px 8px 12px;outline:none;border-radius:9px;cursor:pointer;
    -webkit-appearance:none;appearance:none;
    background-image:linear-gradient(45deg,transparent 50%,var(--fg2) 50%),linear-gradient(135deg,var(--fg2) 50%,transparent 50%);
    background-position:calc(100% - 14px) 15px,calc(100% - 9px) 15px;background-size:5px 5px,5px 5px;background-repeat:no-repeat;
    transition:border-color .12s}
  .model select:hover{border-color:var(--accent)}

  .scroll{flex:1;overflow:auto}
  .stream{max-width:var(--maxw);margin:0 auto;padding:36px 28px 170px;position:relative}

  /* hero */
  .hero{position:absolute;inset:60px 0 0;display:flex;flex-direction:column;align-items:center;justify-content:center;
    text-align:center;padding:28px;animation:heroIn .55s cubic-bezier(.2,.8,.2,1) both}
  @keyframes heroIn{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
  .hero .wave{display:flex;align-items:flex-end;gap:3px;height:36px;margin-bottom:28px}
  .hero .wave span{width:4px;border-radius:3px;background:var(--accent);
    animation:wave 1.1s ease-in-out infinite}
  .hero .wave span:nth-child(1){height:12px;animation-delay:0s}
  .hero .wave span:nth-child(2){height:24px;animation-delay:.08s}
  .hero .wave span:nth-child(3){height:34px;animation-delay:.16s}
  .hero .wave span:nth-child(4){height:20px;animation-delay:.24s}
  .hero .wave span:nth-child(5){height:28px;animation-delay:.32s}
  .hero .wave span:nth-child(6){height:14px;animation-delay:.4s}
  .hero .wave span:nth-child(7){height:22px;animation-delay:.48s}
  @keyframes wave{0%,100%{transform:scaleY(.55);opacity:.65}50%{transform:scaleY(1);opacity:1}}
  .hero .eyebrow{margin-bottom:12px}
  .hero h1{margin:0 0 12px;font-family:var(--serif);font-size:clamp(34px,5vw,52px);line-height:1.05;
    font-weight:600;letter-spacing:-.03em}
  .hero p{margin:0 0 28px;color:var(--muted);max-width:420px;font-size:15px;line-height:1.55}
  .chips{display:flex;gap:10px;flex-wrap:wrap;justify-content:center}
  .chip{padding:11px 18px;font-size:13px;font-weight:500;color:var(--fg);border-radius:999px;
    border:1px solid var(--line);background:var(--surface);backdrop-filter:blur(8px);
    transition:background .14s, border-color .14s, transform .14s}
  .chip:hover{background:var(--accent);border-color:var(--accent);color:#fff;transform:translateY(-1px)}

  /* messages */
  .msg{display:grid;grid-template-columns:72px 1fr;gap:8px;padding:28px 0;
    border-top:1px solid color-mix(in srgb, var(--line) 70%, transparent);animation:fadeUp .35s ease both}
  .msg:first-child{border-top:0}
  @keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
  .msg .role{font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);padding-top:4px;font-weight:600}
  .msg.bot .role{color:var(--accent)}
  .msg .body{min-width:0}
  .msg.user .src{font-size:14.5px;color:var(--fg);word-break:break-all;font-weight:500}
  .doc{border:1px solid var(--line);border-radius:var(--radius);background:var(--surface);
    backdrop-filter:blur(10px);overflow:hidden;box-shadow:0 10px 40px color-mix(in srgb, var(--fg) 4%, transparent)}
  .doc-head{display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding:12px 14px;
    border-bottom:1px solid var(--line);background:color-mix(in srgb, var(--panel) 70%, transparent)}
  .meta{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:11.5px;color:var(--muted);flex:1;min-width:0;align-items:center}
  .meta b{color:var(--fg2);font-weight:600}
  .badge{font-size:10.5px;letter-spacing:.05em;padding:4px 8px;border-radius:6px;
    background:color-mix(in srgb, var(--accent) 14%, transparent);color:var(--accent);font-weight:600;white-space:nowrap}
  .seg,.doc-actions{display:flex;gap:0;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:var(--panel)}
  .seg button,.doc-actions button{padding:6px 12px;color:var(--muted);font-size:11.5px;letter-spacing:.02em;font-weight:500}
  .seg button+button,.doc-actions button+button{border-left:1px solid var(--line)}
  .seg button.on{background:var(--accent);color:#fff}
  .doc-actions button:hover{background:var(--hover);color:var(--fg)}
  .doc-body{padding:22px 24px;max-height:58vh;overflow:auto}
  .raw{white-space:pre-wrap;font:12.5px/1.7 var(--mono);color:var(--fg2)}
  .md h1{font-family:var(--serif);font-size:24px;margin:.1em 0 .55em;font-weight:600;letter-spacing:-.02em}
  .md h2{font-size:12px;margin:1.7em 0 .65em;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
    padding-top:.7em;border-top:1px solid var(--line);color:var(--accent)}
  .md p{margin:.65em 0;color:var(--fg2);font-size:14.5px;line-height:1.65}
  .md hr{border:0;border-top:1px solid var(--line);margin:1.4em 0}
  .md .ts{color:var(--accent);font-weight:600;font-variant-numeric:tabular-nums;font-family:var(--mono);font-size:.92em}
  .md .fm{color:var(--muted);font-size:11.5px;border-left:3px solid var(--accent);padding-left:12px;margin-bottom:18px;
    white-space:pre-wrap;font-family:var(--mono);line-height:1.7}
  .think{display:flex;align-items:center;gap:12px;color:var(--muted);font-size:13.5px;padding:4px 0}
  .spin{width:14px;height:14px;border:2px solid var(--line);border-top-color:var(--accent);border-radius:50%;animation:spin .75s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  .err{color:#c2410c;font-size:13.5px;white-space:pre-wrap;border-left:3px solid var(--accent2);padding-left:12px}

  /* composer */
  .composer-wrap{position:absolute;left:0;right:0;bottom:0;padding:0 28px 24px;
    background:linear-gradient(to top, var(--bg) 55%, transparent)}
  .composer{max-width:var(--maxw);margin:0 auto;border:1px solid var(--line);border-radius:18px;
    background:var(--surface);backdrop-filter:blur(14px);
    box-shadow:0 12px 40px color-mix(in srgb, var(--fg) 6%, transparent);
    transition:border-color .15s, box-shadow .15s}
  .composer.focus{border-color:var(--accent);box-shadow:0 12px 40px color-mix(in srgb, var(--accent) 12%, transparent)}
  .composer.drag{background:color-mix(in srgb, var(--accent) 8%, var(--panel))}
  .file-chip{display:none;align-items:center;gap:10px;padding:10px 14px;border-bottom:1px solid var(--line);
    font-size:12.5px;color:var(--fg2);font-family:var(--mono)}
  .file-chip.show{display:flex}
  .file-chip .x{color:var(--muted);font-size:15px;line-height:1}
  .file-chip .x:hover{color:var(--accent2)}
  .crow{display:flex;align-items:flex-end;gap:6px;padding:8px 8px 8px 12px}
  .attach{width:36px;height:36px;flex:0 0 36px;display:grid;place-items:center;color:var(--fg2);border-radius:10px}
  .attach:hover{color:var(--fg);background:var(--hover)}
  .attach svg{width:17px;height:17px}
  .composer textarea{flex:1;border:0;background:none;color:var(--fg);resize:none;outline:none;font:inherit;font-size:14.5px;
    max-height:168px;padding:10px 4px;line-height:1.5}
  .composer textarea::placeholder{color:var(--muted)}
  .send{width:38px;height:38px;flex:0 0 38px;display:grid;place-items:center;background:var(--accent);color:#fff;
    border-radius:11px;transition:transform .12s, background .12s}
  .send svg{width:17px;height:17px}
  .send:disabled{background:var(--line);color:var(--muted)}
  .send:not(:disabled):hover{background:var(--fg);transform:translateY(-1px)}
  .copts{display:flex;align-items:center;gap:16px;flex-wrap:wrap;padding:10px 14px;border-top:1px solid var(--line);
    color:var(--muted);font-size:11.5px}
  .copts label{display:flex;align-items:center;gap:7px;cursor:pointer;letter-spacing:.04em;font-size:11px;font-weight:500}
  .copts input[type=text]{width:96px;background:var(--panel);border:1px solid var(--line);color:var(--fg);padding:5px 8px;
    font:inherit;font-size:12px;outline:none;border-radius:7px;letter-spacing:0}
  .copts input[type=text]:focus{border-color:var(--accent)}
  .copts input[type=checkbox]{accent-color:var(--accent)}
  .hint{text-align:center;color:var(--muted);font-size:10.5px;letter-spacing:.06em;margin-top:12px;font-family:var(--mono)}
  @media(max-width:720px){
    .sidebar{position:absolute;z-index:5;height:100%}
    .app.collapsed .sidebar{margin-left:-270px}
    .app:not(.collapsed) .main{filter:brightness(.55)}
    .msg{grid-template-columns:56px 1fr}
    .hero h1{font-size:34px}
  }
</style>
</head>
<body>
<div class="app" id="app">
  <aside class="sidebar">
    <div class="side-top">
      <div class="brand"><span class="mark"></span>Speech Data Collector</div>
      <button class="new-btn" id="new">
        <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M7 1.5v11M1.5 7h11"/></svg>
        New transcript</button>
    </div>
    <div class="hist" id="hist"></div>
    <div class="side-foot"><span>LOCAL · PRIVATE</span></div>
  </aside>

  <div class="main">
    <div class="topbar">
      <button class="icon-btn" id="toggle" title="Toggle sidebar">
        <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.4"><rect x="2.5" y="3.5" width="13" height="11" rx="2"/><line x1="7" y1="3.5" x2="7" y2="14.5"/></svg>
      </button>
      <div class="top-title" id="topTitle">New transcript</div>
      <div class="model" title="Transcription model">
        <span class="lbl">Model</span>
        <select id="model">
          <option value="auto">auto</option>
          <option value="parakeet">parakeet</option>
          <option value="parakeet-en">parakeet-en</option>
          <option value="tiny">whisper tiny</option>
          <option value="base">whisper base</option>
          <option value="small">whisper small</option>
          <option value="medium">whisper medium</option>
          <option value="large-v3">whisper large-v3</option>
          <!--API_OPTIONS-->
        </select>
      </div>
      <button class="icon-btn" id="theme" title="Toggle theme">
        <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.4"><circle cx="9" cy="9" r="6.5"/><path d="M9 2.5a6.5 6.5 0 0 1 0 13z" fill="currentColor" stroke="none"/></svg>
      </button>
    </div>

    <div class="scroll" id="scroll">
      <div class="stream" id="stream"></div>
    </div>

    <div class="composer-wrap">
      <div class="composer" id="composer">
        <div class="file-chip" id="fileChip"><span id="fileName"></span><button class="x" id="fileX">✕</button></div>
        <div class="crow">
          <button class="attach" id="attach" title="Attach audio/video file">
            <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M14 8.4l-5.3 5.3a3 3 0 0 1-4.3-4.3l6.1-6.1a2 2 0 0 1 2.9 2.9l-5.7 5.7a1 1 0 0 1-1.5-1.5l5.1-5.1"/></svg>
          </button>
          <textarea id="input" rows="1" placeholder="Paste a YouTube URL, or drop a file…"></textarea>
          <button class="send" id="send" title="Ingest">
            <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M9 14.5V4M4.5 8.5 9 3.8l4.5 4.7"/></svg>
          </button>
        </div>
        <div class="copts">
          <label id="wrapTr"><input type="checkbox" id="transcribe"> Force transcription</label>
          <label><input type="checkbox" id="vad" checked> VAD</label>
          <label>Lang <input type="text" id="lang" placeholder="auto"></label>
          <label title="Build a TTS/STT training dataset (downloads a .zip)"><input type="checkbox" id="dataset"> Dataset</label>
        </div>
        <div class="copts" id="dsOpts" style="display:none">
          <label>Seg s <input type="text" id="segMin" value="1" style="width:42px"> – <input type="text" id="segMax" value="15" style="width:42px"></label>
          <label>Rate <input type="text" id="sr" value="22050" style="width:64px"></label>
          <span style="color:var(--muted);font-size:10px">Playlists/feeds &amp; long sources → use the CLI</span>
        </div>
      </div>
      <div class="hint">LOCAL MODELS CACHE AFTER FIRST DOWNLOAD · REMOTE APIS USE YOUR PROFILE</div>
    </div>
    <input type="file" id="file" hidden accept="audio/*,video/*,.mp3,.m4a,.wav,.mp4,.webm,.mkv,.mov,.flac,.ogg,.opus,.aac">
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
const stream = $('#stream'), scroll = $('#scroll');
let pickedFile = null, current = null;

/* ---- theme + sidebar ---- */
const root = document.documentElement;
root.dataset.theme = localStorage.getItem('hs-theme') || 'light';
$('#theme').onclick = () => { root.dataset.theme = root.dataset.theme==='light'?'dark':'light';
  localStorage.setItem('hs-theme', root.dataset.theme); };
$('#toggle').onclick = () => $('#app').classList.toggle('collapsed');

/* ---- model persistence ---- */
$('#model').value = localStorage.getItem('hs-model') || 'auto';
$('#model').onchange = e => localStorage.setItem('hs-model', e.target.value);

/* ---- history (localStorage) ---- */
const HKEY = 'hs-history';
const load = () => { try { return JSON.parse(localStorage.getItem(HKEY)) || []; } catch { return []; } };
const save = h => localStorage.setItem(HKEY, JSON.stringify(h.slice(0,100)));
function renderHist() {
  const h = load(); const box = $('#hist');
  if (!h.length) { box.innerHTML = '<div class="hist-empty">No transcripts yet.</div>'; return; }
  box.innerHTML = '<span class="lbl">Recent</span>';
  h.forEach(item => {
    const el = document.createElement('div');
    el.className = 'hist-item' + (current===item.id?' active':'');
    el.innerHTML = `<div class="t"></div><div class="s">${esc(item.method||'')} · ${esc(item.duration||'')}</div><button class="del" title="Delete">✕</button>`;
    el.querySelector('.t').textContent = item.title || 'Untitled';
    el.onclick = e => { if (e.target.classList.contains('del')) return; openItem(item); };
    el.querySelector('.del').onclick = e => { e.stopPropagation();
      save(load().filter(x=>x.id!==item.id)); if(current===item.id) newChat(); renderHist(); };
    box.appendChild(el);
  });
}

/* ---- markdown render (safe, minimal) ---- */
function esc(s){ return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function renderMd(md){
  let html='',i=0; const lines=md.split('\n');
  if (lines[0]==='---'){ let j=1; const fm=[]; while(j<lines.length&&lines[j]!=='---')fm.push(lines[j++]);
    if(j<lines.length){ html+='<div class="fm">'+esc(fm.join('\n'))+'</div>'; i=j+1; } }
  let para=[]; const flush=()=>{ if(para.length){ html+='<p>'+inl(para.join(' '))+'</p>'; para=[]; } };
  const inl=s=>esc(s).replace(/\*\*(.+?)\*\*/g,'<b>$1</b>')
    .replace(/(\[\d\d:\d\d:\d\d(?:\s*[–-]\s*\d\d:\d\d:\d\d)?\])/g,'<span class="ts">$1</span>');
  for(;i<lines.length;i++){ const ln=lines[i];
    if(ln.startsWith('## ')){flush();html+='<h2>'+inl(ln.slice(3))+'</h2>';}
    else if(ln.startsWith('# ')){flush();html+='<h1>'+inl(ln.slice(2))+'</h1>';}
    else if(ln.trim()==='---'){flush();html+='<hr>';}
    else if(ln.trim()===''){flush();}
    else para.push(ln); }
  flush(); return html;
}

/* ---- view building ---- */
function clearStream(){ stream.innerHTML=''; }
function showHero(){
  current=null; $('#topTitle').textContent='New transcript'; renderHist();
  clearStream();
  const h=document.createElement('div'); h.className='hero';
  h.innerHTML=`<div class="wave" aria-hidden="true"><span></span><span></span><span></span><span></span><span></span><span></span><span></span></div>
    <div class="eyebrow lbl">Speech → text · training data</div>
    <h1>Speech Data Collector</h1>
    <p>Turn video, podcasts, and recordings into timestamped markdown or TTS/STT datasets — locally or via your STT API.</p>
    <div class="chips">
      <button class="chip" data-ex="https://www.youtube.com/watch?v=9GLYsrMpprs">Try a YouTube video</button>
      <button class="chip" data-attach>Upload an audio file</button>
    </div>`;
  stream.appendChild(h);
  h.querySelectorAll('.chip').forEach(c=>c.onclick=()=>{ if(c.dataset.attach)$('#file').click();
    else{ $('#input').value=c.dataset.ex; autoGrow(); $('#input').focus(); } });
}
function msgShell(role, isBot){
  const m=document.createElement('div'); m.className='msg '+(isBot?'bot':'user');
  m.innerHTML=`<div class="role">${role}</div><div class="body"></div>`;
  return m;
}
function userMsg(text){
  const m=msgShell('You',false);
  const s=document.createElement('div'); s.className='src'; s.textContent=text;
  m.querySelector('.body').appendChild(s); stream.appendChild(m); return m;
}
function botThinking(){
  const m=msgShell('Speech Data Collector',true);
  m.querySelector('.body').innerHTML=`<div class="think"><div class="spin"></div>
    <span>Transcribing… this can take a while for long media.</span></div>`;
  stream.appendChild(m); toBottom(); return m;
}
function docNode(d){
  const wrap=document.createElement('div'); wrap.className='doc';
  wrap.innerHTML=`
    <div class="doc-head">
      <div class="meta">
        <span class="badge">${esc(d.method)}</span>
        <span>lang <b>${esc(d.language)}</b></span>
        <span>duration <b>${esc(d.duration)}</b></span>
        <span><b>${d.sections}</b> sections</span>
      </div>
      <div class="seg"><button class="on" data-v="md">Preview</button><button data-v="raw">Markdown</button></div>
      <div class="doc-actions"><button data-a="copy">Copy</button><button data-a="dl">Download</button></div>
    </div>
    <div class="doc-body"><div class="md"></div><div class="raw" style="display:none"></div></div>`;
  wrap.querySelector('.md').innerHTML=renderMd(d.markdown);
  wrap.querySelector('.raw').textContent=d.markdown;
  const seg=wrap.querySelectorAll('.seg button'), md=wrap.querySelector('.md'), raw=wrap.querySelector('.raw');
  seg.forEach(b=>b.onclick=()=>{ seg.forEach(x=>x.classList.toggle('on',x===b));
    const v=b.dataset.v; md.style.display=v==='md'?'':'none'; raw.style.display=v==='raw'?'':'none'; });
  wrap.querySelector('[data-a=copy]').onclick=()=>navigator.clipboard.writeText(d.markdown);
  wrap.querySelector('[data-a=dl]').onclick=()=>{
    const slug=(d.title||'transcript').toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'')||'transcript';
    const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([d.markdown],{type:'text/markdown'}));
    a.download=slug+'.md'; a.click(); URL.revokeObjectURL(a.href); };
  return wrap;
}
function toBottom(){ scroll.scrollTop=scroll.scrollHeight; }

/* ---- open a saved item ---- */
function openItem(item){
  current=item.id; $('#topTitle').textContent=item.title; renderHist(); clearStream();
  userMsg(item.source);
  const m=msgShell('Speech Data Collector',true); m.querySelector('.body').appendChild(docNode(item));
  stream.appendChild(m); scroll.scrollTop=0;
}

/* ---- submit ---- */
function newChat(){ pickedFile=null; updChip(); $('#input').value=''; autoGrow(); showHero(); }
$('#new').onclick=newChat;

function downloadZip(b64, name){
  const bin=atob(b64); const arr=new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++) arr[i]=bin.charCodeAt(i);
  const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([arr],{type:'application/zip'}));
  a.download=name||'speechcollector-dataset.zip'; a.click(); URL.revokeObjectURL(a.href);
}
function datasetNode(d){
  const wrap=document.createElement('div'); wrap.className='doc';
  const warns=(d.warnings||[]).map(w=>'<p style="color:var(--accent)">! '+esc(w)+'</p>').join('');
  wrap.innerHTML='<div class="doc-head"><div class="meta">'+
    '<span class="badge">DATASET</span>'+
    '<span><b>'+d.clips+'</b> clips</span><span>duration <b>'+esc(d.duration)+'</b></span>'+
    '<span>dropped <b>'+d.dropped+'</b></span></div>'+
    '<div class="doc-actions"><button data-a="dl">Download .zip</button></div></div>'+
    '<div class="doc-body"><div class="md"><p>Dataset ready — the .zip download should have started.</p>'+warns+'</div></div>';
  wrap.querySelector('[data-a=dl]').onclick=()=>downloadZip(d.zip_b64,d.zip_name);
  return wrap;
}

async function submit(){
  const text=$('#input').value.trim();
  if(!pickedFile && !text) return;
  const model=$('#model').value, lang=$('#lang').value.trim(),
        vad=$('#vad').checked, transcribe=$('#transcribe').checked, dataset=$('#dataset').checked;
  const sourceLabel = pickedFile ? ('File · '+pickedFile.name) : text;

  if(stream.querySelector('.hero')) clearStream();
  current=null; $('#topTitle').textContent = pickedFile ? pickedFile.name : (text.slice(0,60)||'Transcript');
  userMsg(sourceLabel);
  const thinking=botThinking();
  $('#send').disabled=true;
  const fileToSend=pickedFile; pickedFile=null; updChip(); $('#input').value=''; autoGrow();

  try{
    let res;
    if(dataset){
      const ds={model,lang,vad,segment_min:$('#segMin').value,segment_max:$('#segMax').value,sample_rate:$('#sr').value};
      if(fileToSend){
        const q=new URLSearchParams({name:fileToSend.name,model,lang,vad:vad?'1':'0',
          segment_min:ds.segment_min,segment_max:ds.segment_max,sample_rate:ds.sample_rate});
        res=await fetch('/api/dataset-file?'+q,{method:'POST',body:fileToSend});
      }else{
        res=await fetch('/api/dataset',{method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify({url:text,...ds})});
      }
      const d=await res.json();
      if(!d.ok) throw new Error((d.error||'Failed')+(d.hint?'\nTry: '+d.hint:''));
      downloadZip(d.zip_b64,d.zip_name);
      thinking.querySelector('.body').innerHTML=''; thinking.querySelector('.body').appendChild(datasetNode(d));
      $('#topTitle').textContent='Dataset · '+d.clips+' clips';
      return;  // datasets aren't added to history (not re-openable as markdown)
    }
    if(fileToSend){
      const q=new URLSearchParams({name:fileToSend.name,model,lang,vad:vad?'1':'0'});
      res=await fetch('/api/file?'+q,{method:'POST',body:fileToSend});
    }else{
      res=await fetch('/api/url',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({url:text,transcribe,model,lang,vad})});
    }
    const d=await res.json();
    if(!d.ok) throw new Error((d.error||'Failed')+(d.hint?'\nTry: '+d.hint:''));
    thinking.querySelector('.body').innerHTML=''; thinking.querySelector('.body').appendChild(docNode(d));
    $('#topTitle').textContent=d.title;
    const item={...d, id:Date.now(), source:sourceLabel};
    const h=load(); h.unshift(item); save(h); current=item.id; renderHist();
  }catch(e){
    const b=thinking.querySelector('.body'); b.innerHTML='';
    const er=document.createElement('div'); er.className='err'; er.textContent=e.message; b.appendChild(er);
  }finally{
    $('#send').disabled=false; toBottom();
  }
}
$('#send').onclick=submit;
$('#dataset').onchange=e=>{ $('#dsOpts').style.display=e.target.checked?'flex':'none'; };

/* ---- composer behaviour ---- */
const input=$('#input'), comp=$('#composer');
function autoGrow(){ input.style.height='auto'; input.style.height=Math.min(input.scrollHeight,168)+'px'; }
input.addEventListener('input',autoGrow);
input.addEventListener('focus',()=>comp.classList.add('focus'));
input.addEventListener('blur',()=>comp.classList.remove('focus'));
input.addEventListener('keydown',e=>{ if(e.key==='Enter'&&!e.shiftKey){ e.preventDefault(); submit(); } });
$('#attach').onclick=()=>$('#file').click();
$('#file').onchange=e=>{ pickedFile=e.target.files[0]||null; updChip(); e.target.value=''; };
function updChip(){ const c=$('#fileChip'); if(pickedFile){ $('#fileName').textContent='File · '+pickedFile.name; c.classList.add('show'); }
  else c.classList.remove('show'); }
$('#fileX').onclick=()=>{ pickedFile=null; updChip(); };
['dragover','dragenter'].forEach(ev=>comp.addEventListener(ev,e=>{e.preventDefault();comp.classList.add('drag');}));
['dragleave','drop'].forEach(ev=>comp.addEventListener(ev,e=>{e.preventDefault();comp.classList.remove('drag');}));
comp.addEventListener('drop',e=>{ if(e.dataTransfer.files[0]){ pickedFile=e.dataTransfer.files[0]; updChip(); } });

/* ---- boot ---- */
renderHist(); showHero();
</script>
</body>
</html>
"""
