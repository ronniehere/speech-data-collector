"""Fully configurable remote STT via HTTP API profiles.

Users describe any speech-to-text HTTP API in a JSON profile (endpoint, headers,
upload style, response field mapping). Secrets are referenced as ``${ENV_VAR}``
and never stored literally. Profiles live under config dirs:

* ``$SPEECHCOLLECTOR_API_DIR`` (if set)
* ``~/.speechcollector/apis/``
* ``%APPDATA%/speechcollector/apis/`` (Windows)

Selected with the model string ``api:<name>`` or ``api:/path/to/profile.json``.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import uuid
from pathlib import Path
from typing import Any, Literal
from urllib import error as urllib_error
from urllib import request as urllib_request

from pydantic import BaseModel, Field, field_validator

from speechcollector.errors import TranscriptionError
from speechcollector.models import Segment, Word

# Lazy import of TranscriptionResult to avoid circular imports at module load;
# typed in annotations via string / deferred import in the function.

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_USER_AGENT = "speechcollector/0.3"
_DEFAULT_TIMEOUT_S = 600

UploadStyle = Literal["multipart", "json_base64", "raw_body"]


class RemoteSTTConfig(BaseModel):
    """JSON profile describing an external STT HTTP API."""

    name: str
    endpoint: str
    method: str = "POST"
    headers: dict[str, str] = Field(default_factory=dict)
    upload: UploadStyle = "multipart"
    audio_field: str = "file"
    filename: str | None = None  # default: source file name
    form_fields: dict[str, Any] = Field(default_factory=dict)
    json_body: dict[str, Any] = Field(default_factory=dict)

    # Response mapping (dotted paths; ``[]`` expands arrays).
    text_path: str | None = None
    segments_path: str | None = None
    segment_text_key: str = "text"
    segment_start_key: str = "start"
    segment_end_key: str = "end"
    words_path: str | None = None  # absolute, or relative to each segment
    word_text_key: str = "word"
    word_start_key: str = "start"
    word_end_key: str = "end"
    word_confidence_key: str | None = None
    language_path: str | None = None
    duration_path: str | None = None

    max_upload_mb: float = 25.0
    timeout_s: float = _DEFAULT_TIMEOUT_S

    @field_validator("upload")
    @classmethod
    def _check_upload(cls, value: str) -> str:
        allowed = ("multipart", "json_base64", "raw_body")
        if value not in allowed:
            raise ValueError(f"upload must be one of {allowed}, got {value!r}")
        return value

    @property
    def supports_word_timestamps(self) -> bool:
        """True when the profile declares a words_path mapping."""
        return bool(self.words_path)


# --- Config dirs / profile loading -----------------------------------------


def api_config_dirs() -> list[Path]:
    """Ordered search paths for named API profiles (first match wins)."""
    dirs: list[Path] = []
    env = os.environ.get("SPEECHCOLLECTOR_API_DIR")
    if env:
        dirs.append(Path(env).expanduser())
    dirs.append(Path.home() / ".speechcollector" / "apis")
    appdata = os.environ.get("APPDATA")
    if appdata:
        dirs.append(Path(appdata) / "speechcollector" / "apis")
    return dirs


def list_api_profiles() -> list[str]:
    """Names of discoverable ``*.json`` profiles (no extension), unique by name."""
    seen: dict[str, Path] = {}
    for directory in api_config_dirs():
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            seen.setdefault(path.stem, path)
    return sorted(seen)


def load_api_profile(spec: str) -> RemoteSTTConfig:
    """Load a profile from a name or filesystem path.

    ``spec`` is either a bare name (``openai``), a path to a ``.json`` file, or
    the full ``api:...`` model string (the prefix is stripped if present).
    """
    if spec.startswith("api:"):
        spec = spec[4:]
    path = Path(spec).expanduser()
    if path.suffix.lower() == ".json" and path.is_file():
        return _load_profile_file(path)
    # Named profile: look in config dirs.
    for directory in api_config_dirs():
        candidate = directory / f"{spec}.json"
        if candidate.is_file():
            return _load_profile_file(candidate)
    searched = ", ".join(str(d) for d in api_config_dirs())
    raise TranscriptionError(
        f"Unknown remote STT API profile: {spec!r}",
        hint=(
            f"Add {spec}.json under one of: {searched} "
            "(or pass api:/path/to/profile.json)."
        ),
    )


def _load_profile_file(path: Path) -> RemoteSTTConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TranscriptionError(
            f"Could not read API profile {path}: {exc}",
            hint="Check the file is valid JSON.",
        ) from exc
    if "name" not in raw:
        raw["name"] = path.stem
    try:
        return RemoteSTTConfig.model_validate(raw)
    except Exception as exc:
        raise TranscriptionError(
            f"Invalid API profile {path}: {exc}",
            hint="See examples/apis/ for a working OpenAI-compatible profile.",
        ) from exc


# --- ${ENV} interpolation --------------------------------------------------


def interpolate_env(value: str, *, env: dict[str, str] | None = None) -> str:
    """Replace ``${VAR}`` placeholders with environment values.

    Raises TranscriptionError when a referenced variable is missing or empty.
    """
    environ = env if env is not None else os.environ

    def _repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in environ or environ[key] == "":
            raise TranscriptionError(
                f"Environment variable {key} is not set (required by API profile).",
                hint=f"Export {key} before running, or put the secret in your shell/env file.",
            )
        return environ[key]

    return _ENV_PATTERN.sub(_repl, value)


def _interpolate_mapping(data: dict[str, Any], *, env: dict[str, str] | None = None) -> dict[str, Any]:
    """Recursively interpolate string values in a JSON-like mapping."""
    out: dict[str, Any] = {}
    for key, val in data.items():
        if isinstance(val, str):
            out[key] = interpolate_env(val, env=env)
        elif isinstance(val, dict):
            out[key] = _interpolate_mapping(val, env=env)
        elif isinstance(val, list):
            out[key] = [
                interpolate_env(v, env=env) if isinstance(v, str) else v for v in val
            ]
        else:
            out[key] = val
    return out


# --- Dotted path extraction ------------------------------------------------


def extract_path(data: Any, path: str | None) -> Any:
    """Walk ``data`` with a dotted path; ``[]`` expands arrays.

    Examples:
        ``segments[].text`` → list of each segment's text
        ``results[0].alts`` → first results entry's alts
        ``language`` → top-level key
    """
    if not path:
        return None
    current: Any = data
    for part in path.split("."):
        if current is None:
            return None
        if part.endswith("[]"):
            key = part[:-2]
            if key:
                if not isinstance(current, dict) or key not in current:
                    return None
                current = current[key]
            if not isinstance(current, list):
                return None
            # Remaining path is applied per-element by the caller when using
            # extract_path on each item; for a terminal ``foo[]`` return the list.
            continue
        m = re.fullmatch(r"(.+)\[(\d+)\]", part)
        if m:
            key, idx_s = m.group(1), int(m.group(2))
            if key:
                if not isinstance(current, dict) or key not in current:
                    return None
                current = current[key]
            if not isinstance(current, list) or idx_s >= len(current):
                return None
            current = current[idx_s]
            continue
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def extract_path_list(data: Any, path: str | None) -> list[Any]:
    """Like :func:`extract_path`, but always returns a list (empty if missing).

    When the path contains ``[]``, collects one value (or sub-object) per array
    element. For nested ``a[].b[].c`` the result is flattened one level at a time.
    """
    if not path:
        return []
    parts = path.split(".")
    return _walk_list(data, parts)


def _walk_list(node: Any, parts: list[str]) -> list[Any]:
    if not parts:
        # Terminal list (e.g. path ``segments``) → the items themselves.
        if isinstance(node, list):
            return node
        return [node] if node is not None else []
    part, rest = parts[0], parts[1:]
    if part.endswith("[]"):
        key = part[:-2]
        arr = node.get(key) if key and isinstance(node, dict) else node if not key else None
        if not isinstance(arr, list):
            return []
        out: list[Any] = []
        for item in arr:
            out.extend(_walk_list(item, rest))
        return out
    m = re.fullmatch(r"(.+)\[(\d+)\]", part)
    if m:
        key, idx = m.group(1), int(m.group(2))
        base = node.get(key) if key and isinstance(node, dict) else node if not key else None
        if not isinstance(base, list) or idx >= len(base):
            return []
        return _walk_list(base[idx], rest)
    if not isinstance(node, dict) or part not in node:
        return []
    return _walk_list(node[part], rest)


# --- HTTP upload (stdlib urllib) -------------------------------------------


def _guess_content_type(path: Path) -> str:
    ctype, _ = mimetypes.guess_type(str(path))
    return ctype or "application/octet-stream"


def _build_multipart(
    path: Path,
    *,
    audio_field: str,
    filename: str,
    form_fields: dict[str, Any],
) -> tuple[bytes, str]:
    """Build a multipart/form-data body; returns (body, content_type)."""
    boundary = f"----speechcollector{uuid.uuid4().hex}"
    lines: list[bytes] = []

    def _part(name: str, value: str) -> None:
        lines.append(f"--{boundary}\r\n".encode())
        lines.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        lines.append(value.encode("utf-8"))
        lines.append(b"\r\n")

    for key, value in form_fields.items():
        if isinstance(value, list):
            # OpenAI-style repeated fields: timestamp_granularities[]=word
            for item in value:
                _part(f"{key}[]", str(item))
        elif isinstance(value, dict):
            _part(key, json.dumps(value, ensure_ascii=False))
        else:
            _part(key, str(value))
    data = path.read_bytes()
    ctype = _guess_content_type(path)
    lines.append(f"--{boundary}\r\n".encode())
    lines.append(
        (
            f'Content-Disposition: form-data; name="{audio_field}"; '
            f'filename="{filename}"\r\n'
        ).encode()
    )
    lines.append(f"Content-Type: {ctype}\r\n\r\n".encode())
    lines.append(data)
    lines.append(b"\r\n")
    lines.append(f"--{boundary}--\r\n".encode())
    body = b"".join(lines)
    return body, f"multipart/form-data; boundary={boundary}"


def http_transcribe(
    path: Path,
    config: RemoteSTTConfig,
    *,
    language: str | None = None,
    env: dict[str, str] | None = None,
) -> Any:
    """POST the audio file per ``config`` and return the parsed JSON response."""
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > config.max_upload_mb:
        raise TranscriptionError(
            f"Audio file is {size_mb:.1f} MB; profile '{config.name}' "
            f"caps uploads at {config.max_upload_mb:g} MB.",
            hint="Use a shorter clip, or raise max_upload_mb in the API profile.",
        )

    headers = {k: interpolate_env(v, env=env) for k, v in config.headers.items()}
    headers.setdefault("User-Agent", _USER_AGENT)

    endpoint = interpolate_env(config.endpoint, env=env)
    # Optional {model}/{language} placeholders in the URL.
    model_val = str(config.form_fields.get("model") or config.json_body.get("model") or "")
    endpoint = endpoint.replace("{model}", model_val).replace(
        "{language}", language or ""
    )

    form_fields = _interpolate_mapping(dict(config.form_fields), env=env)
    json_body = _interpolate_mapping(dict(config.json_body), env=env)
    if language is not None:
        form_fields.setdefault("language", language)
        json_body.setdefault("language", language)

    filename = config.filename or path.name
    body: bytes
    if config.upload == "multipart":
        body, content_type = _build_multipart(
            path,
            audio_field=config.audio_field,
            filename=filename,
            form_fields=form_fields,
        )
        headers["Content-Type"] = content_type
    elif config.upload == "json_base64":
        payload = dict(json_body)
        payload[config.audio_field] = base64.b64encode(path.read_bytes()).decode("ascii")
        body = json.dumps(payload).encode("utf-8")
        headers.setdefault("Content-Type", "application/json")
    else:  # raw_body
        body = path.read_bytes()
        headers.setdefault("Content-Type", _guess_content_type(path))
        # Query-style extras aren't supported; raw_body uses headers + body only.
        _ = form_fields  # reserved for future query params

    req = urllib_request.Request(
        endpoint, data=body, headers=headers, method=config.method.upper()
    )
    try:
        with urllib_request.urlopen(req, timeout=config.timeout_s) as resp:
            raw = resp.read()
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise TranscriptionError(
            f"Remote STT API '{config.name}' returned HTTP {exc.code}: {detail}",
            hint="Check the profile endpoint, headers, and API key env vars.",
        ) from exc
    except urllib_error.URLError as exc:
        raise TranscriptionError(
            f"Could not reach remote STT API '{config.name}': {exc.reason}",
            hint="Check the endpoint URL and your network connection.",
        ) from exc
    except TimeoutError as exc:
        raise TranscriptionError(
            f"Remote STT API '{config.name}' timed out after {config.timeout_s:g}s.",
            hint="Raise timeout_s in the profile, or try a shorter audio file.",
        ) from exc

    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TranscriptionError(
            f"Remote STT API '{config.name}' returned non-JSON response.",
            hint="Check response_format / Accept headers in the profile.",
        ) from exc


# --- Response → TranscriptionResult ----------------------------------------


def _float_or(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def map_response(
    payload: Any,
    config: RemoteSTTConfig,
    *,
    language: str | None,
    word_timestamps: bool,
) -> tuple[list[Segment], list[Word] | None, str, float]:
    """Map a provider JSON payload into segments, optional words, language, duration."""
    segments: list[Segment] = []
    words: list[Word] | None = None

    if config.segments_path:
        raw_segs = extract_path_list(payload, config.segments_path)
        for item in raw_segs:
            if not isinstance(item, dict):
                continue
            text = str(item.get(config.segment_text_key, "")).strip()
            if not text:
                continue
            start_s = max(0.0, _float_or(item.get(config.segment_start_key)))
            end_s = max(start_s, _float_or(item.get(config.segment_end_key), start_s))
            segments.append(Segment(text=text, start_s=start_s, end_s=end_s))

    if not segments and config.text_path:
        text = extract_path(payload, config.text_path)
        if isinstance(text, str) and text.strip():
            segments.append(Segment(text=text.strip(), start_s=0.0, end_s=0.0))

    if word_timestamps:
        if not config.words_path:
            raise TranscriptionError(
                f"API profile '{config.name}' has no words_path; "
                "dataset mode needs word-level timestamps.",
                hint=(
                    "Add a words_path mapping to the profile "
                    "(see examples/apis/openai-compatible.json), "
                    "or use a local --model for dataset builds."
                ),
            )
        collected: list[Word] = []
        # Prefer an absolute path on the payload (e.g. OpenAI top-level ``words``).
        for item in extract_path_list(payload, config.words_path):
            w = _word_from_item(item, config)
            if w is not None:
                collected.append(w)
        # Fall back to words nested under each segment.
        if not collected and config.segments_path:
            raw_segs = extract_path_list(payload, config.segments_path)
            for seg in raw_segs:
                if not isinstance(seg, dict):
                    continue
                nested = extract_path_list(seg, config.words_path)
                if not nested and isinstance(seg.get(config.words_path), list):
                    nested = seg[config.words_path]
                for item in nested:
                    w = _word_from_item(item, config)
                    if w is not None:
                        collected.append(w)
        if not collected:
            raise TranscriptionError(
                f"API profile '{config.name}' returned no word timestamps.",
                hint=(
                    "Confirm the API was asked for word timings "
                    "(e.g. timestamp_granularities / verbose_json) "
                    "and that words_path matches the response."
                ),
            )
        words = collected

    lang = language or "en"
    if config.language_path:
        found = extract_path(payload, config.language_path)
        if isinstance(found, str) and found:
            lang = found

    duration_s = 0.0
    if config.duration_path:
        duration_s = max(0.0, _float_or(extract_path(payload, config.duration_path)))
    if duration_s <= 0.0:
        ends = [s.end_s for s in segments]
        if words:
            ends.extend(w.end_s for w in words)
        duration_s = max(ends) if ends else 0.0

    # If we only had full text and words, back-fill segment end from duration.
    if len(segments) == 1 and segments[0].end_s == 0.0 and duration_s > 0.0:
        segments[0] = Segment(text=segments[0].text, start_s=0.0, end_s=duration_s)

    return segments, words, lang, duration_s


def _word_from_item(item: Any, config: RemoteSTTConfig) -> Word | None:
    if not isinstance(item, dict):
        return None
    text = str(item.get(config.word_text_key, "")).strip()
    if not text:
        return None
    start_s = _float_or(item.get(config.word_start_key))
    end_s = _float_or(item.get(config.word_end_key), start_s)
    conf = 1.0
    if config.word_confidence_key and config.word_confidence_key in item:
        conf = _float_or(item.get(config.word_confidence_key), 1.0)
    return Word(text=text, start_s=start_s, end_s=end_s, confidence=conf)


def transcribe_remote(
    path: Path,
    *,
    config: RemoteSTTConfig,
    language: str | None = None,
    word_timestamps: bool = False,
    on_progress: Any | None = None,
    http_fn: Any | None = None,
) -> Any:
    """Transcribe ``path`` via the remote API described by ``config``.

    ``http_fn`` is injectable for tests (defaults to :func:`http_transcribe`).
    Returns a :class:`~speechcollector.transcribe.TranscriptionResult`.
    """
    from speechcollector.transcribe import TranscriptionResult

    if on_progress is not None:
        on_progress(0.0, 1.0)

    do_http = http_fn or http_transcribe
    payload = do_http(path, config, language=language)
    segments, words, lang, duration_s = map_response(
        payload, config, language=language, word_timestamps=word_timestamps
    )
    if on_progress is not None:
        on_progress(duration_s or 1.0, duration_s or 1.0)

    label = f"api-{config.name}"
    return TranscriptionResult(
        segments=segments,
        language=lang,
        duration_s=duration_s,
        model_size=label,
        method=label,
        words=words,
    )
