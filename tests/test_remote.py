"""Offline tests for remote STT API profiles (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from speechcollector.errors import TranscriptionError
from speechcollector.remote import (
    RemoteSTTConfig,
    extract_path,
    extract_path_list,
    interpolate_env,
    list_api_profiles,
    load_api_profile,
    map_response,
    transcribe_remote,
)
from speechcollector.transcribe import _resolve_engine, transcribe_audio

FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLE_API = (
    Path(__file__).resolve().parents[1] / "examples" / "apis" / "openai-compatible.json"
)


# --- Path extractor --------------------------------------------------------


def test_extract_path_simple() -> None:
    data = {"language": "en", "duration": 12.5}
    assert extract_path(data, "language") == "en"
    assert extract_path(data, "duration") == 12.5
    assert extract_path(data, "missing") is None


def test_extract_path_list_array_wildcard() -> None:
    data = {
        "segments": [
            {"text": "hello", "start": 0.0, "end": 0.5},
            {"text": "world", "start": 0.5, "end": 1.0},
        ]
    }
    segs = extract_path_list(data, "segments")
    assert len(segs) == 2
    assert segs[0]["text"] == "hello"
    texts = extract_path_list(data, "segments[].text")
    assert texts == ["hello", "world"]


def test_extract_path_indexed() -> None:
    data = {"results": [{"alts": [{"transcript": "hi"}]}]}
    assert extract_path(data, "results[0].alts[0].transcript") == "hi"


# --- Env interpolation -----------------------------------------------------


def test_interpolate_env_replaces(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_KEY", "secret123")
    assert interpolate_env("Bearer ${MY_KEY}") == "Bearer secret123"


def test_interpolate_env_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_API_KEY", raising=False)
    with pytest.raises(TranscriptionError, match="MISSING_API_KEY"):
        interpolate_env("Bearer ${MISSING_API_KEY}")


# --- Profile loader --------------------------------------------------------


def test_load_api_profile_from_path() -> None:
    cfg = load_api_profile(str(EXAMPLE_API))
    assert cfg.name == "openai-compatible"
    assert cfg.upload == "multipart"
    assert cfg.words_path == "words"
    assert cfg.supports_word_timestamps


def test_load_api_profile_api_prefix() -> None:
    cfg = load_api_profile(f"api:{EXAMPLE_API}")
    assert cfg.endpoint.endswith("/audio/transcriptions")


def test_load_api_profile_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPEECHCOLLECTOR_API_DIR", str(tmp_path))
    dest = tmp_path / "myapi.json"
    dest.write_text(EXAMPLE_API.read_text(encoding="utf-8"), encoding="utf-8")
    cfg = load_api_profile("myapi")
    assert cfg.name == "openai-compatible"  # name from file content
    assert "myapi" in list_api_profiles()


def test_load_api_profile_unknown_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPEECHCOLLECTOR_API_DIR", str(tmp_path))
    with pytest.raises(TranscriptionError, match="Unknown remote STT"):
        load_api_profile("does-not-exist")


# --- Response mapping ------------------------------------------------------


_OPENAI_PAYLOAD = {
    "text": "Hello world",
    "language": "en",
    "duration": 1.2,
    "segments": [
        {"text": "Hello world", "start": 0.0, "end": 1.2},
    ],
    "words": [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.5, "end": 1.2},
    ],
}


def test_map_response_segments_and_words() -> None:
    cfg = RemoteSTTConfig(
        name="t",
        endpoint="https://example.test/v1",
        segments_path="segments",
        words_path="words",
        language_path="language",
        duration_path="duration",
    )
    segs, words, lang, dur = map_response(
        _OPENAI_PAYLOAD, cfg, language=None, word_timestamps=True
    )
    assert len(segs) == 1 and segs[0].text == "Hello world"
    assert words is not None and len(words) == 2
    assert words[0].text == "Hello" and words[1].end_s == 1.2
    assert lang == "en" and dur == 1.2


def test_map_response_relative_words() -> None:
    payload = {
        "segments": [
            {
                "text": "Hi there",
                "start": 0.0,
                "end": 1.0,
                "words": [
                    {"word": "Hi", "start": 0.0, "end": 0.4},
                    {"word": "there", "start": 0.4, "end": 1.0},
                ],
            }
        ]
    }
    cfg = RemoteSTTConfig(
        name="t",
        endpoint="https://example.test/v1",
        segments_path="segments",
        words_path="words",
    )
    _, words, _, _ = map_response(payload, cfg, language="en", word_timestamps=True)
    assert words is not None and [w.text for w in words] == ["Hi", "there"]


def test_map_response_no_words_path_errors() -> None:
    cfg = RemoteSTTConfig(
        name="plain",
        endpoint="https://example.test/v1",
        text_path="text",
        # no words_path
    )
    with pytest.raises(TranscriptionError, match="no words_path"):
        map_response(
            {"text": "hi"}, cfg, language=None, word_timestamps=True
        )


def test_map_response_empty_words_errors() -> None:
    cfg = RemoteSTTConfig(
        name="t",
        endpoint="https://example.test/v1",
        text_path="text",
        words_path="words",
    )
    with pytest.raises(TranscriptionError, match="no word timestamps"):
        map_response({"text": "hi", "words": []}, cfg, language=None, word_timestamps=True)


# --- transcribe_remote (monkeypatched HTTP) --------------------------------


def test_transcribe_remote_maps_canned_response(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF....WAVE")  # content unused; HTTP is faked
    cfg = load_api_profile(str(EXAMPLE_API))

    def fake_http(path: Path, config: RemoteSTTConfig, *, language=None, env=None):
        assert path == audio
        return _OPENAI_PAYLOAD

    result = transcribe_remote(
        audio,
        config=cfg,
        language=None,
        word_timestamps=True,
        http_fn=fake_http,
    )
    assert result.method == "api-openai-compatible"
    assert result.segments[0].text == "Hello world"
    assert result.words is not None and len(result.words) == 2
    assert result.duration_s == 1.2


def test_transcribe_audio_routes_api_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"x")

    def fake_remote(path, *, config, language=None, word_timestamps=False, on_progress=None, http_fn=None):
        from speechcollector.transcribe import TranscriptionResult

        return TranscriptionResult(
            segments=[],
            language="en",
            duration_s=0.0,
            model_size="api-test",
            method="api-test",
            words=[] if word_timestamps else None,
        )

    monkeypatch.setattr(
        "speechcollector.remote.transcribe_remote", fake_remote
    )
    # Avoid real profile load by pointing at example file.
    engine, spec = _resolve_engine(f"api:{EXAMPLE_API}")
    assert engine == "remote"
    result = transcribe_audio(audio, model_size=f"api:{EXAMPLE_API}", word_timestamps=False)
    assert result.method == "api-test"


def test_resolve_engine_api_prefix() -> None:
    assert _resolve_engine("api:openai") == ("remote", "openai")
    assert _resolve_engine("api:/tmp/x.json") == ("remote", "/tmp/x.json")


def test_dataset_mode_without_words_raises(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"x")
    cfg = RemoteSTTConfig(
        name="plain",
        endpoint="https://example.test/v1",
        text_path="text",
    )

    def fake_http(path, config, *, language=None, env=None):
        return {"text": "hello"}

    with pytest.raises(TranscriptionError, match="no words_path"):
        transcribe_remote(
            audio, config=cfg, word_timestamps=True, http_fn=fake_http
        )


def test_example_profile_is_valid_json() -> None:
    data = json.loads(EXAMPLE_API.read_text(encoding="utf-8"))
    RemoteSTTConfig.model_validate(data)
