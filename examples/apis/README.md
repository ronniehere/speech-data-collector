# Remote STT API profiles

Copy a profile into a config directory and reference it with `--api <name>` or
`--model api:<name>`.

## Config directories (first match wins)

1. `$SPEECHCOLLECTOR_API_DIR` (if set)
2. `~/.speechcollector/apis/`
3. `%APPDATA%\speechcollector\apis\` (Windows)

Example:

```bash
mkdir -p ~/.speechcollector/apis
cp examples/apis/openai-compatible.json ~/.speechcollector/apis/openai.json
export OPENAI_API_KEY=sk-...
speechcollector talk.mp3 --api openai
speechcollector dataset talk.mp3 --api openai --out ./data
```

Or pass a file directly:

```bash
speechcollector talk.mp3 --api-config ./examples/apis/openai-compatible.json
```

## Profile fields

| Field | Purpose |
| --- | --- |
| `endpoint` | HTTP URL (may include `{model}` / `{language}`) |
| `headers` | Request headers; use `${ENV_VAR}` for secrets |
| `upload` | `multipart` \| `json_base64` \| `raw_body` |
| `audio_field` | Form/JSON key for the audio payload |
| `form_fields` / `json_body` | Extra request fields |
| `text_path` | Dotted path to full transcript text (fallback) |
| `segments_path` | Dotted path to segment array (`segments` or `results[].alts`) |
| `segment_*_key` | Keys on each segment object |
| `words_path` | Absolute path or relative-to-segment path for word timings |
| `word_*_key` | Keys on each word object |
| `language_path` / `duration_path` | Optional metadata paths |
| `max_upload_mb` | Pre-flight size guard |

Dotted paths support `[]` to expand arrays (e.g. `segments[].text`).

Dataset mode requires `words_path` and a response that actually includes words.
