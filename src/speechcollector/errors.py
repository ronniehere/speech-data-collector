"""Error types with user-facing messages.

Every error carries a `hint` telling the user what to do next; the CLI
prints message + hint and exits nonzero without a traceback.
"""


class CollectorError(Exception):
    """Base class for all expected failures."""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class InvalidSourceError(CollectorError):
    """The given source is not something speechcollector can ingest."""


class VideoUnavailableError(CollectorError):
    """The video is private, deleted, region-locked, or otherwise unreachable."""


class NoCaptionsError(CollectorError):
    """The video has no captions in any language."""


class CaptionsError(CollectorError):
    """Captions exist but could not be retrieved (e.g. blocked network)."""


class MetadataError(CollectorError):
    """yt-dlp could not fetch or parse video metadata."""


class OutputWriteError(CollectorError):
    """The rendered markdown could not be written to the output path."""


class TranscriptionError(CollectorError):
    """Local whisper transcription failed (model load or audio decode)."""


class AudioDownloadError(CollectorError):
    """yt-dlp could not download the audio stream for transcription."""


class FeedError(CollectorError):
    """A podcast RSS feed could not be fetched or parsed."""


class PlaylistError(CollectorError):
    """A YouTube playlist could not be listed."""


class AudioExportError(CollectorError):
    """ffmpeg/ffprobe was unavailable or a clip could not be sliced/probed."""


class DiarizationError(CollectorError):
    """Speaker diarization could not run (extra not installed, model gated/unauthorized)."""
