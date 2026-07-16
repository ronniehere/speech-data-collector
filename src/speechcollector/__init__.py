"""Speech Data Collector — local media ingestion for markdown and training datasets.

Copyright (c) 2026 Rohan Ahmed / Modern Intelligent Solutions. All rights reserved.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("speechcollector")
except PackageNotFoundError:  # pragma: no cover - source tree without an install
    __version__ = "0.0.0.dev0"
