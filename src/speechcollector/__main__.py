"""Enable ``python -m speechcollector`` (used by MCP server configs and tests)."""

from speechcollector.cli import app

if __name__ == "__main__":
    app()
