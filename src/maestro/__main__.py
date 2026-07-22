"""Entry point for ``python -m maestro`` and the ``maestro`` console script."""

from __future__ import annotations

import contextlib
import sys


def _ensure_utf8_streams() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        with contextlib.suppress(Exception):
            reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    _ensure_utf8_streams()
    from maestro.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
