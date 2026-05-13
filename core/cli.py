from __future__ import annotations

import os
import sys
import traceback
from core.cli_parser import build_parser
from core.console import print_error, print_log, print_warning
from core.runtime import configure_runtime, runtime_warning_context


def main(argv: list[str] | None = None) -> None:
    configure_runtime()
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        with runtime_warning_context():
            args = parser.parse_args(raw_argv)
            args.handler(args)
    except KeyboardInterrupt:
        print_warning("Execution interrupted.")
        raise SystemExit(130)
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print_error(exc.code)
            raise SystemExit(1)
        raise
    except Exception as exc:
        command = getattr(locals().get("args", None), "command", "glyph")
        print_error(f"{command} failed: {exc}")
        if os.environ.get("GLYPH_DEBUG") == "1":
            traceback.print_exc()
        else:
            print_log(
                command,
                "Set GLYPH_DEBUG=1 to print the full traceback.",
                tone="warn",
            )
        raise SystemExit(1)
