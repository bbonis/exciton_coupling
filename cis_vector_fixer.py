#!/usr/bin/env python3
"""
file: fix_second_column.py

Usage:
  python fix_second_column.py <file_path>

Behavior:
  - Count lines N in file.
  - Compute k = (N - 1) / 2.
  - If k is an integer: replace ONLY the 2nd comma-separated field in the FIRST line with k,
    while preserving the original newline at the end of that line.
  - Else: exit non-zero with a warning.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_FIRST_LINE_COL2_RE = re.compile(r"^(?P<before>[^,]*,\s*)(?P<col2>[^,]*)(?P<after>\s*,.*)$")


def _split_eol(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n"):
        return line[:-1], "\n"
    if line.endswith("\r"):
        return line[:-1], "\r"
    return line, ""


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python fix_second_column.py <file_path>", file=sys.stderr)
        return 2

    file_path = Path(argv[1])

    try:
        with file_path.open("r", encoding="utf-8", newline="") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Warning: file not found: {file_path}", file=sys.stderr)
        return 2
    except UnicodeDecodeError as e:
        print(f"Warning: cannot decode file as UTF-8: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"Warning: failed to read file: {e}", file=sys.stderr)
        return 2

    n_lines = len(lines)
    if n_lines < 1:
        print("Warning: file is empty.", file=sys.stderr)
        return 1

    if (n_lines - 1) % 2 != 0:
        k = (n_lines - 1) / 2
        print(
            f"Warning: (lines-1)/2 is not an integer. lines={n_lines}, (lines-1)/2={k}",
            file=sys.stderr,
        )
        return 1

    k_int = (n_lines - 1) // 2

    header_body, header_eol = _split_eol(lines[0])

    m = _FIRST_LINE_COL2_RE.match(header_body)
    if not m:
        print(
            "Warning: first line is not comma-separated with >= 3 fields; cannot replace the second field.",
            file=sys.stderr,
        )
        return 1

    new_header_body = f"{m.group('before')}{k_int}{m.group('after')}"
    lines[0] = new_header_body + header_eol

    try:
        with file_path.open("w", encoding="utf-8", newline="") as f:
            f.write("".join(lines))
    except OSError as e:
        print(f"Warning: failed to write file: {e}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
