#!/usr/bin/env python3
"""Flip inverted NNUE text-corpus game results in place.

The Stockfish corpus stores white-relative scores, but its result field is
black-relative.  bulletformat expects both fields to be white-relative.  The
only valid result tokens are fixed-width (0.0, 0.5, 1.0), so changing 0.0 to
1.0 and vice versa is safe without making a second multi-gigabyte copy.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def relabel(path: Path) -> tuple[int, int, int]:
    records = 0
    decisive = 0
    coherent_before = 0

    # Pass one is deliberately read-only. This makes the operation idempotent
    # in the safe sense: an already-correct or malformed corpus is rejected
    # before any bytes are changed.
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                break

            parts = line.rstrip(b"\r\n").rsplit(b" | ", 2)
            if len(parts) != 3:
                raise ValueError(f"{path}:{records + 1}: expected FEN | score | result")
            try:
                score = int(parts[1])
            except ValueError as exc:
                raise ValueError(f"{path}:{records + 1}: invalid score {parts[1]!r}") from exc

            result = parts[2]
            if result not in (b"0.0", b"0.5", b"1.0"):
                raise ValueError(f"{path}:{records + 1}: invalid result {result!r}")

            records += 1
            if result != b"0.5" and score != 0:
                decisive += 1
                coherent_before += (result == b"1.0") == (score > 0)

    if records == 0:
        raise ValueError(f"{path}: empty corpus")
    if decisive == 0:
        raise ValueError(f"{path}: no decisive records available for coherence check")

    before = coherent_before / decisive
    after = 1.0 - before
    if before > 0.10 or after < 0.90:
        raise ValueError(
            f"{path}: refusing relabel: expected inverted labels, "
            f"coherence was {before:.1%} before and {after:.1%} after"
        )

    with path.open("r+b") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            result = line.rstrip(b"\r\n")[-3:]
            if result != b"0.5":
                stream.seek(offset + len(line.rstrip(b"\r\n")) - 3)
                stream.write(b"1.0" if result == b"0.0" else b"0.0")
                stream.seek(offset + len(line))
    return records, decisive, coherent_before


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    total_records = total_decisive = total_coherent = 0
    for path in args.paths:
        records, decisive, coherent = relabel(path)
        total_records += records
        total_decisive += decisive
        total_coherent += coherent
        print(f"{path}: relabelled {records:,} records")

    before = total_coherent / total_decisive
    print(
        f"coherence: {before:.2%} before -> {1.0 - before:.2%} after "
        f"({total_records:,} records)"
    )


if __name__ == "__main__":
    main()
