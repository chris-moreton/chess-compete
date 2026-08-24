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


def validate_fen(value: bytes) -> None:
    try:
        board, side, castling, ep, halfmove, fullmove = value.decode("ascii").split()
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("FEN must contain six ASCII fields") from exc

    ranks = board.split("/")
    if len(ranks) != 8:
        raise ValueError("FEN board must contain eight ranks")
    pieces = "prnbqkPRNBQK"
    for rank in ranks:
        squares = 0
        for symbol in rank:
            if symbol in pieces:
                squares += 1
            elif symbol in "12345678":
                squares += int(symbol)
            else:
                raise ValueError(f"invalid FEN board symbol {symbol!r}")
        if squares != 8:
            raise ValueError("each FEN rank must contain eight squares")
    if board.count("K") != 1 or board.count("k") != 1:
        raise ValueError("FEN must contain exactly one king of each colour")
    if side not in ("w", "b"):
        raise ValueError("invalid FEN side to move")
    if castling != "-" and (any(c not in "KQkq" for c in castling) or len(set(castling)) != len(castling)):
        raise ValueError("invalid FEN castling rights")
    if ep != "-" and (len(ep) != 2 or ep[0] not in "abcdefgh" or ep[1] not in "36"):
        raise ValueError("invalid FEN en-passant square")
    try:
        if int(halfmove) < 0 or int(fullmove) < 1:
            raise ValueError("invalid FEN move clocks")
    except ValueError as exc:
        raise ValueError("invalid FEN move clocks") from exc


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
                validate_fen(parts[0])
            except ValueError as exc:
                raise ValueError(f"{path}:{records + 1}: invalid FEN: {exc}") from exc
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
