#!/usr/bin/env python3
"""Validate and deterministically deduplicate Rust self-play NNUE records.

Rusty Rival emits white-relative ``FEN | score | result`` records in zstd
shards. This filter accepts one decompressed shard at a time and shares a small
SQLite index across invocations, so duplicate positions are removed across the
whole corpus without keeping hundreds of millions of keys in RAM.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sqlite3
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import BinaryIO

try:
    from scripts.relabel_nnue_results import validate_fen
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    from relabel_nnue_results import validate_fen


EXPECTED_PROVENANCE = {
    "format": "rusty-rival-selfplay-v1",
    "score_pov": "white",
    "result_pov": "white",
}


def position_key(fen: bytes) -> bytes:
    """Return a stable 128-bit key over the state represented by Zobrist."""
    fields = fen.split()
    if len(fields) != 6:
        raise ValueError("FEN must contain six fields")
    # Clocks are history metadata, not part of the chess position. A 128-bit
    # digest keeps the on-disk dedup index bounded while retaining the same
    # collision scale as Rival's u128 Zobrist lock.
    return hashlib.blake2b(b" ".join(fields[:4]), digest_size=16).digest()


def validate_provenance(path: Path) -> None:
    """Require an explicit Rusty Rival white-relative shard contract."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ValueError(f"missing self-play provenance marker: {path}") from exc
    values: dict[str, str] = {}
    for line in lines:
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in values:
            raise ValueError(f"invalid self-play provenance marker: {path}")
        values[key] = value
    for key, expected in EXPECTED_PROVENANCE.items():
        if values.get(key) != expected:
            raise ValueError(f"provenance must declare {key}={expected}")


def _digest(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.digest()


def _publish_prepared(connection: sqlite3.Connection, shard: str, temporary: Path, output: Path) -> None:
    """Materialise a committed prepared shard and publish it atomically."""
    if not temporary.exists():
        with temporary.open("xb") as destination:
            for (record,) in connection.execute(
                "SELECT record FROM prepared_records WHERE shard = ? ORDER BY ordinal",
                (shard,),
            ):
                destination.write(record)
            destination.flush()
            os.fsync(destination.fileno())
    temporary.replace(output)
    directory = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _finish_publication(connection: sqlite3.Connection, shard: str) -> None:
    """Record publication and discard the recoverable staging rows."""
    connection.execute("BEGIN IMMEDIATE")
    connection.execute("UPDATE prepared_shards SET published = 1 WHERE shard = ?", (shard,))
    connection.execute("DELETE FROM prepared_records WHERE shard = ?", (shard,))
    connection.commit()


def prepare(source: BinaryIO, output: Path, seen_db: Path, provenance: Path) -> tuple[int, int]:
    """Validate ``source`` and recoverably write unseen records to ``output``."""
    validate_provenance(provenance)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen_db.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    shard = str(output.resolve())
    records = kept = 0
    committed = False

    connection = sqlite3.connect(seen_db)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE IF NOT EXISTS seen (position BLOB PRIMARY KEY) WITHOUT ROWID")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS prepared_shards "
            "(shard TEXT PRIMARY KEY, records INTEGER NOT NULL, kept INTEGER NOT NULL, "
            "digest BLOB NOT NULL, published INTEGER NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS prepared_records "
            "(shard TEXT NOT NULL, ordinal INTEGER NOT NULL, record BLOB NOT NULL, "
            "PRIMARY KEY (shard, ordinal)) WITHOUT ROWID"
        )
        state = connection.execute(
            "SELECT records, kept, digest, published FROM prepared_shards WHERE shard = ?",
            (shard,),
        ).fetchone()
        if state is not None:
            records, kept, expected_digest, published = state
            if output.exists():
                if _digest(output) != expected_digest:
                    raise ValueError(f"published shard digest mismatch: {output}")
            elif published:
                raise FileNotFoundError(f"published shard was removed: {output}")
            else:
                temporary.unlink(missing_ok=True)
                _publish_prepared(connection, shard, temporary, output)
                if _digest(output) != expected_digest:
                    raise ValueError(f"recovered shard digest mismatch: {output}")
            if not published:
                _finish_publication(connection, shard)
            return records, kept

        if output.exists():
            raise FileExistsError(f"refusing to overwrite untracked output {output}")
        temporary.unlink(missing_ok=True)
        connection.execute("BEGIN IMMEDIATE")
        with temporary.open("xb") as destination:
            for line_number, line in enumerate(source, 1):
                parts = line.rstrip(b"\r\n").rsplit(b" | ", 2)
                if len(parts) != 3:
                    raise ValueError(f"record {line_number}: expected FEN | score | result")
                try:
                    validate_fen(parts[0])
                except ValueError as exc:
                    raise ValueError(f"record {line_number}: invalid FEN: {exc}") from exc
                try:
                    int(parts[1])
                except ValueError as exc:
                    raise ValueError(f"record {line_number}: invalid score {parts[1]!r}") from exc
                if parts[2] not in (b"0.0", b"0.5", b"1.0"):
                    raise ValueError(f"record {line_number}: invalid result {parts[2]!r}")

                records += 1
                inserted = connection.execute(
                    "INSERT OR IGNORE INTO seen(position) VALUES (?)",
                    (position_key(parts[0]),),
                ).rowcount
                if inserted:
                    record = b" | ".join(parts) + b"\n"
                    destination.write(record)
                    connection.execute(
                        "INSERT INTO prepared_records(shard, ordinal, record) VALUES (?, ?, ?)",
                        (shard, kept, record),
                    )
                    kept += 1
            destination.flush()
            os.fsync(destination.fileno())
        digest = _digest(temporary)
        connection.execute(
            "INSERT INTO prepared_shards(shard, records, kept, digest, published) "
            "VALUES (?, ?, ?, ?, 0)",
            (shard, records, kept, digest),
        )
        connection.commit()
        committed = True
        _publish_prepared(connection, shard, temporary, output)
        _finish_publication(connection, shard)
    except Exception:
        if not committed:
            connection.rollback()
            temporary.unlink(missing_ok=True)
        raise
    finally:
        connection.close()
    return records, kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="decompressed text shard, or - for stdin")
    parser.add_argument("output", type=Path)
    parser.add_argument("--seen-db", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    args = parser.parse_args()

    stream = nullcontext(sys.stdin.buffer) if args.input == "-" else Path(args.input).open("rb")
    with stream as source:
        records, kept = prepare(source, args.output, args.seen_db, args.provenance)
    print(f"{args.output}: kept {kept:,} of {records:,} records ({records - kept:,} duplicates)")


if __name__ == "__main__":
    main()
