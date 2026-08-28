import io
from pathlib import Path

import pytest

from scripts import prepare_selfplay_nnue_data as selfplay


prepare = selfplay.prepare


def record(fen: str, score: int, result: str) -> bytes:
    return f"{fen} | {score} | {result}\n".encode()


def provenance(tmp_path: Path, *, score_pov: str = "white") -> Path:
    marker = tmp_path / "selfplay-format-v1.env"
    marker.write_text(
        "format=rusty-rival-selfplay-v1\n"
        f"score_pov={score_pov}\n"
        "result_pov=white\n"
        "engine_tag=v1.0.56\n"
    )
    return marker


def test_prepare_preserves_white_relative_labels_and_deduplicates_across_shards(tmp_path: Path):
    database = tmp_path / "seen.sqlite"
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    position = "8/8/8/8/8/8/8/K6k w - - 0 1"
    same_position_different_clocks = "8/8/8/8/8/8/8/K6k w - - 37 92"
    distinct = "8/8/8/8/8/8/8/K6k b - - 0 1"

    marker = provenance(tmp_path)
    assert prepare(io.BytesIO(record(position, 250, "1.0")), first, database, marker) == (1, 1)
    assert prepare(
        io.BytesIO(record(same_position_different_clocks, 300, "0.0") + record(distinct, -50, "0.5")),
        second,
        database,
        marker,
    ) == (2, 1)

    assert first.read_bytes() == record(position, 250, "1.0")
    assert second.read_bytes() == record(distinct, -50, "0.5")


def test_prepare_rejects_malformed_input_without_output_or_seen_rows(tmp_path: Path):
    database = tmp_path / "seen.sqlite"
    output = tmp_path / "output.txt"
    good = record("8/8/8/8/8/8/8/K6k w - - 0 1", 10, "0.5")

    with pytest.raises(ValueError, match="invalid result"):
        prepare(
            io.BytesIO(good + record("8/8/8/8/8/8/8/K6k b - - 0 1", 20, "draw")),
            output,
            database,
            provenance(tmp_path),
        )

    assert not output.exists()
    retry = tmp_path / "retry.txt"
    assert prepare(io.BytesIO(good), retry, database, provenance(tmp_path)) == (1, 1)


def test_prepare_refuses_to_overwrite_output(tmp_path: Path):
    output = tmp_path / "output.txt"
    output.write_text("existing")

    with pytest.raises(FileExistsError):
        prepare(io.BytesIO(b""), output, tmp_path / "seen.sqlite", provenance(tmp_path))


def test_prepare_rejects_missing_or_non_white_provenance(tmp_path: Path):
    source = io.BytesIO(record("8/8/8/8/8/8/8/K6k w - - 0 1", 10, "0.5"))
    with pytest.raises(ValueError, match="missing self-play provenance"):
        prepare(source, tmp_path / "missing.txt", tmp_path / "seen.sqlite", tmp_path / "missing.env")
    with pytest.raises(ValueError, match="score_pov=white"):
        prepare(
            io.BytesIO(source.getvalue()),
            tmp_path / "black.txt",
            tmp_path / "seen.sqlite",
            provenance(tmp_path, score_pov="side-to-move"),
        )


def test_prepare_recovers_after_database_commit_before_publication(tmp_path: Path, monkeypatch):
    database = tmp_path / "seen.sqlite"
    output = tmp_path / "output.txt"
    marker = provenance(tmp_path)
    expected = record("8/8/8/8/8/8/8/K6k w - - 0 1", 10, "0.5")
    real_publish = selfplay._publish_prepared

    def interrupt(*_args):
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(selfplay, "_publish_prepared", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        prepare(io.BytesIO(expected), output, database, marker)
    assert not output.exists()

    monkeypatch.setattr(selfplay, "_publish_prepared", real_publish)
    assert prepare(io.BytesIO(b""), output, database, marker) == (1, 1)
    assert output.read_bytes() == expected


def test_prepare_recovers_after_publication_before_state_update(tmp_path: Path, monkeypatch):
    database = tmp_path / "seen.sqlite"
    output = tmp_path / "output.txt"
    marker = provenance(tmp_path)
    expected = record("8/8/8/8/8/8/8/K6k w - - 0 1", 10, "0.5")
    real_finish = selfplay._finish_publication

    def interrupt(*_args):
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(selfplay, "_finish_publication", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        prepare(io.BytesIO(expected), output, database, marker)
    assert output.read_bytes() == expected

    monkeypatch.setattr(selfplay, "_finish_publication", real_finish)
    assert prepare(io.BytesIO(b""), output, database, marker) == (1, 1)
    assert output.read_bytes() == expected
