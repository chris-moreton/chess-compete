from pathlib import Path

import pytest

from scripts.relabel_nnue_results import relabel


def test_relabel_flips_decisive_results_in_place(tmp_path: Path):
    corpus = tmp_path / "sample.txt"
    original = (
        "8/8/8/8/8/8/8/K6k w - - 0 1 | 250 | 0.0\n"
        "8/8/8/8/8/8/8/K6k b - - 0 1 | -100 | 1.0\n"
        "8/8/8/8/8/8/8/K6k w - - 0 1 | 0 | 0.5\n"
    )
    corpus.write_text(original)

    assert relabel(corpus) == (3, 2, 0)
    assert corpus.stat().st_size == len(original)
    assert corpus.read_text().endswith("| 250 | 1.0\n8/8/8/8/8/8/8/K6k b - - 0 1 | -100 | 0.0\n8/8/8/8/8/8/8/K6k w - - 0 1 | 0 | 0.5\n")


def test_relabel_refuses_already_coherent_data(tmp_path: Path):
    corpus = tmp_path / "sample.txt"
    corpus.write_text("8/8/8/8/8/8/8/K6k w - - 0 1 | 250 | 1.0\n")

    with pytest.raises(ValueError, match="expected inverted labels"):
        relabel(corpus)
    assert corpus.read_text().endswith("| 250 | 1.0\n")


def test_relabel_rejects_malformed_records(tmp_path: Path):
    corpus = tmp_path / "sample.txt"
    corpus.write_text("not a training record\n")

    with pytest.raises(ValueError, match="expected FEN"):
        relabel(corpus)
