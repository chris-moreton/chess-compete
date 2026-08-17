"""The live LMP contract rejects an engine-only threshold change."""

import subprocess
import sys
import shutil
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
CHECK = REPO / "scripts" / "check_lmp_contract.py"


def _engine_fixture(tmp_path: Path, thresholds: str) -> Path:
    constants = tmp_path / "engine" / "src" / "engine_constants.rs"
    constants.parent.mkdir(parents=True)
    constants.write_text(
        "pub const LMP_MOVE_THRESHOLDS: [u8; 9] = "
        f"[{thresholds}];\n"
    )
    return constants.parent.parent


def _run(engine_root: Path, spsa_root: Path = REPO):
    return subprocess.run(
        [sys.executable, str(CHECK), "--engine-root", str(engine_root), "--spsa-root", str(spsa_root)],
        text=True,
        capture_output=True,
        check=False,
    )


def test_live_contract_accepts_matching_engine_fixture(tmp_path):
    result = _run(_engine_fixture(tmp_path, "0, 9, 6, 9, 19, 28, 39, 52, 67"))

    assert result.returncode == 0, result.stderr


def test_live_contract_rejects_engine_only_threshold_drift(tmp_path):
    result = _run(_engine_fixture(tmp_path, "0, 9, 6, 9, 20, 28, 39, 52, 67"))

    assert result.returncode == 1
    assert "lmp_threshold_depth4=19 but engine ships 20" in result.stderr


def test_live_contract_rejects_stale_lmp_substitution_pattern(tmp_path):
    spsa_root = tmp_path / "chess-compete"
    shutil.copytree(REPO / "compete", spsa_root / "compete")
    build = spsa_root / "compete" / "spsa" / "build.py"
    build.write_text(build.read_text().replace(r"\[u8; 9\]", r"\[u8; 4\]", 1))

    result = _run(
        _engine_fixture(tmp_path, "0, 9, 6, 9, 19, 28, 39, 52, 67"),
        spsa_root,
    )

    assert result.returncode == 1
    assert "lmp_move_thresholds.pattern must match the live engine exactly once" in result.stderr


def test_live_contract_accepts_rustfmt_trailing_comma(tmp_path):
    result = _run(_engine_fixture(tmp_path, "0, 9, 6, 9, 19, 28, 39, 52, 67,"))

    assert result.returncode == 0, result.stderr
