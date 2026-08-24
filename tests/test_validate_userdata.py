from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate-userdata.sh"
LAUNCHER = ROOT / "scripts" / "launch-nnue-training.sh"


def run_validator(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(VALIDATOR), str(path)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_training_launcher_passes_raw_userdata():
    result = run_validator(LAUNCHER)
    assert result.returncode == 0, result.stderr
    assert "USER-DATA AWS ENCODING OK" in result.stdout


def test_validator_rejects_preencoded_userdata(tmp_path: Path):
    bad_launcher = tmp_path / "launch.sh"
    source = LAUNCHER.read_text()
    source = source.replace(
        '# ---------- Launch instance ----------',
        'USER_DATA_B64=$(echo "$USER_DATA" | base64)\n\n# ---------- Launch instance ----------',
    ).replace('--user-data "$USER_DATA"', '--user-data "$USER_DATA_B64"')
    bad_launcher.write_text(source)

    result = run_validator(bad_launcher)
    assert result.returncode == 3
    assert "must not be pre-encoded" in result.stderr
