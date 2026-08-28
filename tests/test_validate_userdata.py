from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate-userdata.sh"
LAUNCHER = ROOT / "scripts" / "launch-nnue-training.sh"
DATAGEN_LAUNCHER = ROOT / "scripts" / "launch-datagen.sh"
SHARD_NAMER = ROOT / "scripts" / "nnue-shard-name.sh"


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


def test_datagen_launcher_passes_raw_valid_userdata():
    result = run_validator(DATAGEN_LAUNCHER)
    assert result.returncode == 0, result.stderr
    assert "USER-DATA SYNTAX OK" in result.stdout
    assert "USER-DATA AWS ENCODING OK" in result.stdout


def test_datagen_launcher_uses_native_resumable_shards():
    source = DATAGEN_LAUNCHER.read_text()
    assert 'printf "datagen __GAMES__ __NODES__ %s __RANDOM_PLIES__ __THREADS__' in source
    assert 'aws s3 sync /var/lib/rival-datagen/' in source
    assert 'aws s3 sync "__S3_PATH__" /var/lib/rival-datagen/' in source
    assert 'while sleep 300' in source
    assert '--run-name)' in source
    assert 'nnue-data-selfplay/runs/${RUN_NAME}/' in source
    assert 'nnue-data-selfplay/${ENGINE_TAG}/n${NODES}/' not in source
    assert 'selfplay-format-v1.env' in source
    assert 'score_pov=white' in source
    assert 'cmp -s "$MANIFEST" "$EXISTING_MANIFEST"' in source
    assert 'generate-nnue-data.py' not in source
    assert 'USER_DATA_B64' not in source
    assert 'curl -fsIL "$PORTABLE_ASSET_URL"' in source
    assert source.index('curl -fsIL "$PORTABLE_ASSET_URL"') < source.index("aws ec2 run-instances")
    assert "ProcessorInfo.SupportedArchitectures" in source
    assert '" $ARCHITECTURES " != *" x86_64 "*' in source
    assert source.index("ProcessorInfo.SupportedArchitectures") < source.index("aws ec2 run-instances")


def test_cloud_jobs_terminate_on_success_and_failure():
    for launcher in (DATAGEN_LAUNCHER, LAUNCHER):
        source = launcher.read_text()
        assert 'trap persist_and_shutdown EXIT' in source
        assert source.index('aws s3 cp /var/log/') < source.index('shutdown -h now 2>/dev/null || true')
        assert "--instance-initiated-shutdown-behavior terminate" in source


def test_training_failure_handler_persists_log_and_checkpoints():
    source = LAUNCHER.read_text()
    assert 'aws s3 cp /var/log/nnue-training.log' in source
    assert 'aws s3 sync /home/ubuntu/nnue-train/checkpoints/' in source
    assert '--provenance "$provenance"' in source


def test_training_launcher_validates_values_substituted_into_userdata():
    source = LAUNCHER.read_text()
    assert '"$HIDDEN_SIZE" != "256" && "$HIDDEN_SIZE" != "512"' in source
    assert source.count('"$HIDDEN_SIZE" != "256" && "$HIDDEN_SIZE" != "512"') == 2
    assert '"$BRANCH" =~ ^[A-Za-z0-9._/-]+$' in source
    assert '"$NET_ID" =~ ^[A-Za-z0-9._-]+$' in source
    assert '"$EFFECTIVE_S3_DATA_PATH" =~ ^s3://' in source


def test_converted_shard_names_do_not_collide_for_ambiguous_paths():
    first = subprocess.run(
        [str(SHARD_NAMER), "a__b/c.zst"], check=True, capture_output=True, text=True
    ).stdout.strip()
    second = subprocess.run(
        [str(SHARD_NAMER), "a/b__c.zst"], check=True, capture_output=True, text=True
    ).stdout.strip()
    assert first != second
    assert first.startswith("c-")
    assert second.startswith("b__c-")


def test_bullet_converter_and_trainer_are_pinned_to_the_same_revision():
    revision = "c1a3433ba0ab4ce177a42240249fa8e1ecdbe45d"
    assert revision in LAUNCHER.read_text()
    assert f'rev = "{revision}"' in (ROOT / "scripts" / "nnue-train" / "Cargo.toml").read_text()


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
    assert "must pass raw" in result.stderr


def test_validator_rejects_encoded_userdata_via_alternate_variable(tmp_path: Path):
    bad_launcher = tmp_path / "launch.sh"
    source = LAUNCHER.read_text()
    source = source.replace(
        '# ---------- Launch instance ----------',
        'ENCODED_PAYLOAD=$(echo "$USER_DATA" | base64)\n\n# ---------- Launch instance ----------',
    ).replace('--user-data "$USER_DATA"', '--user-data "$ENCODED_PAYLOAD"')
    bad_launcher.write_text(source)

    result = run_validator(bad_launcher)
    assert result.returncode == 3
    assert "must pass raw" in result.stderr


def test_validator_rejects_invalid_userdata_shell_syntax(tmp_path: Path):
    bad_launcher = tmp_path / "launch.sh"
    source = LAUNCHER.read_text().replace(
        'echo "=== $(date) Verifying GPU ==="',
        'if then\necho "broken"',
    )
    bad_launcher.write_text(source)

    result = run_validator(bad_launcher)
    assert result.returncode != 0
    assert "syntax error" in result.stderr
