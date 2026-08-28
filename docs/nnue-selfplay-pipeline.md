# Rusty Rival self-play NNUE pipeline

This is the reproducible path from Rival self-play games to Bullet trainer
input. It replaces the legacy Python match generator for new self-play corpora;
the Stockfish-labelled `.txt` path remains supported for existing experiments.

## 1. Generate sharded data

Use a Rusty Rival release that contains the native `datagen` command:

```bash
./scripts/launch-datagen.sh <release-tag> \
  --games 50000 \
  --nodes 3000 \
  --hours 12
```

The 3,000-node default follows the fixed-CPU result from NET-397: deeper labels
improve a position at fixed volume, but did not beat buying more positions with
the same generation budget. Each run uploads to its own prefix:

```text
s3://chess-compete-builds/nnue-data-selfplay/runs/<run-name>/
```

Sealed shards are uploaded every five minutes. If a spot instance is
interrupted, reuse the run name printed by the first launch:

```bash
./scripts/launch-datagen.sh <release-tag> \
  --games 50000 \
  --nodes 3000 \
  --run-name <original-run-name>
```

The replacement downloads the durable prefix before invoking `datagen`, which
continues at the first missing game index. An explicit run name is the stable
S3 identity (`nnue-data-selfplay/runs/<name>`). Its `selfplay-format-v1.env`
manifest pins the engine tag, node budget, random-opening plies, and
white-relative label contract; resume rejects any mismatch rather than silently
starting another corpus.

The launcher prefers AVX2 when the instance CPU supports it, falls back to the
portable binary, passes raw cloud-init user data, and sets three termination
guards: an instance shutdown deadline, a hard watchdog, and EC2
`instance-initiated-shutdown-behavior=terminate`. The root EBS volume is also
deleted on termination. An exit trap shuts the instance down immediately after
either success or failure, so a failed setup cannot idle until the deadline.
The trap uploads the run log before shutdown, including on failures.
Before requesting an instance, the launcher also verifies that the portable
release asset exists; a typo or unreleased tag therefore fails at zero cloud
cost.

## 2. Train from the self-play prefix

Point the existing trainer launcher at the desired S3 prefix:

```bash
./scripts/launch-nnue-training.sh \
  --branch main \
  --s3-data-path s3://chess-compete-builds/nnue-data-selfplay/runs/<run-name>/ \
  --hidden-size 256 \
  --net-id rival-selfplay-256x2-<experiment> \
  --hours 12
```

The trainer handles formats by provenance:

- `.txt`: legacy Stockfish corpus; verifies and corrects its inverted result
  labels before conversion.
- `.zst`: native Rival corpus; decompresses, validates the already
  white-relative score/result convention against the run's required provenance
  marker, and deduplicates position state across every selected shard before
  Bullet conversion.

Deduplication uses a disk-backed SQLite set of 128-bit fingerprints over the
first four FEN fields (board, mover, castling and en-passant). Move clocks are
intentionally excluded, matching chess-position/Zobrist identity. Input paths
are sorted with the C locale, so the retained first occurrence is deterministic
across runs. Converted filenames include a SHA-256 digest of the complete S3
relative path, preventing identically named or separator-ambiguous shards from
overwriting one another. The index and raw shard are deleted after successful conversion to
limit peak disk use. The Bullet converter and CUDA trainer are pinned to the
same upstream revision, so the binary format and training implementation cannot
drift independently between runs.

Prepared shard publication uses recoverable SQLite staging. If conversion is
interrupted immediately before or after the atomic file rename, retry either
finishes the committed publication or safely rebuilds it without duplicating
positions. The training EXIT handler uploads its log and performs a final
best-effort checkpoint sync before shutdown.

## 3. Required validation before spending compute

```bash
python3 -m pytest -q tests/test_prepare_selfplay_nnue_data.py tests/test_validate_userdata.py
scripts/validate-userdata.sh scripts/launch-datagen.sh
scripts/validate-userdata.sh scripts/launch-nnue-training.sh
```

Do not launch a volume run until the Rusty Rival release tag is confirmed to
contain the reviewed resumability fixes from rusty-rival PR #75. No instance
should remain running after a generation or training job finishes.
