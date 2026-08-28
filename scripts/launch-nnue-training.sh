#!/bin/bash
# Launch a GPU EC2 instance to train an NNUE neural network.
#
# Usage:
#   ./scripts/launch-nnue-training.sh
#   ./scripts/launch-nnue-training.sh --on-demand
#   ./scripts/launch-nnue-training.sh --type g5.2xlarge
#
# The instance downloads training data from S3, converts to bullet format,
# trains the network, and uploads checkpoints to S3.
#
# Prerequisites:
#   - AWS CLI configured
#   - Training data in S3 as legacy Stockfish .txt files or Rust self-play .zst shards

set -euo pipefail

# ---------- Load .env ----------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "Error: .env file not found at $ENV_FILE"
    exit 1
fi

# ---------- Defaults ----------
INSTANCE_TYPE="g5.xlarge"  # NVIDIA A10G, 24GB VRAM, 4 vCPUs
REGION="us-east-1"  # Best GPU spot availability
MAX_HOURS=12
USE_SPOT=true
MAX_SPOT_PRICE="1.50"
S3_BUCKET="chess-compete-builds"
# Branch of chess-compete the instance checks out for scripts/nnue-train.
# The trainer config lives in git, so training the wrong branch silently
# trains the wrong architecture — always pass --branch for experiments.
BRANCH="main"
# Fraction of data shards to train on (NET-325). 1.0 = all. Shards are dropped
# alternately rather than by prefix, so the kept set spans every generation
# batch instead of over-representing one of them.
DATA_FRACTION="1.0"
# S3 prefix to pull training data from. Override to train on a different corpus
# (e.g. a depth-12 set) without touching the default depth-9 one (NET-326).
S3_DATA_PATH=""
NET_ID=""
SUPERBATCHES=""
HIDDEN_SIZE=""

# ---------- Parse arguments ----------
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)      INSTANCE_TYPE="$2"; shift 2 ;;
        --region|-r) REGION="$2"; shift 2 ;;
        --hours)     MAX_HOURS="$2"; shift 2 ;;
        --branch|-b) BRANCH="$2"; shift 2 ;;
        --data-fraction) DATA_FRACTION="$2"; shift 2 ;;
        --s3-data-path)  S3_DATA_PATH="$2"; shift 2 ;;
        --net-id)        NET_ID="$2"; shift 2 ;;
        --superbatches)  SUPERBATCHES="$2"; shift 2 ;;
        --hidden-size)   HIDDEN_SIZE="$2"; shift 2 ;;
        --on-demand) USE_SPOT=false; shift ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        -*)          echo "Unknown option: $1"; exit 1 ;;
        *)           shift ;;
    esac
done

if [[ "$HIDDEN_SIZE" != "256" && "$HIDDEN_SIZE" != "512" ]]; then
    echo "Error: --hidden-size must be explicitly set to 256 or 512" >&2
    exit 1
fi
if [[ -z "$NET_ID" || "$NET_ID" != *"${HIDDEN_SIZE}x2"* ]]; then
    echo "Error: --net-id must be set and contain ${HIDDEN_SIZE}x2" >&2
    exit 1
fi
if [[ ! "$MAX_HOURS" =~ ^[1-9][0-9]*$ \
    || ! "$INSTANCE_TYPE" =~ ^[A-Za-z0-9.]+$ \
    || ! "$REGION" =~ ^[a-z0-9-]+$ \
    || ! "$S3_BUCKET" =~ ^[a-z0-9.-]+$ \
    || ! "$MAX_SPOT_PRICE" =~ ^[0-9]+([.][0-9]+)?$ \
    || ! "$BRANCH" =~ ^[A-Za-z0-9._/-]+$ \
    || ! "$NET_ID" =~ ^[A-Za-z0-9._-]+$ \
    || ( "$HIDDEN_SIZE" != "256" && "$HIDDEN_SIZE" != "512" ) \
    || ! "$DATA_FRACTION" =~ ^(0[.][0-9]*[1-9][0-9]*|1([.]0+)?)$ \
    || ( -n "$SUPERBATCHES" && ! "$SUPERBATCHES" =~ ^[1-9][0-9]*$ ) ]]; then
    echo "Error: invalid training launcher argument" >&2
    exit 1
fi
EFFECTIVE_S3_DATA_PATH="${S3_DATA_PATH:-s3://${S3_BUCKET}/nnue-data-sf/}"
if [[ ! "$EFFECTIVE_S3_DATA_PATH" =~ ^s3://[a-z0-9.-]+/[A-Za-z0-9._/-]+$ ]]; then
    echo "Error: invalid S3 data path" >&2
    exit 1
fi

SHUTDOWN_MINUTES=$((MAX_HOURS * 60))
TIMESTAMP=$(date -u +%Y%m%d-%H%M%S)
LOG_NAME="nnue-training-${NET_ID}-${TIMESTAMP}.log"

echo "NNUE Training"
echo "  Instance:    $INSTANCE_TYPE"
echo "  Region:      $REGION"
echo "  Max hours:   $MAX_HOURS"
echo "  Branch:      $BRANCH"
echo "  Data frac:   $DATA_FRACTION"
echo "  Data path:   $EFFECTIVE_S3_DATA_PATH"
echo "  net_id:      ${NET_ID:-<default>}"
echo "  Hidden size: $HIDDEN_SIZE"
echo "  Superbatches:${SUPERBATCHES:-<default 600>}"
echo "  S3 data:     s3://${S3_BUCKET}/nnue-data-sf/"
echo "  S3 output:   s3://${S3_BUCKET}/nnue-checkpoints-sf/"
echo ""

# ---------- Build user-data script ----------
USER_DATA=$(cat <<'USERDATA'
#!/bin/bash
set -euo pipefail
exec > >(tee /var/log/nnue-training.log) 2>&1

echo "=== $(date) Starting NNUE training setup ==="

# Schedule auto-termination
shutdown -h +__SHUTDOWN_MINUTES__
persist_and_shutdown() {
    status=$?
    trap - EXIT
    set +e
    if command -v aws >/dev/null 2>&1; then
        aws s3 cp /var/log/nnue-training.log \
            "s3://__S3_BUCKET__/nnue-checkpoints-sf/logs/__LOG_NAME__" --only-show-errors
        if [ -d /home/ubuntu/nnue-train/checkpoints ]; then
            aws s3 sync /home/ubuntu/nnue-train/checkpoints/ \
                s3://__S3_BUCKET__/nnue-checkpoints-sf/ --only-show-errors
        fi
    fi
    shutdown -h now 2>/dev/null || true
    exit "$status"
}
trap persist_and_shutdown EXIT

# Hard deadline watchdog
(
    sleep $((__SHUTDOWN_MINUTES__ * 60 + 300))
    INSTANCE_ID=$(curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/instance-id || true)
    REGION=$(curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/placement/region || true)
    if [ -n "$INSTANCE_ID" ] && [ -n "$REGION" ]; then
        aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" 2>&1 || true
    fi
    shutdown -h now 2>/dev/null || true
) &
disown

# Deep Learning AMI has NVIDIA drivers + CUDA pre-installed
# Just install minimal extras without running apt-get update (avoids triggering upgrades)
export DEBIAN_FRONTEND=noninteractive
apt-get install -y git awscli zstd 2>/dev/null \
    || { apt-get update -y && apt-get install -y git awscli zstd; }

# Verify GPU
echo "=== $(date) Verifying GPU ==="
nvidia-smi
CUDA_DIR=$(find /usr/local -maxdepth 1 -name 'cuda-*' -type d | sort -V | tail -1)
echo "CUDA: $CUDA_DIR"
$CUDA_DIR/bin/nvcc --version

# Install Rust
su - ubuntu -c 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y'

# Write CUDA env for ubuntu user
cat > /etc/profile.d/cuda.sh << CUDAEOF
export CUDA_PATH=$CUDA_DIR
export PATH=$CUDA_DIR/bin:\$PATH
export LD_LIBRARY_PATH=$CUDA_DIR/lib64:\$LD_LIBRARY_PATH
CUDAEOF

echo "=== $(date) Downloading training data from S3 ==="

su - ubuntu -c '
    set -euo pipefail
    source ~/.cargo/env
    source /etc/profile.d/cuda.sh

    cd ~
    mkdir -p data

    # Download all training data files from S3
    aws s3 sync __S3_DATA_PATH__ ~/raw-data/
    echo "Downloaded $(find ~/raw-data -type f \( -name "*.txt" -o -name "*.zst" \) | wc -l) data files"

    # Clone the exact Bullet revision used by the trainer manifest. Letting the
    # converter and trainer follow a moving HEAD makes identical input produce
    # an unrepeatable toolchain.
    git clone https://github.com/jw1912/bullet.git
    git -C ~/bullet checkout --detach c1a3433ba0ab4ce177a42240249fa8e1ecdbe45d
    git clone -b __BRANCH__ https://github.com/chris-moreton/chess-compete.git
    echo "chess-compete branch: $(cd ~/chess-compete && git rev-parse --abbrev-ref HEAD) @ $(cd ~/chess-compete && git rev-parse --short HEAD)"

    # Build bullet-utils (no CUDA needed for this)
    echo "=== $(date) Building bullet-utils ==="
    cd ~/bullet
    echo "bullet revision: $(git rev-parse HEAD)"
    cargo build --release --package bullet-utils
    UTILS=~/bullet/target/release/bullet-utils

    # Convert legacy text and Rust self-play zstd shards to bullet format.
    #
    # Each .txt is deleted as soon as its .data is written (NET-322). Nothing
    # reads the raw text after conversion, and keeping both copies previously
    # left the 100GB volume at 99% full mid-training - one slightly larger
    # dataset away from failing hours into a GPU run. Deleting inside the loop
    # rather than after it also caps the peak, since full raw and converted
    # copies of the dataset never coexist. The originals live permanently in
    # S3, so this is not destructive.
    #
    # The && guard means a failed conversion leaves its input in place to
    # diagnose rather than silently discarding it.
    echo "=== $(date) Converting data to bullet format ==="
    mkdir -p ~/data
    mapfile -d "" raw_files < <(find ~/raw-data -type f \( -name "*.txt" -o -name "*.zst" \) -print0 | LC_ALL=C sort -z)
    if [ ${#raw_files[@]} -eq 0 ]; then
        echo "ERROR: no .txt or .zst training shards downloaded"
        exit 1
    fi
    for raw_file in "${raw_files[@]}"; do
        relative=${raw_file#"$HOME/raw-data/"}
        name=$(basename "$relative")
        # S3 prefixes can contain several runs whose shard basenames repeat or
        # whose paths collide under separator replacement. Hash the complete
        # relative path so every input has a stable, injective practical name.
        base=$(~/chess-compete/scripts/nnue-shard-name.sh "$relative")
        case "$name" in
            *.txt)
                echo "  Correcting legacy inverted result labels in $base..."
                python3 ~/chess-compete/scripts/relabel_nnue_results.py "$raw_file" || exit 1
                prepared="$raw_file"
                ;;
            *.zst)
                prepared=~/raw-data/${base}.prepared.txt
                provenance=$(dirname "$raw_file")/selfplay-format-v1.env
                echo "  Validating and deduplicating white-relative self-play shard $base..."
                zstd -qdc "$raw_file" \
                    | python3 ~/chess-compete/scripts/prepare_selfplay_nnue_data.py \
                        - "$prepared" --seen-db ~/selfplay-seen.sqlite \
                        --provenance "$provenance" \
                    || exit 1
                ;;
        esac
        if [ -s "$prepared" ]; then
            echo "  Converting $base..."
            $UTILS convert --from text --input "$prepared" --output ~/data/${base}.data || exit 1
        else
            echo "  Skipping $base: every position was already present in an earlier shard"
        fi
        rm -f "$prepared" "$raw_file"
    done
    # The dedup index is needed only during conversion. Free it before the
    # shuffle creates a transient second copy of each binary shard.
    rm -f ~/selfplay-seen.sqlite ~/selfplay-seen.sqlite-wal ~/selfplay-seen.sqlite-shm

    echo "Converted files:"
    ls -lh ~/data/*.data
    echo "Disk after conversion:"
    df -h /

    # Shuffle each file. This transiently needs an extra copy of the largest
    # shard, so headroom matters here too - hence the df either side.
    echo "=== $(date) Shuffling data files ==="
    for data_file in ~/data/*.data; do
        base=$(basename "$data_file" .data)
        echo "  Shuffling $base..."
        $UTILS shuffle --input "$data_file" --output ~/data/${base}_shuffled.data --mem-used-mb 4096
        mv ~/data/${base}_shuffled.data "$data_file"
    done

    # Reduce to the requested data fraction (NET-325/326). The selection logic
    # lives in scripts/select_data_fraction.py rather than inline: a heredoc
    # nested inside this single-quoted su block is unmaintainable, and when it
    # was corrupted the breakage was invisible to bash -n and cost two idle
    # 6-hour GPU instances.
    if [ "__DATA_FRACTION__" != "1.0" ]; then
        echo "=== $(date) Reducing to data fraction __DATA_FRACTION__ ==="
        python3 ~/chess-compete/scripts/select_data_fraction.py "__DATA_FRACTION__" ~/data || exit 1
        echo "  shards remaining: $(ls ~/data/*.data | wc -l)"
    fi

    # Skip interleave (OOMs on large datasets) - train on shuffled individual files
    echo "=== $(date) Skipping interleave - training on individual shuffled files ==="
    echo "Data files: $(ls ~/data/*.data | wc -l) files, $(du -sh ~/data/ | cut -f1) total"
    echo "Disk before training:"
    df -h /

    # Copy training code
    cp -r ~/chess-compete/scripts/nnue-train ~/nnue-train
    cd ~/nnue-train

    # Symlink data directory
    ln -sf ~/data data

    # Build and run the trainer with CUDA
    echo "=== $(date) Building NNUE trainer ==="
    cargo build --release 2>&1
    if [ $? -ne 0 ]; then
        echo "ERROR: cargo build failed!"
        exit 1
    fi

    echo "=== $(date) Starting training ==="
    echo "Training data: $(ls -lh data/*.data | wc -l) shards, $(du -sh data/ | cut -f1)"

    # Upload checkpoints periodically during training. Without this the only
    # sync is the one after cargo run returns, so a spot interruption or the
    # shutdown timer destroys the entire run. The trainer writes a checkpoint
    # every 50 superbatches (save_rate), so 20 minutes is comfortably finer
    # grained than the loss of work it protects against.
    (
        while true; do
            sleep 1200
            aws s3 sync checkpoints/ s3://__S3_BUCKET__/nnue-checkpoints-sf/ 2>/dev/null || true
        done
    ) &
    PERIODIC_SYNC_PID=$!

    NET_ID="__NET_ID__" NNUE_HIDDEN_SIZE="__HIDDEN_SIZE__" SUPERBATCHES="__SUPERBATCHES__" cargo run --release 2>&1

    kill $PERIODIC_SYNC_PID 2>/dev/null || true

    echo "=== $(date) Training complete ==="

    # Upload checkpoints to S3
    echo "=== $(date) Uploading checkpoints to S3 ==="
    aws s3 sync checkpoints/ s3://__S3_BUCKET__/nnue-checkpoints-sf/
    echo "Checkpoints uploaded to s3://__S3_BUCKET__/nnue-checkpoints-sf/"
'

echo "=== $(date) All done ==="
shutdown -h now
USERDATA
)

# Substitute values
USER_DATA="${USER_DATA//__S3_BUCKET__/$S3_BUCKET}"
USER_DATA="${USER_DATA//__BRANCH__/$BRANCH}"
USER_DATA="${USER_DATA//__DATA_FRACTION__/$DATA_FRACTION}"
USER_DATA="${USER_DATA//__S3_DATA_PATH__/$EFFECTIVE_S3_DATA_PATH}"
USER_DATA="${USER_DATA//__NET_ID__/$NET_ID}"
USER_DATA="${USER_DATA//__HIDDEN_SIZE__/$HIDDEN_SIZE}"
USER_DATA="${USER_DATA//__SUPERBATCHES__/$SUPERBATCHES}"
USER_DATA="${USER_DATA//__SHUTDOWN_MINUTES__/$SHUTDOWN_MINUTES}"
USER_DATA="${USER_DATA//__MAX_HOURS__/$MAX_HOURS}"
USER_DATA="${USER_DATA//__LOG_NAME__/$LOG_NAME}"

# ---------- Launch instance ----------
REGION_ARGS=(--region "$REGION")

# Deep Learning AMI with NVIDIA drivers pre-installed (avoids driver install reboot issue)
AMI_ID=$(aws ec2 describe-images "${REGION_ARGS[@]}" \
    --owners amazon \
    --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text)

SPOT_ARGS=()
if [ "$USE_SPOT" = true ]; then
    echo "Launching spot instance in $REGION (AMI: $AMI_ID)..."
    SPOT_ARGS=(--instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"'"$MAX_SPOT_PRICE"'","SpotInstanceType":"one-time"}}')
else
    echo "Launching on-demand instance in $REGION (AMI: $AMI_ID)..."
fi

# AWS CLI base64-encodes the user-data payload for RunInstances. Passing a
# pre-encoded value here makes cloud-init receive inert base64 text instead of
# a shell script (observed on cloud-init 26.1 / AWS CLI 2.31).
INSTANCE_ID=$(aws ec2 run-instances "${REGION_ARGS[@]}" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    ${SPOT_ARGS[@]+"${SPOT_ARGS[@]}"} \
    --iam-instance-profile Name=SSMInstanceProfile \
    --instance-initiated-shutdown-behavior terminate \
    --user-data "$USER_DATA" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=nnue-training}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo ""
echo "================================================"
echo "Instance launched: $INSTANCE_ID"
echo "Region: $REGION"
echo "Type: $INSTANCE_TYPE"
echo "Auto-terminate: ${MAX_HOURS}h"
echo "Checkpoints: s3://${S3_BUCKET}/nnue-checkpoints-sf/"
echo "================================================"
echo ""
echo "Logs: aws ssm start-session --target $INSTANCE_ID --region $REGION"
echo "      sudo tail -f /var/log/nnue-training.log"
echo ""
echo "Terminate early:"
echo "  aws ec2 terminate-instances --region $REGION --instance-ids $INSTANCE_ID"
