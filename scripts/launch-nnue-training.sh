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
#   - Training data in s3://chess-compete-builds/nnue-data/

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

# ---------- Parse arguments ----------
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)      INSTANCE_TYPE="$2"; shift 2 ;;
        --region|-r) REGION="$2"; shift 2 ;;
        --hours)     MAX_HOURS="$2"; shift 2 ;;
        --on-demand) USE_SPOT=false; shift ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        -*)          echo "Unknown option: $1"; exit 1 ;;
        *)           shift ;;
    esac
done

SHUTDOWN_MINUTES=$((MAX_HOURS * 60))

echo "NNUE Training"
echo "  Instance:    $INSTANCE_TYPE"
echo "  Region:      $REGION"
echo "  Max hours:   $MAX_HOURS"
echo "  S3 data:     s3://${S3_BUCKET}/nnue-data/"
echo "  S3 output:   s3://${S3_BUCKET}/nnue-checkpoints/"
echo ""

# ---------- Build user-data script ----------
USER_DATA=$(cat <<'USERDATA'
#!/bin/bash
exec > >(tee /var/log/nnue-training.log) 2>&1
set -e

echo "=== $(date) Starting NNUE training setup ==="

# Schedule auto-termination
shutdown -h +__SHUTDOWN_MINUTES__

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

# Install system packages (Deep Learning AMI already has NVIDIA drivers + CUDA)
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y git awscli build-essential

# Verify GPU + CUDA
nvidia-smi
CUDA_PATH=$(find /usr/local -maxdepth 1 -name 'cuda-*' -type d | sort -V | tail -1)
echo "CUDA_PATH: $CUDA_PATH"
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

nvcc --version

# Install Rust
su - ubuntu -c 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y'

echo "=== $(date) Downloading training data from S3 ==="

# Write CUDA env for ubuntu user
CUDA_PATH_RESOLVED=$(find /usr/local -maxdepth 1 -name 'cuda-*' -type d | sort -V | tail -1)
cat > /etc/profile.d/cuda.sh << CUDAEOF
export CUDA_PATH=$CUDA_PATH_RESOLVED
export PATH=$CUDA_PATH_RESOLVED/bin:\$PATH
export LD_LIBRARY_PATH=$CUDA_PATH_RESOLVED/lib64:\$LD_LIBRARY_PATH
CUDAEOF

su - ubuntu -c '
    source ~/.cargo/env
    source /etc/profile.d/cuda.sh

    cd ~
    mkdir -p data

    # Download all training data files from S3
    aws s3 sync s3://__S3_BUCKET__/nnue-data/ ~/raw-data/
    echo "Downloaded $(ls ~/raw-data/*.txt | wc -l) data files"

    # Clone bullet and chess-compete
    git clone https://github.com/jw1912/bullet.git
    git clone https://github.com/chris-moreton/chess-compete.git

    # Build bullet-utils (no CUDA needed for this)
    echo "=== $(date) Building bullet-utils ==="
    cd ~/bullet
    cargo build --release --package bullet-utils
    UTILS=~/bullet/target/release/bullet-utils

    # Convert text files to bullet binary format
    echo "=== $(date) Converting data to bullet format ==="
    for txt_file in ~/raw-data/*.txt; do
        base=$(basename "$txt_file" .txt)
        echo "  Converting $base..."
        $UTILS convert-to-chess-board "$txt_file" "data/${base}.data"
    done

    # Shuffle each file
    echo "=== $(date) Shuffling data files ==="
    for data_file in data/*.data; do
        echo "  Shuffling $(basename $data_file)..."
        $UTILS shuffle "$data_file"
    done

    # Interleave all files into one
    echo "=== $(date) Interleaving data files ==="
    data_files=$(ls data/*.data | tr "\n" " ")
    $UTILS interleave $data_files --output data/training.data
    rm -f data/v1.0.39_*.data  # Remove individual files to save disk

    echo "Training data ready: $(ls -lh data/training.data)"

    # Copy training code
    cp -r ~/chess-compete/scripts/nnue-train ~/nnue-train
    cd ~/nnue-train

    # Build the trainer with CUDA
    echo "=== $(date) Building NNUE trainer ==="
    cargo build --release 2>&1

    # Symlink data directory
    ln -sf ~/data data

    # Run training
    echo "=== $(date) Starting training ==="
    cargo run --release 2>&1

    echo "=== $(date) Training complete ==="

    # Upload checkpoints to S3
    echo "=== $(date) Uploading checkpoints to S3 ==="
    aws s3 sync checkpoints/ s3://__S3_BUCKET__/nnue-checkpoints/
    echo "Checkpoints uploaded to s3://__S3_BUCKET__/nnue-checkpoints/"
'

echo "=== $(date) All done, shutting down ==="
shutdown -h now
USERDATA
)

# Substitute values
USER_DATA="${USER_DATA//__S3_BUCKET__/$S3_BUCKET}"
USER_DATA="${USER_DATA//__SHUTDOWN_MINUTES__/$SHUTDOWN_MINUTES}"
USER_DATA="${USER_DATA//__MAX_HOURS__/$MAX_HOURS}"

USER_DATA_B64=$(echo "$USER_DATA" | base64)

# ---------- Launch instance ----------
REGION_ARGS=(--region "$REGION")

# Use Deep Learning AMI with NVIDIA drivers + CUDA pre-installed
AMI_ID=$(aws ec2 describe-images "${REGION_ARGS[@]}" \
    --owners amazon \
    --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text)

if [ "$AMI_ID" = "None" ] || [ -z "$AMI_ID" ]; then
    echo "ERROR: Deep Learning AMI not found in $REGION"
    exit 1
fi

SPOT_ARGS=()
if [ "$USE_SPOT" = true ]; then
    echo "Launching spot instance in $REGION (AMI: $AMI_ID)..."
    SPOT_ARGS=(--instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"'"$MAX_SPOT_PRICE"'","SpotInstanceType":"one-time"}}')
else
    echo "Launching on-demand instance in $REGION (AMI: $AMI_ID)..."
fi

INSTANCE_ID=$(aws ec2 run-instances "${REGION_ARGS[@]}" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    ${SPOT_ARGS[@]+"${SPOT_ARGS[@]}"} \
    --iam-instance-profile Name=SSMInstanceProfile \
    --instance-initiated-shutdown-behavior terminate \
    --user-data "$USER_DATA_B64" \
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
echo "Checkpoints: s3://${S3_BUCKET}/nnue-checkpoints/"
echo "================================================"
echo ""
echo "Logs: aws ssm start-session --target $INSTANCE_ID --region $REGION"
echo "      sudo tail -f /var/log/nnue-training.log"
echo ""
echo "Terminate early:"
echo "  aws ec2 terminate-instances --region $REGION --instance-ids $INSTANCE_ID"
