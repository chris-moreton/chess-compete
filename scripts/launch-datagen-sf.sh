#!/bin/bash
# Launch EC2 spot instances to generate NNUE training data using Stockfish.
#
# Usage:
#   ./scripts/launch-datagen-sf.sh --games 500000
#   ./scripts/launch-datagen-sf.sh --games 500000 --workers 8
#
# Uses Stockfish at depth 9 with 8 random opening plies for diversity.
# Data uploaded to s3://chess-compete-builds/nnue-data-sf/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"
if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
fi

GAMES=500000
WORKERS=4
DEPTH=9
HASH=16
REGION="us-west-2"
S3_BUCKET="chess-compete-builds"
S3_PATH="s3://${S3_BUCKET}/nnue-data-sf/"
INSTANCE_TYPE="c6a.4xlarge"

while [[ $# -gt 0 ]]; do
    case $1 in
        --games)    GAMES="$2"; shift 2 ;;
        --workers)  WORKERS="$2"; shift 2 ;;
        --depth)    DEPTH="$2"; shift 2 ;;
        --region|-r) REGION="$2"; shift 2 ;;
        -h|--help)  echo "Usage: $0 [--games N] [--workers N] [--depth N]"; exit 0 ;;
        *)          echo "Unknown: $1"; exit 1 ;;
    esac
done

GAMES_PER_WORKER=$(( (GAMES + WORKERS - 1) / WORKERS ))
REGIONS=("us-east-1" "us-east-2" "us-west-2" "eu-west-2")

echo "Stockfish NNUE Data Generation"
echo "  Total games:     $GAMES"
echo "  Workers:         $WORKERS ($GAMES_PER_WORKER games each)"
echo "  Depth:           $DEPTH (fixed)"
echo "  Random plies:    8"
echo "  S3 output:       $S3_PATH"
echo ""

launched=0
for i in $(seq 1 $WORKERS); do
    region_idx=$(( (i - 1) % ${#REGIONS[@]} ))
    region="${REGIONS[$region_idx]}"

    TIMESTAMP=$(date +%Y%m%d-%H%M%S)-${i}
    OUTPUT_FILENAME="sf_d${DEPTH}_${GAMES_PER_WORKER}games_${TIMESTAMP}.txt"

    USER_DATA=$(cat <<'USERDATA'
#!/bin/bash
exec > >(tee /var/log/datagen.log) 2>&1

echo "=== $(date) Starting Stockfish datagen ==="
shutdown -h +300

# Watchdog
(sleep 18300; INSTANCE_ID=$(curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/instance-id || true); REGION=$(curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/placement/region || true); [ -n "$INSTANCE_ID" ] && [ -n "$REGION" ] && aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" 2>&1 || true; shutdown -h now 2>/dev/null || true) & disown

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3 python3-pip git awscli

# Install Stockfish
curl -L https://github.com/official-stockfish/Stockfish/releases/download/sf_17.1/stockfish-ubuntu-x86-64-sse41-popcnt.tar -o /tmp/stockfish.tar
tar -xf /tmp/stockfish.tar -C /tmp
mv /tmp/stockfish/stockfish-ubuntu-x86-64-sse41-popcnt /usr/local/bin/stockfish
chmod +x /usr/local/bin/stockfish
rm -rf /tmp/stockfish.tar /tmp/stockfish

su - ubuntu -c '
    cd ~
    git clone https://github.com/chris-moreton/chess-compete.git
    cd chess-compete
    python3 -m pip install -r requirements.txt

    python3 scripts/generate-nnue-data.py \
        --engine /usr/local/bin/stockfish \
        --engine-tag stockfish-17.1 \
        --depth __DEPTH__ \
        --games __GAMES__ \
        --hash __HASH__ \
        --eval-noise 0 \
        --random-plies 8 \
        --fixed-depth \
        --concurrency __CONCURRENCY__ \
        --output /tmp/__OUTPUT_FILENAME__ \
        --upload "__S3_PATH__" \
        --api-url "__API_URL__" \
        --api-key "__API_KEY__"
'

echo "=== $(date) Done ==="
shutdown -h now
USERDATA
)

    VCPUS=$(aws ec2 describe-instance-types --instance-types "$INSTANCE_TYPE" --query 'InstanceTypes[0].VCpuInfo.DefaultVCpus' --output text 2>/dev/null || echo 48)
    CONCURRENCY=$VCPUS

    USER_DATA="${USER_DATA//__DEPTH__/$DEPTH}"
    USER_DATA="${USER_DATA//__GAMES__/$GAMES_PER_WORKER}"
    USER_DATA="${USER_DATA//__HASH__/$HASH}"
    USER_DATA="${USER_DATA//__CONCURRENCY__/$CONCURRENCY}"
    USER_DATA="${USER_DATA//__OUTPUT_FILENAME__/$OUTPUT_FILENAME}"
    USER_DATA="${USER_DATA//__S3_PATH__/$S3_PATH}"
    USER_DATA="${USER_DATA//__API_URL__/${SPSA_API_URL:-}}"
    USER_DATA="${USER_DATA//__API_KEY__/${SPSA_API_KEY:-}}"

    USER_DATA_B64=$(echo "$USER_DATA" | base64)

    AMI_ID=$(aws ec2 describe-images --region "$region" \
        --owners 099720109477 \
        --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" "Name=state,Values=available" \
        --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
        --output text)

    INSTANCE_ID=$(aws ec2 run-instances --region "$region" \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}' \
        --iam-instance-profile Name=SSMInstanceProfile \
        --instance-initiated-shutdown-behavior terminate \
        --user-data "$USER_DATA_B64" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=nnue-datagen-sf-${i}}]" \
        --query 'Instances[0].InstanceId' \
        --output text 2>/dev/null) || { echo "  Failed to launch in $region"; continue; }

    echo "  Worker $i: $INSTANCE_ID ($region, $GAMES_PER_WORKER games)"
    launched=$((launched + 1))
done

echo ""
echo "Launched $launched/$WORKERS workers"
echo "Monitor: ${SPSA_API_URL:-}/datagen"
