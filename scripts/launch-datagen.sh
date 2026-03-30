#!/bin/bash
# Launch an EC2 spot instance to generate NNUE training data via self-play.
#
# Usage:
#   ./scripts/launch-datagen.sh v1.0.38 --games 50000
#   ./scripts/launch-datagen.sh v1.0.38 --games 100000 --depth 10 --type c6a.24xlarge
#
# The instance downloads the engine, runs self-play games, uploads
# training data to S3, then self-terminates.
#
# Prerequisites:
#   - AWS CLI configured
#   - .env file with SPSA_API_URL and SPSA_API_KEY (for S3 bucket access)

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
INSTANCE_TYPE="c6a.12xlarge"  # 48 vCPUs - good balance of cost/speed
GAMES=50000
DEPTH=8
HASH=16  # Per engine instance (many running concurrently)
REGION="us-west-2"
MAX_SPOT_PRICE="2.00"
S3_BUCKET="chess-compete-builds"

# ---------- Parse arguments ----------
ENGINE_TAG=""
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --games)     GAMES="$2"; shift 2 ;;
        --depth)     DEPTH="$2"; shift 2 ;;
        --type)      INSTANCE_TYPE="$2"; shift 2 ;;
        --hash)      HASH="$2"; shift 2 ;;
        --region|-r) REGION="$2"; shift 2 ;;
        --bucket)    S3_BUCKET="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        -*)          echo "Unknown option: $1"; exit 1 ;;
        *)           POSITIONAL+=("$1"); shift ;;
    esac
done

if [ ${#POSITIONAL[@]} -lt 1 ]; then
    echo "Usage: $0 <engine_tag> [options]"
    echo "  e.g. $0 v1.0.38 --games 50000"
    exit 1
fi
ENGINE_TAG="${POSITIONAL[0]}"

# ---------- Validate ----------
if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI not found"
    exit 1
fi

# Get vCPU count
VCPUS=$(aws ec2 describe-instance-types --instance-types "$INSTANCE_TYPE" \
    --query 'InstanceTypes[0].VCpuInfo.DefaultVCpus' --output text)
CONCURRENCY=$((VCPUS))  # One engine per vCPU (single-threaded engines)

# Estimate time and cost
read -r EST_HOURS MAX_HOURS <<< $(python3 -c "
# ~60 moves per game, ~0.05s per move at depth 8 with 2M NPS
secs_per_game = 60 * 0.05 * ($DEPTH / 8.0) ** 2  # rough scaling
batches = ($GAMES + $CONCURRENCY - 1) // $CONCURRENCY
est = batches * secs_per_game / 3600
max_h = max(2, int(est + 1.5))
print(f'{est:.1f} {max_h}')
")
SHUTDOWN_MINUTES=$((MAX_HOURS * 60))

S3_PATH="s3://${S3_BUCKET}/nnue-data/"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT_FILENAME="${ENGINE_TAG}_d${DEPTH}_${GAMES}games_${TIMESTAMP}.txt"

echo "NNUE Data Generation"
echo "  Engine:      $ENGINE_TAG"
echo "  Instance:    $INSTANCE_TYPE ($VCPUS vCPUs)"
echo "  Concurrency: $CONCURRENCY engines"
echo "  Games:       $GAMES"
echo "  Depth:       $DEPTH"
echo "  Region:      $REGION"
echo "  Est. time:   ~${EST_HOURS}h (auto-terminate after ${MAX_HOURS}h)"
echo "  S3 output:   ${S3_PATH}${OUTPUT_FILENAME}"
echo ""

# Spot price
SPOT_PRICE=$(aws ec2 describe-spot-price-history --region "$REGION" \
    --instance-types "$INSTANCE_TYPE" --product-descriptions "Linux/UNIX" \
    --start-time "$(date -u +%Y-%m-%dT%H:%M:%S)" \
    --query 'SpotPriceHistory[0].SpotPrice' --output text 2>/dev/null | head -1 | tr -d '[:space:]' || echo "unknown")
if [ -n "$SPOT_PRICE" ] && [ "$SPOT_PRICE" != "unknown" ] && [ "$SPOT_PRICE" != "None" ]; then
    EST_COST=$(python3 -c "print(f'{float(\"$SPOT_PRICE\") * $EST_HOURS:.2f}')")
    echo "  Spot price:  \$$SPOT_PRICE/hr (est. cost: ~\$$EST_COST)"
else
    echo "  Spot price:  unknown"
fi
echo ""

# ---------- Build user-data script ----------
USER_DATA=$(cat <<'USERDATA'
#!/bin/bash
exec > >(tee /var/log/datagen.log) 2>&1
set -e

echo "=== $(date) Starting NNUE data generation ==="

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

# Install packages (Ubuntu 22.04)
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3 python3-pip git awscli

# Download engine
su - ubuntu -c '
    cd ~
    git clone https://github.com/chris-moreton/chess-compete.git
    cd chess-compete
    python3 -m pip install -r requirements.txt

    REPO="chris-moreton/rusty-rival"
    echo "=== Downloading __ENGINE_TAG__ ==="
    curl -sL "https://github.com/${REPO}/releases/download/__ENGINE_TAG__/rusty-rival-__ENGINE_TAG__-linux-x86_64" -o ~/engine
    chmod +x ~/engine
    ~/engine <<< "uci" | head -1
'

echo "=== $(date) Engine ready, starting data generation ==="

# Generate data
su - ubuntu -c '
    cd ~/chess-compete
    python3 scripts/generate-nnue-data.py \
        --engine ~/engine \
        --depth __DEPTH__ \
        --games __GAMES__ \
        --hash __HASH__ \
        --concurrency __CONCURRENCY__ \
        --output /tmp/__OUTPUT_FILENAME__ \
        --upload "__S3_PATH__"
'

echo "=== $(date) Data generation complete, shutting down ==="
shutdown -h now
USERDATA
)

# Substitute values
USER_DATA="${USER_DATA//__ENGINE_TAG__/$ENGINE_TAG}"
USER_DATA="${USER_DATA//__GAMES__/$GAMES}"
USER_DATA="${USER_DATA//__DEPTH__/$DEPTH}"
USER_DATA="${USER_DATA//__HASH__/$HASH}"
USER_DATA="${USER_DATA//__CONCURRENCY__/$CONCURRENCY}"
USER_DATA="${USER_DATA//__OUTPUT_FILENAME__/$OUTPUT_FILENAME}"
USER_DATA="${USER_DATA//__S3_PATH__/$S3_PATH}"
USER_DATA="${USER_DATA//__MAX_HOURS__/$MAX_HOURS}"
USER_DATA="${USER_DATA//__SHUTDOWN_MINUTES__/$SHUTDOWN_MINUTES}"

USER_DATA_B64=$(echo "$USER_DATA" | base64)

# ---------- Launch instance ----------
REGION_ARGS=(--region "$REGION")

# Ubuntu 22.04 AMI
AMI_ID=$(aws ec2 describe-images "${REGION_ARGS[@]}" \
    --owners 099720109477 \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text)

echo "Launching spot instance in $REGION (AMI: $AMI_ID)..."

INSTANCE_ID=$(aws ec2 run-instances "${REGION_ARGS[@]}" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"'"$MAX_SPOT_PRICE"'","SpotInstanceType":"one-time"}}' \
    --iam-instance-profile Name=SSMInstanceProfile \
    --instance-initiated-shutdown-behavior terminate \
    --user-data "$USER_DATA_B64" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=nnue-datagen-${ENGINE_TAG}}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo ""
echo "================================================"
echo "Instance launched: $INSTANCE_ID"
echo "Region: $REGION"
echo "Type: $INSTANCE_TYPE ($VCPUS vCPUs, $CONCURRENCY concurrent engines)"
echo "Generating: $GAMES games at depth $DEPTH"
echo "Output: ${S3_PATH}${OUTPUT_FILENAME}"
echo "Auto-terminate: ${MAX_HOURS}h"
echo "================================================"
echo ""
echo "Logs: aws ssm start-session --target $INSTANCE_ID --region $REGION"
echo "      sudo tail -f /var/log/datagen.log"
echo ""
echo "Terminate early:"
echo "  aws ec2 terminate-instances --region $REGION --instance-ids $INSTANCE_ID"
