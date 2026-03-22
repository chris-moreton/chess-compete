#!/bin/bash
# Launch an EC2 spot instance to run a H2H engine match
#
# Usage:
#   ./scripts/launch-h2h-match.sh v1.0.36 v1.0.37-rc1
#   ./scripts/launch-h2h-match.sh v1.0.36 v1.0.37-rc1 --games 5000 --tc "0/1:00+0.5"
#   ./scripts/launch-h2h-match.sh v1.0.36 v1.0.37-rc1 --games 5000 --type c6a.48xlarge
#
# The instance builds both engines from git tags, runs cutechess-cli,
# reports results live to the dashboard, uploads PGN, then self-terminates.
#
# Prerequisites:
#   - AWS CLI configured
#   - .env file with SPSA_API_URL and SPSA_API_KEY
#   - cutechess-cli Linux binary in S3 or installable

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
INSTANCE_TYPE="c6a.24xlarge"  # 96 vCPUs
GAMES=5000
TC="0/1:00+0.5"
HASH=128
THREADS=1
REGION="us-west-2"  # Default to cheapest region
MAX_SPOT_PRICE="4.00"
TARGET_NPS=2000000  # 2M NPS reference for timemult

# ---------- Parse arguments ----------
ENGINE1=""
ENGINE2=""
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --games)     GAMES="$2"; shift 2 ;;
        --tc)        TC="$2"; shift 2 ;;
        --type)      INSTANCE_TYPE="$2"; shift 2 ;;
        --hash)      HASH="$2"; shift 2 ;;
        --threads)   THREADS="$2"; shift 2 ;;
        --region|-r) REGION="$2"; shift 2 ;;
        --target-nps) TARGET_NPS="$2"; shift 2 ;;
        --no-pilot) NO_PILOT=true; shift ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        -*)          echo "Unknown option: $1"; exit 1 ;;
        *)           POSITIONAL+=("$1"); shift ;;
    esac
done

if [ ${#POSITIONAL[@]} -lt 2 ]; then
    echo "Usage: $0 <engine1_tag> <engine2_tag> [options]"
    exit 1
fi
ENGINE1="${POSITIONAL[0]}"
ENGINE2="${POSITIONAL[1]}"

# ---------- Validate ----------
if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI not found"
    exit 1
fi
if [ -z "$SPSA_API_URL" ] || [ -z "$SPSA_API_KEY" ]; then
    echo "Error: SPSA_API_URL and SPSA_API_KEY must be set in .env"
    exit 1
fi

# Get vCPU count for this instance type
VCPUS=$(aws ec2 describe-instance-types --instance-types "$INSTANCE_TYPE" \
    --query 'InstanceTypes[0].VCpuInfo.DefaultVCpus' --output text)
CONCURRENCY=$((VCPUS / 2))

# Estimate time (use python3 for float math - bc is unreliable on macOS)
AVG_SECS_PER_GAME=135
read -r EST_HOURS MAX_HOURS <<< $(python3 -c "
batches = ($GAMES + $CONCURRENCY - 1) // $CONCURRENCY
est = batches * $AVG_SECS_PER_GAME / 3600
max_h = max(2, int(est + 1.5))
print(f'{est:.1f} {max_h}')
")
SHUTDOWN_MINUTES=$((MAX_HOURS * 60))

echo "H2H Match: $ENGINE1 vs $ENGINE2"
echo "  Instance:    $INSTANCE_TYPE ($VCPUS vCPUs)"
echo "  Concurrency: $CONCURRENCY games"
echo "  Games:       $GAMES"
echo "  TC:          $TC"
echo "  Region:      $REGION"
echo "  Est. time:   ~${EST_HOURS}h (auto-terminate after ${MAX_HOURS}h)"
echo ""

# Get spot price estimate
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
exec > >(tee /var/log/h2h-match.log) 2>&1
set -e

echo "=== $(date) Starting H2H match setup ==="

# Schedule auto-termination
echo "=== Auto-shutdown scheduled in __MAX_HOURS__ hour(s) ==="
shutdown -h +__SHUTDOWN_MINUTES__

# Hard deadline watchdog
(
    sleep $((__SHUTDOWN_MINUTES__ * 60 + 300))
    echo "=== $(date) WATCHDOG: hard deadline reached, self-terminating ==="
    INSTANCE_ID=$(curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/instance-id || true)
    REGION=$(curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/placement/region || true)
    if [ -n "$INSTANCE_ID" ] && [ -n "$REGION" ]; then
        aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" 2>&1 || true
    fi
    shutdown -h now 2>/dev/null || true
) &
disown

# Install system packages (Ubuntu)
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3 python3-pip python3-venv git cutechess awscli

# Download engine binaries from GitHub releases
su - ubuntu -c '
    cd ~
    git clone https://github.com/chris-moreton/chess-compete.git
    cd chess-compete
    python3 -m pip install --break-system-packages -r requirements.txt

    REPO="chris-moreton/rusty-rival"

    # Download engine 1
    echo "=== Downloading __ENGINE1__ ==="
    curl -sL "https://github.com/${REPO}/releases/download/__ENGINE1__/rusty-rival-__ENGINE1__-linux-x86_64" -o ~/engine1
    chmod +x ~/engine1

    # Download engine 2
    echo "=== Downloading __ENGINE2__ ==="
    curl -sL "https://github.com/${REPO}/releases/download/__ENGINE2__/rusty-rival-__ENGINE2__-linux-x86_64" -o ~/engine2
    chmod +x ~/engine2

    # Verify downloads
    ~/engine1 <<< "uci" | head -1
    ~/engine2 <<< "uci" | head -1
'

echo "=== $(date) Engines ready, starting match ==="

# Run the match with API reporting
su - ubuntu -c '
    cd ~/chess-compete
    python3 scripts/h2h-reporter.py \
        --engine1 ~/engine1 --tag1 "__ENGINE1__" \
        --engine2 ~/engine2 --tag2 "__ENGINE2__" \
        --games __GAMES__ \
        --tc "__TC__" \
        --concurrency __CONCURRENCY__ \
        --hash __HASH__ \
        --threads __THREADS__ \
        --target-nps __TARGET_NPS__ \
        --book ~/chess-compete/openings/8moves_v3.pgn \
        --book-format pgn \
        --api-url "__API_URL__" \
        --api-key "__API_KEY__"
'

echo "=== $(date) Match complete, shutting down ==="
shutdown -h now
USERDATA
)

# Substitute values
USER_DATA="${USER_DATA//__ENGINE1__/$ENGINE1}"
USER_DATA="${USER_DATA//__ENGINE2__/$ENGINE2}"
USER_DATA="${USER_DATA//__GAMES__/$GAMES}"
USER_DATA="${USER_DATA//__TC__/$TC}"
USER_DATA="${USER_DATA//__CONCURRENCY__/$CONCURRENCY}"
USER_DATA="${USER_DATA//__HASH__/$HASH}"
USER_DATA="${USER_DATA//__THREADS__/$THREADS}"
USER_DATA="${USER_DATA//__API_URL__/$SPSA_API_URL}"
USER_DATA="${USER_DATA//__API_KEY__/$SPSA_API_KEY}"
USER_DATA="${USER_DATA//__TARGET_NPS__/$TARGET_NPS}"
USER_DATA="${USER_DATA//__MAX_HOURS__/$MAX_HOURS}"
USER_DATA="${USER_DATA//__SHUTDOWN_MINUTES__/$SHUTDOWN_MINUTES}"

USER_DATA_B64=$(echo "$USER_DATA" | base64)

# ---------- Launch instance ----------
REGION_ARGS=(--region "$REGION")

# Auto-detect Ubuntu AMI
AMI_ID=$(aws ec2 describe-images "${REGION_ARGS[@]}" \
    --owners 099720109477 \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*" "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text)

echo "Launching spot instance in $REGION (AMI: $AMI_ID)..."

# Launch via run-instances with spot request
INSTANCE_ID=$(aws ec2 run-instances "${REGION_ARGS[@]}" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"'"$MAX_SPOT_PRICE"'","SpotInstanceType":"one-time"}}' \
    --iam-instance-profile Name=SSMInstanceProfile \
    --instance-initiated-shutdown-behavior terminate \
    --user-data "$USER_DATA_B64" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=h2h-match-${ENGINE1}-vs-${ENGINE2}}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo ""
echo "================================================"
echo "Instance launched: $INSTANCE_ID"
echo "Region: $REGION"
echo "Type: $INSTANCE_TYPE ($VCPUS vCPUs, $CONCURRENCY concurrent games)"
echo "Match: $ENGINE1 vs $ENGINE2 ($GAMES games)"
echo "Auto-terminate: ${MAX_HOURS}h"
echo "================================================"
echo ""
echo "Monitor: $SPSA_API_URL/h2h"
echo "Logs:    aws ssm start-session --target $INSTANCE_ID --region $REGION"
echo "         sudo tail -f /var/log/h2h-match.log"
echo ""
echo "Terminate early:"
echo "  aws ec2 terminate-instances --region $REGION --instance-ids $INSTANCE_ID"
