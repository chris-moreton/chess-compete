#!/bin/bash
# Launch SPSA worker EC2 instances
#
# Usage:
#   ./scripts/launch-ec2-workers.sh -H 4              # 1 instance, auto-terminate after 4 hours
#   ./scripts/launch-ec2-workers.sh -n 3 -H 6          # 3 instances, 6 hours
#   ./scripts/launch-ec2-workers.sh -n 3 -H 4 --spot   # 3 spot instances (cheaper)
#   ./scripts/launch-ec2-workers.sh -n 3 -c 8 -H 2     # 3 instances, 8 concurrent games, 2 hours
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - .env file with SPSA_API_URL and SPSA_API_KEY
#
# Watching logs:
#   aws ssm start-session --target <instance-id>
#   sudo tail -f /var/log/spsa-worker.log

set -e

# ---------- Load .env ----------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
    unset CONCURRENCY COMPUTER_NAME  # .env values are for local use, not EC2
else
    echo "Error: .env file not found at $ENV_FILE"
    exit 1
fi

# ---------- Configuration ----------
INSTANCE_TYPE="${INSTANCE_TYPE:-c7a.4xlarge}"
AMI_ID="${AMI_ID:-}"  # Auto-detected if empty
KEY_NAME="${KEY_NAME:-}"
SECURITY_GROUP="${SECURITY_GROUP:-}"
CONCURRENCY="${CONCURRENCY:-16}"
COUNT=1
MAX_HOURS=""
USE_SPOT=false

# ---------- Parse arguments ----------
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--count)       COUNT="$2"; shift 2 ;;
        -c|--concurrency) CONCURRENCY="$2"; shift 2 ;;
        -H|--hours)       MAX_HOURS="$2"; shift 2 ;;
        -t|--type)        INSTANCE_TYPE="$2"; shift 2 ;;
        --ami)            AMI_ID="$2"; shift 2 ;;
        --key)            KEY_NAME="$2"; shift 2 ;;
        --sg)             SECURITY_GROUP="$2"; shift 2 ;;
        --spot)           USE_SPOT=true; shift ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------- Validate ----------
if [ -z "$SPSA_API_URL" ]; then
    echo "Error: SPSA_API_URL not found in .env"
    exit 1
fi
if [ -z "$SPSA_API_KEY" ]; then
    echo "Error: SPSA_API_KEY not found in .env"
    exit 1
fi
if [ -z "$MAX_HOURS" ]; then
    echo "Error: -H/--hours is required (e.g. -H 4 for 4 hours)"
    exit 1
fi

# Auto-detect latest Amazon Linux 2023 AMI if not specified
if [ -z "$AMI_ID" ]; then
    echo "Auto-detecting latest Amazon Linux 2023 AMI..."
    AMI_ID=$(aws ec2 describe-images \
        --owners amazon \
        --filters "Name=name,Values=al2023-ami-2023*-x86_64" "Name=state,Values=available" \
        --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
        --output text)
    echo "  Using AMI: $AMI_ID"
fi

# ---------- Build user-data script ----------
USER_DATA=$(cat <<'USERDATA'
#!/bin/bash
exec > >(tee /var/log/spsa-worker.log) 2>&1
set -e

echo "=== $(date) Starting SPSA worker setup ==="

# Schedule auto-termination (shutdown triggers terminate via instance setting)
echo "=== Auto-shutdown scheduled in __MAX_HOURS__ hour(s) ==="
shutdown -h +__SHUTDOWN_MINUTES__

# Install system packages
yum install -y python3.11 python3.11-pip git gcc

# Install Rust
su - ec2-user -c 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y'

# Install Stockfish
curl -L https://github.com/official-stockfish/Stockfish/releases/download/sf_17.1/stockfish-ubuntu-x86-64-sse41-popcnt.tar -o /tmp/stockfish.tar
tar -xf /tmp/stockfish.tar -C /tmp
mv /tmp/stockfish/stockfish-ubuntu-x86-64-sse41-popcnt /usr/local/bin/stockfish
chmod +x /usr/local/bin/stockfish
rm -rf /tmp/stockfish.tar /tmp/stockfish

# Clone repos as ec2-user
su - ec2-user -c '
    cd ~
    git clone https://github.com/chris-moreton/chess-compete.git
    git clone https://github.com/chris-moreton/rusty-rival.git
    cd chess-compete
    python3.11 -m pip install -r requirements.txt
    sed -i "s|engine_path = .*|engine_path = \"/usr/local/bin/stockfish\"|" compete/spsa/config.toml
'

echo "=== $(date) Setup complete, starting worker ==="

# Run the worker as ec2-user with Rust in PATH
su - ec2-user -c '
    source ~/.cargo/env
    export SPSA_API_URL="__SPSA_API_URL__"
    export SPSA_API_KEY="__SPSA_API_KEY__"
    export RUSTFLAGS="-C target-cpu=native"
    export RUST_MIN_STACK=4097152
    cd ~/chess-compete
    python3.11 -m compete --spsa-http -c __CONCURRENCY__
'
USERDATA
)

# Substitute actual values into user-data
SHUTDOWN_MINUTES=$((MAX_HOURS * 60))
USER_DATA="${USER_DATA//__SPSA_API_URL__/$SPSA_API_URL}"
USER_DATA="${USER_DATA//__SPSA_API_KEY__/$SPSA_API_KEY}"
USER_DATA="${USER_DATA//__CONCURRENCY__/$CONCURRENCY}"
USER_DATA="${USER_DATA//__MAX_HOURS__/$MAX_HOURS}"
USER_DATA="${USER_DATA//__SHUTDOWN_MINUTES__/$SHUTDOWN_MINUTES}"

# ---------- Build AWS CLI args ----------
AWS_ARGS=(
    aws ec2 run-instances
    --count "$COUNT"
    --image-id "$AMI_ID"
    --instance-type "$INSTANCE_TYPE"
    --user-data "$USER_DATA"
    --instance-initiated-shutdown-behavior terminate
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=spsa-worker}]"
    --iam-instance-profile Name=SSMInstanceProfile
)

if [ -n "$KEY_NAME" ]; then
    AWS_ARGS+=(--key-name "$KEY_NAME")
fi

if [ -n "$SECURITY_GROUP" ]; then
    AWS_ARGS+=(--security-group-ids "$SECURITY_GROUP")
fi

if [ "$USE_SPOT" = true ]; then
    AWS_ARGS+=(--instance-market-options '{"MarketType":"spot"}')
fi

# ---------- Launch ----------
echo "Launching $COUNT x $INSTANCE_TYPE instances (concurrency=$CONCURRENCY, auto-terminate in ${MAX_HOURS}h)..."
RESULT=$("${AWS_ARGS[@]}" --query 'Instances[*].InstanceId' --output text)

echo ""
echo "Launched instances:"
for ID in $RESULT; do
    echo "  $ID"
done

echo ""
echo "Watch logs:"
for ID in $RESULT; do
    echo "  aws ssm start-session --target $ID  # then: sudo tail -f /var/log/spsa-worker.log"
done

echo ""
echo "Terminate all workers now:"
echo "  aws ec2 terminate-instances --instance-ids $RESULT"
