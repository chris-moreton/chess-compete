#!/bin/bash
# Launch SPSA worker EC2 instances
#
# Usage:
#   ./scripts/launch-ec2-workers.sh -H 4              # 1 instance, auto-terminate after 4 hours
#   ./scripts/launch-ec2-workers.sh -n 3 -H 6          # 3 instances, 6 hours
#   ./scripts/launch-ec2-workers.sh -n 3 -H 4 --spot   # 3 spot instances (diversified fleet)
#   ./scripts/launch-ec2-workers.sh -n 3 -c 8 -H 2     # 3 instances, 8 concurrent games, 2 hours
#   ./scripts/launch-ec2-workers.sh -n 4 -H 8 --spot -r us-east-1  # spot in US East
#
# Spot mode uses a diversified fleet across multiple instance types and AZs
# to reduce the risk of all workers being terminated at once.
#
# Workers self-terminate after 10 minutes of no work (--idle-timeout to change).
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
REGION=""
CONCURRENCY="${CONCURRENCY:-16}"
IDLE_TIMEOUT=10  # Minutes of no work before self-terminating (0 = disabled)
COUNT=1
MAX_HOURS=""
USE_SPOT=false

# Spot fleet: diversified instance types (all 16-vCPU compute-optimized or general-purpose)
# Spot fleet: c7a only (3.7GHz AMD EPYC Genoa)
SPOT_INSTANCE_TYPES=("c7a.4xlarge")

# ---------- Parse arguments ----------
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--count)       COUNT="$2"; shift 2 ;;
        -c|--concurrency) CONCURRENCY="$2"; shift 2 ;;
        -H|--hours)       MAX_HOURS="$2"; shift 2 ;;
        --idle-timeout)   IDLE_TIMEOUT="$2"; shift 2 ;;
        -t|--type)        INSTANCE_TYPE="$2"; shift 2 ;;
        --ami)            AMI_ID="$2"; shift 2 ;;
        --key)            KEY_NAME="$2"; shift 2 ;;
        --sg)             SECURITY_GROUP="$2"; shift 2 ;;
        --region|-r)      REGION="$2"; shift 2 ;;
        --spot)           USE_SPOT=true; shift ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------- Validate ----------
if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI not found"
    exit 1
fi
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

# Build region args for all AWS CLI calls
REGION_ARGS=()
if [ -n "$REGION" ]; then
    REGION_ARGS=(--region "$REGION")
    echo "Using region: $REGION"
fi

# Auto-detect latest Amazon Linux 2023 AMI if not specified
if [ -z "$AMI_ID" ]; then
    echo "Auto-detecting latest Amazon Linux 2023 AMI..."
    AMI_ID=$(aws ec2 describe-images "${REGION_ARGS[@]}" \
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
    python3.11 -m compete --spsa-http -c __CONCURRENCY__ --idle-timeout __IDLE_TIMEOUT__
'
USERDATA
)

# Substitute actual values into user-data
SHUTDOWN_MINUTES=$((MAX_HOURS * 60))
USER_DATA="${USER_DATA//__SPSA_API_URL__/$SPSA_API_URL}"
USER_DATA="${USER_DATA//__SPSA_API_KEY__/$SPSA_API_KEY}"
USER_DATA="${USER_DATA//__CONCURRENCY__/$CONCURRENCY}"
USER_DATA="${USER_DATA//__IDLE_TIMEOUT__/$IDLE_TIMEOUT}"
USER_DATA="${USER_DATA//__MAX_HOURS__/$MAX_HOURS}"
USER_DATA="${USER_DATA//__SHUTDOWN_MINUTES__/$SHUTDOWN_MINUTES}"

# Encode user-data as base64 for fleet API
USER_DATA_B64=$(echo "$USER_DATA" | base64)

if [ "$USE_SPOT" = true ]; then
    # ---------- Spot Fleet (diversified across instance types and AZs) ----------
    echo "Checking instance type availability in region..."

    # Query which instance types are actually offered in each AZ
    AVAILABLE=$(aws ec2 describe-instance-type-offerings "${REGION_ARGS[@]}" \
        --location-type availability-zone \
        --filters "Name=instance-type,Values=$(IFS=,; echo "${SPOT_INSTANCE_TYPES[*]}")" \
        --query 'InstanceTypeOfferings[*].[InstanceType,Location]' \
        --output text)

    # Get default subnet for each AZ (single API call, stored as "subnet\taz" lines)
    AZ_SUBNET_MAP=$(aws ec2 describe-subnets "${REGION_ARGS[@]}" \
        --filters "Name=default-for-az,Values=true" \
        --query 'Subnets[*].[SubnetId,AvailabilityZone]' \
        --output text)

    # Build overrides only for valid type/AZ combinations
    OVERRIDES=""
    VALID_TYPES=()
    SKIPPED_TYPES=()
    while IFS=$'\t' read -r ITYPE AZ; do
        # Look up subnet for this AZ
        SUBNET=$(echo "$AZ_SUBNET_MAP" | awk -v az="$AZ" '$2 == az {print $1; exit}')
        if [ -z "$SUBNET" ]; then
            continue  # No default subnet in this AZ
        fi
        if [ -n "$OVERRIDES" ]; then
            OVERRIDES="${OVERRIDES},"
        fi
        OVERRIDES="${OVERRIDES}{\"InstanceType\":\"${ITYPE}\",\"SubnetId\":\"${SUBNET}\"}"
        # Track unique valid types
        if [[ ! " ${VALID_TYPES[*]} " =~ " ${ITYPE} " ]]; then
            VALID_TYPES+=("$ITYPE")
        fi
    done <<< "$AVAILABLE"

    # Report which types were excluded
    for ITYPE in "${SPOT_INSTANCE_TYPES[@]}"; do
        if [[ ! " ${VALID_TYPES[*]} " =~ " ${ITYPE} " ]]; then
            SKIPPED_TYPES+=("$ITYPE")
        fi
    done

    if [ ${#SKIPPED_TYPES[@]} -gt 0 ]; then
        echo "  Excluded (not offered in region): ${SKIPPED_TYPES[*]}"
    fi

    if [ -z "$OVERRIDES" ]; then
        echo "Error: None of the configured instance types are available in this region."
        exit 1
    fi

    echo "Launching diversified spot fleet: $COUNT instances across ${#VALID_TYPES[@]} types..."
    echo "  Instance types: ${VALID_TYPES[*]}"

    # Build optional fields for launch template
    OPTIONAL_LT_FIELDS=""
    if [ -n "$KEY_NAME" ]; then
        OPTIONAL_LT_FIELDS="${OPTIONAL_LT_FIELDS},\"KeyName\":\"${KEY_NAME}\""
    fi
    if [ -n "$SECURITY_GROUP" ]; then
        OPTIONAL_LT_FIELDS="${OPTIONAL_LT_FIELDS},\"SecurityGroupIds\":[\"${SECURITY_GROUP}\"]"
    fi

    # Create a temporary launch template (required for create-fleet)
    LT_NAME="spsa-worker-$(date +%s)"
    LT_DATA="{\"ImageId\":\"${AMI_ID}\",\"UserData\":\"${USER_DATA_B64}\",\"IamInstanceProfile\":{\"Name\":\"SSMInstanceProfile\"},\"InstanceInitiatedShutdownBehavior\":\"terminate\"${OPTIONAL_LT_FIELDS}}"

    LT_ID=$(aws ec2 create-launch-template "${REGION_ARGS[@]}" \
        --launch-template-name "$LT_NAME" \
        --launch-template-data "$LT_DATA" \
        --query 'LaunchTemplate.LaunchTemplateId' \
        --output text)
    echo "  Created launch template: $LT_ID ($LT_NAME)"

    # Create fleet with capacity-optimized-prioritized allocation
    FLEET_RESULT=$(aws ec2 create-fleet "${REGION_ARGS[@]}" \
        --type instant \
        --target-capacity-specification "TotalTargetCapacity=${COUNT},DefaultTargetCapacityType=spot" \
        --spot-options "AllocationStrategy=capacity-optimized-prioritized" \
        --launch-template-configs "[{\"LaunchTemplateSpecification\":{\"LaunchTemplateId\":\"${LT_ID}\",\"Version\":\"\$Latest\"},\"Overrides\":[${OVERRIDES}]}]" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=spsa-worker}]" \
        --output json)

    # Extract instance IDs
    INSTANCE_IDS=$(echo "$FLEET_RESULT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
ids = [iid for inst in data.get('Instances', []) for iid in inst.get('InstanceIds', [])]
print(' '.join(ids))
")

    # Check for errors
    ERRORS=$(echo "$FLEET_RESULT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
errors = data.get('Errors', [])
if errors:
    for e in errors:
        print(f\"  {e.get('ErrorCode', '?')}: {e.get('ErrorMessage', '?')}\")
" 2>/dev/null || true)

    # Clean up launch template
    aws ec2 delete-launch-template "${REGION_ARGS[@]}" --launch-template-id "$LT_ID" > /dev/null 2>&1

    LAUNCHED=$(echo "$INSTANCE_IDS" | wc -w | tr -d ' ')
    echo ""
    echo "Launched $LAUNCHED/$COUNT spot instances (capacity-optimized across types/AZs):"
    for ID in $INSTANCE_IDS; do
        echo "  $ID"
    done

    if [ -n "$ERRORS" ]; then
        echo ""
        echo "Fleet errors (partial capacity):"
        echo "$ERRORS"
    fi

    if [ "$LAUNCHED" -eq 0 ]; then
        echo "Error: No instances launched. Check spot capacity in your region."
        exit 1
    fi

    REGION_HINT=""
    if [ -n "$REGION" ]; then
        REGION_HINT=" --region $REGION"
    fi

    echo ""
    echo "Watch logs:"
    for ID in $INSTANCE_IDS; do
        echo "  aws ssm start-session${REGION_HINT} --target $ID  # then: sudo tail -f /var/log/spsa-worker.log"
    done

    echo ""
    echo "Terminate all workers now:"
    echo "  aws ec2 terminate-instances${REGION_HINT} --instance-ids $INSTANCE_IDS"

else
    # ---------- On-demand (original approach) ----------
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

    if [ -n "$REGION" ]; then
        AWS_ARGS+=(--region "$REGION")
    fi

    echo "Launching $COUNT x $INSTANCE_TYPE on-demand instances (concurrency=$CONCURRENCY, auto-terminate in ${MAX_HOURS}h)..."
    RESULT=$("${AWS_ARGS[@]}" --query 'Instances[*].InstanceId' --output text)

    REGION_HINT=""
    if [ -n "$REGION" ]; then
        REGION_HINT=" --region $REGION"
    fi

    echo ""
    echo "Launched instances:"
    for ID in $RESULT; do
        echo "  $ID"
    done

    echo ""
    echo "Watch logs:"
    for ID in $RESULT; do
        echo "  aws ssm start-session${REGION_HINT} --target $ID  # then: sudo tail -f /var/log/spsa-worker.log"
    done

    echo ""
    echo "Terminate all workers now:"
    echo "  aws ec2 terminate-instances${REGION_HINT} --instance-ids $RESULT"
fi
