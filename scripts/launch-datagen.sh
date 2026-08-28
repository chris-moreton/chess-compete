#!/bin/bash
# Launch a bounded EC2 spot job using Rusty Rival's native self-play datagen.
#
# Usage:
#   ./scripts/launch-datagen.sh v1.0.56 --games 50000 --nodes 3000
#   ./scripts/launch-datagen.sh v1.0.56 --games 1000000 --nodes 3000 --type c6a.24xlarge --hours 12
#   ./scripts/launch-datagen.sh v1.0.56 --games 1000000 --run-name my-run  # resume
#
# The instance downloads a released engine, writes resumable zstd shards,
# uploads them to a run-specific S3 prefix, and terminates. This script only
# launches when explicitly invoked; preparing or testing it starts no instance.

set -euo pipefail

INSTANCE_TYPE="c6a.12xlarge"
GAMES=50000
NODES=3000
RANDOM_PLIES=8
REGION="us-west-2"
MAX_HOURS=12
MAX_SPOT_PRICE="2.00"
USE_SPOT=true
S3_BUCKET="chess-compete-builds"
DISK_GB=100
RUN_NAME=""

ENGINE_TAG=""
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --games) GAMES="$2"; shift 2 ;;
        --nodes) NODES="$2"; shift 2 ;;
        --random-plies) RANDOM_PLIES="$2"; shift 2 ;;
        --type) INSTANCE_TYPE="$2"; shift 2 ;;
        --region|-r) REGION="$2"; shift 2 ;;
        --hours) MAX_HOURS="$2"; shift 2 ;;
        --max-spot-price) MAX_SPOT_PRICE="$2"; shift 2 ;;
        --bucket) S3_BUCKET="$2"; shift 2 ;;
        --disk-gb) DISK_GB="$2"; shift 2 ;;
        --run-name) RUN_NAME="$2"; shift 2 ;;
        --on-demand) USE_SPOT=false; shift ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        -*) echo "Unknown option: $1" >&2; exit 1 ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done

if [ ${#POSITIONAL[@]} -ne 1 ]; then
    echo "Usage: $0 <engine_tag> [options]" >&2
    exit 1
fi
ENGINE_TAG="${POSITIONAL[0]}"
if [[ ! "$ENGINE_TAG" =~ ^v[0-9A-Za-z._-]+$ ]]; then
    echo "Error: invalid engine tag" >&2
    exit 1
fi
for value in "$GAMES" "$NODES" "$RANDOM_PLIES" "$MAX_HOURS" "$DISK_GB"; do
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "Error: numeric options must be unsigned integers" >&2
        exit 1
    fi
done
if (( GAMES == 0 || NODES == 0 || MAX_HOURS == 0 || DISK_GB < 20 )); then
    echo "Error: games, nodes, and hours must be positive; disk must be at least 20GB" >&2
    exit 1
fi
if [[ ! "$INSTANCE_TYPE" =~ ^[A-Za-z0-9.]+$ \
    || ! "$REGION" =~ ^[a-z0-9-]+$ \
    || ! "$S3_BUCKET" =~ ^[a-z0-9.-]+$ \
    || ! "$MAX_SPOT_PRICE" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Error: invalid instance, region, bucket, or spot-price value" >&2
    exit 1
fi
if [[ -n "$RUN_NAME" && ! "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Error: run name may contain only letters, digits, dot, underscore, and dash" >&2
    exit 1
fi
if ! command -v aws >/dev/null 2>&1; then
    echo "Error: AWS CLI not found" >&2
    exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
    echo "Error: curl not found" >&2
    exit 1
fi
PORTABLE_ASSET_URL="https://github.com/chris-moreton/rusty-rival/releases/download/${ENGINE_TAG}/rusty-rival-${ENGINE_TAG}-linux-x86_64"
if ! curl -fsIL "$PORTABLE_ASSET_URL" >/dev/null; then
    echo "Error: release asset does not exist: $PORTABLE_ASSET_URL" >&2
    echo "No instance was launched." >&2
    exit 1
fi

VCPUS=$(aws ec2 describe-instance-types --region "$REGION" --instance-types "$INSTANCE_TYPE" \
    --query 'InstanceTypes[0].VCpuInfo.DefaultVCpus' --output text)
if [[ ! "$VCPUS" =~ ^[0-9]+$ ]] || (( VCPUS == 0 )); then
    echo "Error: could not determine vCPU count for $INSTANCE_TYPE" >&2
    exit 1
fi
ARCHITECTURES=$(aws ec2 describe-instance-types --region "$REGION" --instance-types "$INSTANCE_TYPE" \
    --query 'InstanceTypes[0].ProcessorInfo.SupportedArchitectures' --output text)
if [[ " $ARCHITECTURES " != *" x86_64 "* ]]; then
    echo "Error: $INSTANCE_TYPE does not support the required x86_64 architecture" >&2
    exit 1
fi
THREADS=$VCPUS
SHUTDOWN_MINUTES=$((MAX_HOURS * 60))
TIMESTAMP=$(date -u +%Y%m%d-%H%M%S)
RUN_NAME="${RUN_NAME:-${ENGINE_TAG}_n${NODES}_${GAMES}games_${TIMESTAMP}}"
# The generated or explicit run name is the stable identity. The manifest below
# rejects engine/node/opening drift when that identity is resumed.
S3_PATH="s3://${S3_BUCKET}/nnue-data-selfplay/runs/${RUN_NAME}/"
LOG_NAME="datagen-${TIMESTAMP}.log"

echo "Rusty Rival native self-play datagen"
echo "  Engine:       $ENGINE_TAG"
echo "  Instance:     $INSTANCE_TYPE ($VCPUS vCPUs)"
echo "  Games:        $GAMES"
echo "  Nodes/move:   $NODES"
echo "  Random plies: $RANDOM_PLIES"
echo "  Threads:      $THREADS"
echo "  Run name:     $RUN_NAME (reuse with --run-name to resume)"
echo "  Hard limit:   ${MAX_HOURS}h (instance terminates)"
echo "  Disk:         ${DISK_GB}GB"
echo "  S3 output:    $S3_PATH"
echo

USER_DATA=$(cat <<'USERDATA'
#!/bin/bash
set -euo pipefail
exec > >(tee /var/log/datagen.log) 2>&1

echo "=== $(date) Starting native Rusty Rival datagen ==="
shutdown -h +__SHUTDOWN_MINUTES__
persist_and_shutdown() {
    status=$?
    trap - EXIT
    set +e
    if command -v aws >/dev/null 2>&1; then
        aws s3 cp /var/log/datagen.log "__S3_PATH__logs/__LOG_NAME__" --only-show-errors
    fi
    shutdown -h now 2>/dev/null || true
    exit "$status"
}
trap persist_and_shutdown EXIT
(
    sleep $((__SHUTDOWN_MINUTES__ * 60 + 300))
    INSTANCE_ID=$(curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/instance-id || true)
    INSTANCE_REGION=$(curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/placement/region || true)
    if [ -n "$INSTANCE_ID" ] && [ -n "$INSTANCE_REGION" ]; then
        aws ec2 terminate-instances --region "$INSTANCE_REGION" --instance-ids "$INSTANCE_ID" 2>&1 || true
    fi
    shutdown -h now 2>/dev/null || true
) &
disown

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y awscli curl

REPO="chris-moreton/rusty-rival"
ASSET_SUFFIX="linux-x86_64"
if grep -q -m1 -w avx2 /proc/cpuinfo; then
    ASSET_SUFFIX="linux-x86_64-avx2"
fi
ASSET="rusty-rival-__ENGINE_TAG__-${ASSET_SUFFIX}"
URL="https://github.com/${REPO}/releases/download/__ENGINE_TAG__/${ASSET}"
echo "Downloading $URL"
if ! curl -fL "$URL" -o /usr/local/bin/rusty-rival; then
    ASSET="rusty-rival-__ENGINE_TAG__-linux-x86_64"
    curl -fL "https://github.com/${REPO}/releases/download/__ENGINE_TAG__/${ASSET}" \
        -o /usr/local/bin/rusty-rival
fi
chmod +x /usr/local/bin/rusty-rival
printf "uci\nquit\n" | /usr/local/bin/rusty-rival

install -d -o ubuntu -g ubuntu /var/lib/rival-datagen
su - ubuntu -c '
    set -euo pipefail
    OUTPUT_BASE=/var/lib/rival-datagen/__RUN_NAME__
    MANIFEST=/var/lib/rival-datagen/selfplay-format-v1.env
    EXISTING_MANIFEST=/tmp/selfplay-format-v1.env
    printf "%s\n" \
        "format=rusty-rival-selfplay-v1" \
        "score_pov=white" \
        "result_pov=white" \
        "engine_tag=__ENGINE_TAG__" \
        "nodes=__NODES__" \
        "random_plies=__RANDOM_PLIES__" > "$MANIFEST"
    if aws s3 cp "__S3_PATH__selfplay-format-v1.env" "$EXISTING_MANIFEST" --only-show-errors 2>/dev/null; then
        cmp -s "$MANIFEST" "$EXISTING_MANIFEST" || {
            echo "ERROR: run manifest disagrees with engine, node budget, random plies, or label format" >&2
            exit 1
        }
    else
        aws s3 cp "$MANIFEST" "__S3_PATH__selfplay-format-v1.env" --only-show-errors
    fi
    echo "Restoring any durable shards from __S3_PATH__"
    aws s3 sync "__S3_PATH__" /var/lib/rival-datagen/ --exclude "*" --include "*.zst"
    (
        while sleep 300; do
            aws s3 sync /var/lib/rival-datagen/ "__S3_PATH__" --exclude "*" --include "*.zst" \
                || echo "WARNING: periodic shard upload failed; retrying in five minutes" >&2
        done
    ) &
    SYNC_PID=$!
    trap "kill $SYNC_PID 2>/dev/null || true" EXIT
    printf "datagen __GAMES__ __NODES__ %s __RANDOM_PLIES__ __THREADS__\nquit\n" "$OUTPUT_BASE" \
        | /usr/local/bin/rusty-rival
    compgen -G "$OUTPUT_BASE.*.zst" >/dev/null \
        || { echo "ERROR: datagen produced no durable shards" >&2; exit 1; }
    aws s3 sync /var/lib/rival-datagen/ "__S3_PATH__" --exclude "*" --include "*.zst"
'

echo "=== $(date) Datagen upload complete; terminating ==="
shutdown -h now
USERDATA
)

USER_DATA="${USER_DATA//__ENGINE_TAG__/$ENGINE_TAG}"
USER_DATA="${USER_DATA//__RUN_NAME__/$RUN_NAME}"
USER_DATA="${USER_DATA//__GAMES__/$GAMES}"
USER_DATA="${USER_DATA//__NODES__/$NODES}"
USER_DATA="${USER_DATA//__RANDOM_PLIES__/$RANDOM_PLIES}"
USER_DATA="${USER_DATA//__THREADS__/$THREADS}"
USER_DATA="${USER_DATA//__S3_PATH__/$S3_PATH}"
USER_DATA="${USER_DATA//__SHUTDOWN_MINUTES__/$SHUTDOWN_MINUTES}"
USER_DATA="${USER_DATA//__LOG_NAME__/$LOG_NAME}"

REGION_ARGS=(--region "$REGION")
AMI_ID=$(aws ec2 describe-images "${REGION_ARGS[@]}" \
    --owners 099720109477 \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' --output text)

SPOT_ARGS=()
if [ "$USE_SPOT" = true ]; then
    SPOT_ARGS=(--instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"'"$MAX_SPOT_PRICE"'","SpotInstanceType":"one-time"}}')
fi

# AWS CLI performs the required base64 encoding. Passing encoded text here
# makes cloud-init receive inert data rather than this script.
INSTANCE_ID=$(aws ec2 run-instances "${REGION_ARGS[@]}" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    ${SPOT_ARGS[@]+"${SPOT_ARGS[@]}"} \
    --iam-instance-profile Name=SSMInstanceProfile \
    --instance-initiated-shutdown-behavior terminate \
    --user-data "$USER_DATA" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":'"$DISK_GB"',"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=nnue-datagen-${RUN_NAME}}]" \
    --query 'Instances[0].InstanceId' --output text)

echo "Instance launched: $INSTANCE_ID"
echo "Logs: aws ssm start-session --target $INSTANCE_ID --region $REGION"
echo "Terminate early: aws ec2 terminate-instances --region $REGION --instance-ids $INSTANCE_ID"
