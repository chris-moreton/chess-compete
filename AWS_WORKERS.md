# AWS SPSA Workers

Launch EC2 instances that automatically set up and run SPSA workers, then self-terminate after a set number of hours.

## Prerequisites

### 1. Install AWS CLI

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf awscliv2.zip aws/
```

On macOS:
```bash
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /
rm AWSCLIV2.pkg
```

### 2. Create an IAM access key

1. Log into the [AWS Console](https://console.aws.amazon.com/)
2. Go to **IAM** > **Users** > your username
3. **Security credentials** tab > **Create access key**
4. Select **Command Line Interface (CLI)**
5. Copy the Access Key ID and Secret Access Key

### 3. Configure AWS CLI

```bash
aws configure
```

Enter your access key, secret key, and default region (e.g. `eu-west-2` for London).

Verify it works:
```bash
aws sts get-caller-identity
```

### 4. Create the SSM Instance Profile (one-time setup)

This allows you to view logs on running instances via SSM Session Manager.

Create a trust policy file:
```bash
cat > /tmp/trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "ec2.amazonaws.com" },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
```

Create the role and instance profile:
```bash
aws iam create-role --role-name SSMInstanceRole --assume-role-policy-document file:///tmp/trust-policy.json
aws iam attach-role-policy --role-name SSMInstanceRole --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam create-instance-profile --instance-profile-name SSMInstanceProfile
aws iam add-role-to-instance-profile --instance-profile-name SSMInstanceProfile --role-name SSMInstanceRole
```

### 5. Create S3 Build Cache Bucket (one-time setup)

This allows workers to share compiled engine binaries instead of each building independently.

```bash
# Create the bucket (private — workers use IAM for both reads and writes)
aws s3 mb s3://chess-compete-builds --region eu-west-2

# Add read/write permission to SSMInstanceRole
aws iam put-role-policy --role-name SSMInstanceRole --policy-name S3BuildCache \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
      "Resource": ["arn:aws:s3:::chess-compete-builds", "arn:aws:s3:::chess-compete-builds/*"]
    }]
  }'

# Auto-expire old builds after 7 days
aws s3api put-bucket-lifecycle-configuration --bucket chess-compete-builds \
  --lifecycle-configuration '{
    "Rules": [{
      "ID": "expire-old-builds",
      "Status": "Enabled",
      "Filter": {"Prefix": "spsa/"},
      "Expiration": {"Days": 7}
    }]
  }'
```

Enable the cache in `compete/spsa/config.toml`:
```toml
[build]
s3_build_cache = "chess-compete-builds"
```

When enabled, the first worker to build an engine uploads it to S3. Subsequent workers download the binary (~3s) instead of compiling (~30-60s).

### 6. Install SSM Session Manager plugin

This lets you connect to instances to view logs.

Ubuntu/Debian/WSL:
```bash
curl "https://s3.amazonaws.com/session-manager-downloads/plugin/latest/ubuntu_64bit/session-manager-plugin.deb" -o "session-manager-plugin.deb"
sudo dpkg -i session-manager-plugin.deb
rm session-manager-plugin.deb
```

macOS:
```bash
curl "https://s3.amazonaws.com/session-manager-downloads/plugin/latest/mac/sessionmanager-bundle.zip" -o "sessionmanager-bundle.zip"
unzip sessionmanager-bundle.zip
sudo ./sessionmanager-bundle/install -i /usr/local/sessionmanagerplugin -b /usr/local/bin/session-manager-plugin
rm -rf sessionmanager-bundle.zip sessionmanager-bundle/
```

## Launching Workers

The launch script reads `SPSA_API_URL` and `SPSA_API_KEY` from the `.env` file.

```bash
# Launch 1 instance, auto-terminate after 4 hours
./scripts/launch-ec2-workers.sh -n 1 -H 4

# Launch 3 instances, auto-terminate after 6 hours
./scripts/launch-ec2-workers.sh -n 3 -H 6

# Launch 3 spot instances (60-70% cheaper, may be reclaimed with 2 min notice)
./scripts/launch-ec2-workers.sh -n 3 -H 4 --spot

# Override concurrency (default: 16) or instance type (default: c7a.4xlarge)
./scripts/launch-ec2-workers.sh -n 2 -H 4 -c 8 -t c6a.xlarge
```

The `-H` (hours) flag is required to prevent accidentally leaving instances running.

Each instance will:
1. Install dependencies (Rust, Stockfish, Python packages)
2. Clone the repos and build the engine
3. Start the SPSA worker
4. Auto-terminate after the specified hours

Setup takes a few minutes before games start.

## Watching Logs

Connect to an instance:
```bash
aws ssm start-session --target <instance-id>
```

Then:
```bash
sudo tail -f /var/log/spsa-worker.log
```

Type `exit` to disconnect (the instance keeps running).

## Terminating Workers Early

Terminate specific instances:
```bash
aws ec2 terminate-instances --instance-ids i-0abc123 i-0def456
```

Find all running workers:
```bash
aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=spsa-worker" "Name=instance-state-name,Values=running" \
    --query 'Reservations[*].Instances[*].[InstanceId,LaunchTime]' \
    --output table
```

Terminate all workers:
```bash
aws ec2 terminate-instances --instance-ids $(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=spsa-worker" "Name=instance-state-name,Values=running" \
    --query 'Reservations[*].Instances[*].InstanceId' --output text)
```

## Costs

- **c7a.4xlarge** (16 vCPUs, 32GB RAM): ~$0.60/hr on-demand, ~$0.20/hr spot
- Instances auto-terminate after `-H` hours
- Spot instances (`--spot`) are significantly cheaper but can be reclaimed with 2 minutes notice — fine for SPSA workers since losing a few in-flight games doesn't matter

## Worker Identity

Each EC2 instance reports results using its hostname (e.g. `ip-172-31-42-15`), so multiple workers are automatically distinguished in the API.
