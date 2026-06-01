#!/usr/bin/env bash
# One-click AWS launcher for genetic-algorithm.cpp
#
# Prerequisites (one-time):
#   1. AWS CLI installed + `aws configure` done
#   2. EC2 key pair exists (set KEY_NAME below)
#   3. Security group allowing SSH from your IP (set SECURITY_GROUP_ID)
#   4. S3 bucket for code/results (set S3_BUCKET — script creates if missing)
#
# Usage:
#   ./aws/launch.sh                       # default instance
#   INSTANCE_TYPE=r7iz.8xlarge ./aws/launch.sh
#   POPULATION_SIZE=80000 ELITE_COUNT=800 ./aws/launch.sh

set -euo pipefail

# ── Config (override via env) ────────────────────────────────────────────────
REGION="${REGION:-eu-central-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-r7iz.2xlarge}"
KEY_NAME="${KEY_NAME:-ga-key}"
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:?Set SECURITY_GROUP_ID env var (sg-xxxxx)}"
S3_BUCKET="${S3_BUCKET:?Set S3_BUCKET env var}"
PROBLEM_SIZE="${PROBLEM_SIZE:-1000x1000}"

# GA hyperparameters (patched into source before build)
POPULATION_SIZE="${POPULATION_SIZE:-20000}"
GENERATIONS="${GENERATIONS:-100}"
ELITE_COUNT="${ELITE_COUNT:-200}"
NUM_RUNS="${NUM_RUNS:-3}"

# Ubuntu 22.04 LTS AMIs per region (amd64). Add yours if missing.
declare -A AMI_BY_REGION=(
  ["us-east-1"]="ami-0c7217cdde317cfec"
  ["us-west-2"]="ami-008fe2fc65df48dac"
  ["eu-central-1"]="ami-04e601abe3e1a910f"
  ["eu-west-1"]="ami-0905a3c97561e0b69"
)
AMI_ID="${AMI_BY_REGION[$REGION]:?No AMI mapped for region $REGION — add it to AMI_BY_REGION}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_ID="ga-${POPULATION_SIZE}-${GENERATIONS}-${ELITE_COUNT}-${NUM_RUNS}runs-${STAMP}"

echo "==> Run ID: $RUN_ID"
echo "==> Region: $REGION  Instance: $INSTANCE_TYPE  Bucket: $S3_BUCKET"

# ── 1. Ensure bucket exists ──────────────────────────────────────────────────
if ! aws s3api head-bucket --bucket "$S3_BUCKET" --region "$REGION" 2>/dev/null; then
  echo "==> Creating bucket s3://$S3_BUCKET"
  if [[ "$REGION" == "us-east-1" ]]; then
    aws s3api create-bucket --bucket "$S3_BUCKET" --region "$REGION"
  else
    aws s3api create-bucket --bucket "$S3_BUCKET" --region "$REGION" \
      --create-bucket-configuration LocationConstraint="$REGION"
  fi
fi

# ── 2. Patch hyperparameters into source ─────────────────────────────────────
SRC_DIR="$(mktemp -d)"
trap 'rm -rf "$SRC_DIR"' EXIT
cp "$REPO_ROOT/genetic-algorithm.cpp" "$SRC_DIR/genetic-algorithm.cpp"

python3 - "$SRC_DIR/genetic-algorithm.cpp" <<PY
import re, sys
path = sys.argv[1]
with open(path) as f: src = f.read()
src = re.sub(r'(PROBLEM_SIZE\s*=\s*)"[^"]+"',         r'\1"${PROBLEM_SIZE}"',     src)
src = re.sub(r'(POPULATION_SIZE\s*=\s*)\d+',          r'\1${POPULATION_SIZE}',    src)
src = re.sub(r'(GENERATIONS\s*=\s*)\d+',              r'\1${GENERATIONS}',        src)
src = re.sub(r'(ELITE_COUNT\s*=\s*)\d+',              r'\1${ELITE_COUNT}',        src)
src = re.sub(r'(NUM_RUNS\s*=\s*)\d+',                 r'\1${NUM_RUNS}',           src)
with open(path, 'w') as f: f.write(src)
PY

# ── 3. Bundle and upload to S3 ───────────────────────────────────────────────
BUNDLE="$SRC_DIR/bundle.tar.gz"
tar -czf "$BUNDLE" \
  -C "$SRC_DIR" genetic-algorithm.cpp \
  -C "$REPO_ROOT" "podaci-${PROBLEM_SIZE}"

aws s3 cp "$BUNDLE" "s3://$S3_BUCKET/$RUN_ID/bundle.tar.gz" --region "$REGION"
echo "==> Bundle uploaded"

# ── 4. Render user-data script ───────────────────────────────────────────────
USER_DATA_FILE="$SRC_DIR/user-data.sh"
sed \
  -e "s|__S3_BUCKET__|$S3_BUCKET|g" \
  -e "s|__RUN_ID__|$RUN_ID|g" \
  -e "s|__REGION__|$REGION|g" \
  -e "s|__PROBLEM_SIZE__|$PROBLEM_SIZE|g" \
  "$REPO_ROOT/aws/user-data.sh" > "$USER_DATA_FILE"

# ── 5. IAM instance profile (for S3 upload from instance) ────────────────────
ROLE_NAME="ga-runner-role"
PROFILE_NAME="ga-runner-profile"
if ! aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null 2>&1; then
  echo "==> Creating IAM role/profile for S3 access"
  aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document '{
    "Version":"2012-10-17","Statement":[{"Effect":"Allow",
    "Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
  aws iam create-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE_NAME" --role-name "$ROLE_NAME"
  echo "==> Waiting 10s for IAM propagation"
  sleep 10
fi

# ── 6. Launch instance ───────────────────────────────────────────────────────
echo "==> Launching $INSTANCE_TYPE"
INSTANCE_ID=$(aws ec2 run-instances \
  --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SECURITY_GROUP_ID" \
  --iam-instance-profile "Name=$PROFILE_NAME" \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=50,VolumeType=gp3}' \
  --user-data "file://$USER_DATA_FILE" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$RUN_ID}]" \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "==> Instance: $INSTANCE_ID"
echo "==> Watch logs: aws ec2 get-console-output --instance-id $INSTANCE_ID --region $REGION --output text"
echo "==> SSH (after boot): ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@\$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --region $REGION --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"
echo ""
echo "==> Instance auto-terminates after run. Results land in:"
echo "    s3://$S3_BUCKET/$RUN_ID/results/"
echo ""
echo "==> Fetch when done:"
echo "    ./aws/fetch-results.sh $RUN_ID"
