#!/usr/bin/env bash
# Runs on EC2 boot. Builds and runs GA, uploads results, terminates instance.
set -euxo pipefail

# Placeholders replaced by launch.sh
S3_BUCKET="__S3_BUCKET__"
RUN_ID="__RUN_ID__"
REGION="__REGION__"
PROBLEM_SIZE="__PROBLEM_SIZE__"

LOG=/var/log/ga-run.log
exec > >(tee -a "$LOG") 2>&1

echo "==> Provisioning $(date)"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y build-essential g++ awscli zram-tools

# zram swap (mimic macOS-style memory compression for sparse chromosomes)
echo -e "ALGO=zstd\nPERCENT=200" > /etc/default/zramswap
systemctl restart zramswap || true

# Disable transparent hugepage stalls
echo never > /sys/kernel/mm/transparent_hugepage/enabled || true

WORK=/opt/ga
mkdir -p "$WORK"
cd "$WORK"

echo "==> Fetching bundle"
aws s3 cp "s3://$S3_BUCKET/$RUN_ID/bundle.tar.gz" bundle.tar.gz --region "$REGION"
tar -xzf bundle.tar.gz
mkdir -p "rezultati-${PROBLEM_SIZE}"

echo "==> Building"
g++ -O3 -march=native -flto -DNDEBUG -std=c++17 genetic-algorithm.cpp -o ga

echo "==> Running"
/usr/bin/time -v ./ga 2>&1 | tee run.log

echo "==> Uploading results"
aws s3 cp "rezultati-${PROBLEM_SIZE}/" "s3://$S3_BUCKET/$RUN_ID/results/" \
  --recursive --region "$REGION"
aws s3 cp run.log    "s3://$S3_BUCKET/$RUN_ID/results/run.log"    --region "$REGION"
aws s3 cp /var/log/ga-run.log "s3://$S3_BUCKET/$RUN_ID/results/provision.log" --region "$REGION" || true

echo "==> Done $(date). Shutting down."
shutdown -h now
