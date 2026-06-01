#!/usr/bin/env bash
# Pull results from S3 once the run finished.
# Usage: ./aws/fetch-results.sh <run-id>

set -euo pipefail
RUN_ID="${1:?Pass the RUN_ID printed by launch.sh}"
REGION="${REGION:-eu-central-1}"
S3_BUCKET="${S3_BUCKET:?Set S3_BUCKET env var}"
PROBLEM_SIZE="${PROBLEM_SIZE:-1000x1000}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/rezultati-${PROBLEM_SIZE}"
mkdir -p "$DEST"

echo "==> Pulling s3://$S3_BUCKET/$RUN_ID/results/ -> $DEST"
aws s3 cp "s3://$S3_BUCKET/$RUN_ID/results/" "$DEST/" \
  --recursive --region "$REGION"
echo "==> Done"
