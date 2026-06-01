#!/usr/bin/env bash
# Attach memory profiler to an already-running process by PID.
# Usage:
#   ./aws/attach-mem.sh <pid>
#   ./aws/attach-mem.sh         # auto-detects 'ga' process

set -euo pipefail

if [[ $# -ge 1 ]]; then
  PID="$1"
else
  PID=$(pgrep -x ga | head -1 || true)
  if [[ -z "$PID" ]]; then
    echo "No 'ga' process found. Pass PID explicitly."
    exit 1
  fi
  echo "==> Auto-detected GA pid: $PID"
fi

if ! kill -0 "$PID" 2>/dev/null; then
  echo "PID $PID not running"
  exit 1
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
CSV="mem-profile-attached-${PID}-${STAMP}.csv"
SUMMARY="mem-profile-attached-${PID}-${STAMP}.summary"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-2}"

echo "elapsed_sec,rss_mb,vsz_mb" > "$CSV"
echo "==> Sampling PID $PID every ${SAMPLE_INTERVAL}s -> $CSV"
echo "==> Stop with Ctrl-C (CSV stays valid) or wait until process exits"

START_TIME=$(date +%s)

while kill -0 "$PID" 2>/dev/null; do
  NOW=$(date +%s)
  ELAPSED=$((NOW - START_TIME))
  STATS=$(ps -o rss=,vsz= -p "$PID" 2>/dev/null || echo "0 0")
  RSS_KB=$(awk '{print $1}' <<<"$STATS")
  VSZ_KB=$(awk '{print $2}' <<<"$STATS")
  RSS_MB=$(awk -v k="$RSS_KB" 'BEGIN{printf "%.1f", k/1024}')
  VSZ_MB=$(awk -v k="$VSZ_KB" 'BEGIN{printf "%.1f", k/1024}')
  echo "${ELAPSED},${RSS_MB},${VSZ_MB}" | tee -a "$CSV"
  sleep "$SAMPLE_INTERVAL"
done

python3 - "$CSV" "$SUMMARY" <<'PY'
import csv, sys, statistics
csv_path, summary_path = sys.argv[1], sys.argv[2]
rss, vsz = [], []
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rss.append(float(row["rss_mb"]))
        vsz.append(float(row["vsz_mb"]))
if not rss:
    print("No samples")
    sys.exit(0)
with open(summary_path, "w") as f:
    f.write(f"samples:        {len(rss)}\n")
    f.write(f"peak_rss_mb:    {max(rss):.1f}\n")
    f.write(f"mean_rss_mb:    {statistics.mean(rss):.1f}\n")
    f.write(f"final_rss_mb:   {rss[-1]:.1f}\n")
    f.write(f"peak_vsz_mb:    {max(vsz):.1f}\n")
    f.write(f"mean_vsz_mb:    {statistics.mean(vsz):.1f}\n")
print(open(summary_path).read())
PY

echo "==> CSV:     $CSV"
echo "==> Summary: $SUMMARY"
