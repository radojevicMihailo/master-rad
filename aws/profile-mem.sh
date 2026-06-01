#!/usr/bin/env bash
# Memory profiler — runs GA + samples RSS each second, writes CSV.
#
# Usage:
#   ./aws/profile-mem.sh ./ga [args...]
#
# Output:
#   mem-profile-<timestamp>.csv  with columns: elapsed_sec,rss_mb,vsz_mb
#   mem-profile-<timestamp>.summary  with peak RSS, mean, samples

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <ga-binary> [args...]"
  exit 1
fi

GA_CMD=("$@")
STAMP="$(date +%Y%m%d-%H%M%S)"
CSV="mem-profile-${STAMP}.csv"
SUMMARY="mem-profile-${STAMP}.summary"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-1}"  # seconds

echo "elapsed_sec,rss_mb,vsz_mb" > "$CSV"

echo "==> Starting GA in background"
"${GA_CMD[@]}" &
GA_PID=$!
START_TIME=$(date +%s)

echo "==> PID $GA_PID, sampling every ${SAMPLE_INTERVAL}s -> $CSV"

# Detect ps flags per OS
if [[ "$(uname)" == "Darwin" ]]; then
  PS_FLAGS="-o rss=,vsz="    # kbytes
else
  PS_FLAGS="-o rss=,vsz="    # kbytes on Linux too
fi

while kill -0 "$GA_PID" 2>/dev/null; do
  NOW=$(date +%s)
  ELAPSED=$((NOW - START_TIME))
  STATS=$(ps $PS_FLAGS -p "$GA_PID" 2>/dev/null || echo "0 0")
  RSS_KB=$(awk '{print $1}' <<<"$STATS")
  VSZ_KB=$(awk '{print $2}' <<<"$STATS")
  RSS_MB=$(awk -v k="$RSS_KB" 'BEGIN{printf "%.1f", k/1024}')
  VSZ_MB=$(awk -v k="$VSZ_KB" 'BEGIN{printf "%.1f", k/1024}')
  echo "${ELAPSED},${RSS_MB},${VSZ_MB}" >> "$CSV"
  sleep "$SAMPLE_INTERVAL"
done

wait "$GA_PID" || true
EXIT_CODE=$?

# Summary
python3 - "$CSV" "$SUMMARY" <<'PY'
import csv, sys, statistics
csv_path, summary_path = sys.argv[1], sys.argv[2]
rss = []
vsz = []
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rss.append(float(row["rss_mb"]))
        vsz.append(float(row["vsz_mb"]))
if not rss:
    print("No samples collected")
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

echo "==> GA exit code: $EXIT_CODE"
echo "==> CSV:     $CSV"
echo "==> Summary: $SUMMARY"
