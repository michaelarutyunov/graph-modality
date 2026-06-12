#!/usr/bin/env bash
# Autonomous v4_think extraction runner with checkpoints.
# Logs to s2_extraction/_v4_run.log
set -uo pipefail
cd "$(dirname "$0")/.."

OUT_DIR=s1_data/graphs/v4_think/free_text
LOG=s2_extraction/_v4_run.log
CHECKPOINTS=(100 250 500 750 1250)

count() { ls "$OUT_DIR" 2>/dev/null | wc -l | tr -d ' '; }

# wait for any already-running extractor.py invocation to finish to avoid
# duplicate concurrent API calls against the same cache dir
while pgrep -f "s2_extraction/extractor.py" > /dev/null; do
  sleep 5
done

for ckpt in "${CHECKPOINTS[@]}"; do
  echo "=== targeting checkpoint $ckpt ===" >> "$LOG"
  while [ "$(count)" -lt "$ckpt" ]; do
    PYTHONPATH=. uv run python s2_extraction/extractor.py \
      --backend deepseek-think --prompt-version v4 \
      --out-dir "$OUT_DIR" --order stratified --limit 50 --concurrency 3 >> "$LOG" 2>&1
    n=$(count)
    echo "--- progress: $n graphs in $OUT_DIR ---" >> "$LOG"
    if [ "$n" -ge 1250 ]; then
      break
    fi
  done
  echo "=== checkpoint $ckpt reached ($(count) graphs) — running quality report ===" >> "$LOG"
  PYTHONPATH=. uv run python s2_extraction/quality_report.py --graph-dir "$OUT_DIR" >> "$LOG" 2>&1
  echo "=== CHECKPOINT_DONE $ckpt ===" >> "$LOG"
  if [ "$(count)" -ge 1250 ]; then
    break
  fi
done
echo "=== ALL DONE ===" >> "$LOG"
