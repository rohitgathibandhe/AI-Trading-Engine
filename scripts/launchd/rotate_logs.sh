#!/bin/bash
# Nightly copy-truncate rotation for the agent's launchd stdout/stderr log (grew to 336 MB).
# copy-truncate is safe here: launchd opens the file O_APPEND, so the writer continues at offset 0.
# Runs 20:00 IST, after every EOD job that reads the day's log (latest: daily_self_diagnosis 16:20).
# Keeps KEEP_DAYS gzipped archives in state/log_archive/.
set -u
STATE=/Users/Rohit/AI-Trading-Engine/data_engine/market_ai/state
ARCH="$STATE/log_archive"
KEEP_DAYS=14
MIN_BYTES=$((5 * 1024 * 1024))
mkdir -p "$ARCH"
for f in "$STATE/intraday_v83_runner.log"; do
  [ -f "$f" ] || continue
  size=$(stat -f %z "$f")
  [ "$size" -lt "$MIN_BYTES" ] && continue
  out="$ARCH/$(basename "$f" .log)_$(date +%Y%m%d_%H%M%S).log.gz"
  gzip -c "$f" > "$out" && : > "$f" && echo "$(date '+%F %T') rotated $(basename "$f") ($((size/1024/1024)) MB) -> $(basename "$out")"
done
find "$ARCH" -name '*.log.gz' -mtime +"$KEEP_DAYS" -print -delete
