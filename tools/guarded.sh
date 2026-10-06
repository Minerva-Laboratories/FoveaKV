#!/usr/bin/env bash
# Run a command so it never pushes a shared machine into the OOM killer:
#  - memory-capped with systemd-run (CAP, default 3G, no swap);
#  - started only when MemAvailable >= START_MB (default 4500);
#  - killed if MemAvailable drops below MIN_MB (default 1500) while it runs, then retried (up to TRIES, default 5).
# Usage: tools/guarded.sh <logfile> <command...>
set -u
log=$1; shift
CAP=${CAP:-3G}; START_MB=${START_MB:-4500}; MIN_MB=${MIN_MB:-1500}; TRIES=${TRIES:-5}
avail() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
for try in $(seq 1 "$TRIES"); do
  while [ "$(avail)" -lt "$START_MB" ]; do sleep 10; done
  systemd-run --user --scope -q -p MemoryMax="$CAP" -p MemorySwapMax=0 "$@" > "$log" 2>&1 &
  pid=$!
  killed=0
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$(avail)" -lt "$MIN_MB" ]; then
      pkill -TERM -P "$pid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null; killed=1
      echo "guard: MemAvailable $(avail) MB < $MIN_MB, stopped (try $try)" >> "$log.guard"
      break
    fi
    sleep 1
  done
  wait "$pid" 2>/dev/null; rc=$?
  [ "$killed" = 0 ] && exit "$rc"
  sleep 30
done
echo "guard: gave up after $TRIES tries" >> "$log.guard"
exit 99
