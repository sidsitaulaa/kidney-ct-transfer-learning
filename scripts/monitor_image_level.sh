#!/bin/bash
# Log host and container memory every INTERVAL seconds while the image-level container exists.
# Writes results/image_level/memory.log; stops a few polls after the container has exited.
#   scripts/monitor_image_level.sh [container-name]
NAME=${1:-kidney-image-level}
INTERVAL=${INTERVAL:-20}
LOG="$(dirname "$(readlink -f "$0")")/../results/image_level/memory.log"
mkdir -p "$(dirname "$LOG")"
[ -s "$LOG" ] || echo "time host_avail_mb container_state mem_usage mem_pct cpu_pct" >> "$LOG"
gone=0
while true; do
  avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  state=$(docker inspect -f '{{.State.Status}}{{if .State.OOMKilled}}+OOMKILLED{{end}}' "$NAME" 2>/dev/null || echo absent)
  stats=$(docker stats --no-stream --format '{{.MemUsage}}|{{.MemPerc}}|{{.CPUPerc}}' "$NAME" 2>/dev/null | sed 's/ //g; s/|/ /g')
  echo "$(date '+%F %T') ${avail} ${state} ${stats:--}" >> "$LOG"
  case "$state" in running|paused|restarting|created) gone=0 ;; *) gone=$((gone+1)) ;; esac
  [ "$gone" -ge 3 ] && break
  sleep "$INTERVAL"
done
