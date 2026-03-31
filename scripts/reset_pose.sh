#!/usr/bin/env bash
set -euo pipefail

TRIGGER_TOPIC="${1:-/reset_rosbot}"
DONE_TOPIC="${2:-/reset_rosbot_done}"
RETRY_PERIOD="${3:-0.5}"

waiter_pid=""

cleanup() {
  if [[ -n "${waiter_pid}" ]] && kill -0 "${waiter_pid}" 2>/dev/null; then
    kill "${waiter_pid}" 2>/dev/null || true
    wait "${waiter_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Waiting for reset completion on ${DONE_TOPIC} ..."

ros2 topic echo --once "${DONE_TOPIC}" std_msgs/msg/Empty >/dev/null 2>&1 &
waiter_pid=$!

sleep 0.2

while kill -0 "${waiter_pid}" 2>/dev/null; do
  ros2 topic pub --once "${TRIGGER_TOPIC}" std_msgs/msg/Empty "{}" >/dev/null 2>&1 || true
  sleep "${RETRY_PERIOD}"
done

wait "${waiter_pid}"
echo "Reset completed."