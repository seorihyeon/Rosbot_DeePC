#!/usr/bin/env bash
set -euo pipefail

RESET_SERVICE="${1:-/reset_rosbot}"

echo "Calling reset service ${RESET_SERVICE} ..."
ros2 service call "${RESET_SERVICE}" std_srvs/srv/Trigger "{}"
