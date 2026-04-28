#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PATCH="${WORKSPACE_DIR}/src/patches/rosbot_ros_odometry_gt.patch"
TARGET="${WORKSPACE_DIR}/src/rosbot_ros"

if git -C "$TARGET" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
  echo "Patch already applied"
  exit 0
fi

git -C "$TARGET" apply --check "$PATCH"
git -C "$TARGET" apply "$PATCH"
echo "Patch applied"
