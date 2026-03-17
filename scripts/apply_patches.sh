#!/usr/bin/env bash
set -euo pipefail

PATCH="/ws/patches/rosbot_ros_odometry_gt.patch"
TARGET="/ws/src/rosbot_ros"

if git -C "$TARGET" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
  echo "Patch already applied"
  exit 0
fi

git -C "$TARGET" apply --check "$PATCH"
git -C "$TARGET" apply "$PATCH"
echo "Patch applied"