#!/usr/bin/env bash
set -eo pipefail

cd /ws
rm -rf build/rosbot_deepc install/rosbot_deepc log

ROS_DISTRO="${ROS_DISTRO:-humble}"
source "/opt/ros/${ROS_DISTRO}/setup.bash"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/ws/log/ros}"
export IGN_LOG_PATH="${IGN_LOG_PATH:-/ws/log/ignition}"
mkdir -p "${ROS_LOG_DIR}" "${IGN_LOG_PATH}"

colcon build --packages-select rosbot_deepc
source /ws/install/setup.bash
ros2 launch rosbot_deepc deepc_collect.launch.py
