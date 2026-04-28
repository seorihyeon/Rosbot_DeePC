#!/usr/bin/env bash
set -eo pipefail

ROS_DISTRO="${ROS_DISTRO:-humble}"
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source /ws/install/setup.bash
export ROS_LOG_DIR="${ROS_LOG_DIR:-/ws/log/ros}"
export IGN_LOG_PATH="${IGN_LOG_PATH:-/ws/log/ignition}"
mkdir -p "${ROS_LOG_DIR}" "${IGN_LOG_PATH}"
ros2 launch rosbot_local_bringup sim_with_gt.launch.py
