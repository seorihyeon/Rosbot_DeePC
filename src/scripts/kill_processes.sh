#!/usr/bin/env bash

patterns=(
  "gz sim"
  "ign gazebo"
  "ruby .*ign .*gazebo"
  "controller_manager"
  "parameter_bridge"
  "robot_state_publisher"
  "ros_gz"
  "spawner"
)

for pattern in "${patterns[@]}"; do
  pkill -f "${pattern}" 2>/dev/null || true
done
