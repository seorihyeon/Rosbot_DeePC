#!/bin/bash
cd /ws
rm -rf build/rosbot_deepc install/rosbot_deepc log
source /opt/ros/jazzy/setup.bash
colcon build --packages-select rosbot_deepc
source /ws/install/setup.bash
ros2 launch rosbot_deepc deepc_collect.launch.py
