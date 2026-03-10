#!/bin/bash

source /ws/install/setup.bash
ros2 launch rosbot_gazebo simulation.launch.py \
	use_sim:=True robot_model:=rosbot \
	gz_headless_mode:=True rviz:=False \
	gz_world:=/ws/src/worlds/flat_empty.sdf \
	x:=0.0 y:=0.0 yaw:=0.0
