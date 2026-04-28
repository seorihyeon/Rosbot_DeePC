#!/bin/bash
pkill -f "gz sim"
pkill -f controller_manager
pkill -f parameter_bridge
pkill -f robot_state_publisher
pkill -f ros_gz
pkill -f spawner
