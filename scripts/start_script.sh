#!/bin/bash
unset ROS_LOCALHOST_ONLY
unset ROS_AUTOMATIC_DISCOVERY_RANGE
unset ROS_STATIC_PEERS
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI

export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

source /opt/ros/jazzy/setup.bash
source /ws/install/setup.bash
