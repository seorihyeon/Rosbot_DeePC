#!/usr/bin/env bash
set -eo pipefail

usage() {
  cat <<'EOF'
Usage:
  reset_pose.sh
  reset_pose.sh /reset_rosbot
  reset_pose.sh x y yaw [z] [reset_service]

Environment:
  RESET_NODE=/reset_rosbot_server
EOF
}

as_float() {
  case "$1" in
    *.*|*e*|*E*) printf '%s' "$1" ;;
    *) printf '%s.0' "$1" ;;
  esac
}

RESET_SERVICE="/reset_rosbot"
RESET_NODE="${RESET_NODE:-/reset_rosbot_server}"
SET_POSE=false

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

case "$#" in
  0)
    ;;
  1)
    RESET_SERVICE="$1"
    ;;
  3)
    SET_POSE=true
    RESET_X="$(as_float "$1")"
    RESET_Y="$(as_float "$2")"
    RESET_YAW="$(as_float "$3")"
    RESET_Z="0.1"
    ;;
  4)
    SET_POSE=true
    RESET_X="$(as_float "$1")"
    RESET_Y="$(as_float "$2")"
    RESET_YAW="$(as_float "$3")"
    if [[ "$4" == /* ]]; then
      RESET_Z="0.1"
      RESET_SERVICE="$4"
    else
      RESET_Z="$(as_float "$4")"
    fi
    ;;
  5)
    SET_POSE=true
    RESET_X="$(as_float "$1")"
    RESET_Y="$(as_float "$2")"
    RESET_YAW="$(as_float "$3")"
    RESET_Z="$(as_float "$4")"
    RESET_SERVICE="$5"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

ROS_DISTRO="${ROS_DISTRO:-humble}"
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source /ws/install/setup.bash
export ROS_LOG_DIR="${ROS_LOG_DIR:-/ws/log/ros}"
mkdir -p "${ROS_LOG_DIR}"

if [[ "${SET_POSE}" == true ]]; then
  printf 'Setting reset pose on %s: x=%s, y=%s, z=%s, yaw=%s\n' \
    "${RESET_NODE}" "${RESET_X}" "${RESET_Y}" "${RESET_Z}" "${RESET_YAW}"
  ros2 param set "${RESET_NODE}" target_x "${RESET_X}"
  ros2 param set "${RESET_NODE}" target_y "${RESET_Y}"
  ros2 param set "${RESET_NODE}" target_z "${RESET_Z}"
  ros2 param set "${RESET_NODE}" target_yaw "${RESET_YAW}"
fi

echo "Calling reset service ${RESET_SERVICE} ..."
ros2 service call "${RESET_SERVICE}" std_srvs/srv/Trigger "{}"
