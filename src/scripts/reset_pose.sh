#!/usr/bin/env bash
set -eo pipefail

usage() {
  cat <<'EOF'
Usage:
  reset_pose.sh
  reset_pose.sh /reset_rosbot
  reset_pose.sh x y yaw [reset_service]
EOF
}

as_float() {
  case "$1" in
    *.*|*e*|*E*) printf '%s' "$1" ;;
    *) printf '%s.0' "$1" ;;
  esac
}

RESET_SERVICE="/reset_rosbot"
RESET_X="0.0"
RESET_Y="0.0"
RESET_YAW="0.0"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

case "$#" in
  0)
    ;;
  1)
    if [[ "$1" != /* ]]; then
      usage >&2
      exit 2
    fi
    RESET_SERVICE="$1"
    ;;
  3)
    RESET_X="$(as_float "$1")"
    RESET_Y="$(as_float "$2")"
    RESET_YAW="$(as_float "$3")"
    ;;
  4)
    RESET_X="$(as_float "$1")"
    RESET_Y="$(as_float "$2")"
    RESET_YAW="$(as_float "$3")"
    RESET_SERVICE="$4"
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

printf 'Calling reset service %s: x=%s, y=%s, yaw=%s\n' \
  "${RESET_SERVICE}" "${RESET_X}" "${RESET_Y}" "${RESET_YAW}"

ros2 service call "${RESET_SERVICE}" rosbot_interfaces/srv/ResetPose \
  "{x: ${RESET_X}, y: ${RESET_Y}, yaw: ${RESET_YAW}}"
