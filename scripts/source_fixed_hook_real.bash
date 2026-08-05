#!/usr/bin/env bash

# Source this file; do not execute it.  The bluerov workspace's px4_msgs is
# synchronized from the active PX4-Autopilot checkout before it is built.  This
# script prevents older px4_ws overlays from taking precedence over it.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: source this script instead of executing it:" >&2
  echo "  source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash" >&2
  exit 2
fi

_fixed_hook_ros_setup=/opt/ros/jazzy/setup.bash
_fixed_hook_control_setup=/home/yecheng/bluerov_ws/install/local_setup.bash
_fixed_hook_px4_prefix=/home/yecheng/bluerov_ws/install/px4_msgs

for _fixed_hook_required in \
  "${_fixed_hook_ros_setup}" \
  "${_fixed_hook_control_setup}"
do
  if [[ ! -r "${_fixed_hook_required}" ]]; then
    echo "ERROR: required setup file is missing: ${_fixed_hook_required}" >&2
    unset _fixed_hook_required
    return 1
  fi
done
unset _fixed_hook_required

# ament's prepend-unique helper does not move an entry that is already later
# in a colon-separated variable.  Remove only px4_msgs entries from the two
# workspaces so local_setup can reliably put the firmware-matched copy first,
# even in a terminal that previously loaded the wrong overlay.
_fixed_hook_remove_prefix_from_var()
{
  local _fixed_hook_var_name="${1}"
  local _fixed_hook_remove_prefix="${2}"
  local _fixed_hook_old_value="${!_fixed_hook_var_name-}"
  local _fixed_hook_new_value=""
  local _fixed_hook_entry
  local IFS=:

  for _fixed_hook_entry in ${_fixed_hook_old_value}; do
    [[ -z "${_fixed_hook_entry}" ]] && continue
    if [[ "${_fixed_hook_entry}" == "${_fixed_hook_remove_prefix}"* ]]; then
      continue
    fi
    if [[ -z "${_fixed_hook_new_value}" ]]; then
      _fixed_hook_new_value="${_fixed_hook_entry}"
    else
      _fixed_hook_new_value="${_fixed_hook_new_value}:${_fixed_hook_entry}"
    fi
  done
  export "${_fixed_hook_var_name}=${_fixed_hook_new_value}"
}

source "${_fixed_hook_ros_setup}"

for _fixed_hook_path_var in \
  AMENT_PREFIX_PATH \
  CMAKE_PREFIX_PATH \
  COLCON_PREFIX_PATH \
  PYTHONPATH \
  LD_LIBRARY_PATH
do
  _fixed_hook_remove_prefix_from_var \
    "${_fixed_hook_path_var}" "/home/yecheng/px4_ws/install/px4_msgs"
  _fixed_hook_remove_prefix_from_var \
    "${_fixed_hook_path_var}" "/home/yecheng/bluerov_ws/install/px4_msgs"
done
unset _fixed_hook_path_var

source "${_fixed_hook_control_setup}"

export ACADOS_SOURCE_DIR=/home/yecheng/acados
export LD_LIBRARY_PATH=/home/yecheng/acados/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

_fixed_hook_active_prefix="$(ros2 pkg prefix px4_msgs 2>/dev/null)"
_fixed_hook_versions="$(python3 -c \
  "from px4_msgs.msg import VehicleCommandAck as A, VehicleStatus as S; print(f'{A.MESSAGE_VERSION} {S.MESSAGE_VERSION}')" \
  2>/dev/null)"

if [[ "${_fixed_hook_active_prefix}" != "${_fixed_hook_px4_prefix}" ]]; then
  echo "ERROR: wrong px4_msgs overlay is active: ${_fixed_hook_active_prefix}" >&2
  echo "Expected: ${_fixed_hook_px4_prefix}" >&2
  unset -f _fixed_hook_remove_prefix_from_var
  return 1
fi

if [[ "${_fixed_hook_versions}" != "0 1" ]]; then
  echo "ERROR: px4_msgs versions do not match the current firmware." >&2
  echo "Expected ACK/Status versions '0 1', got '${_fixed_hook_versions}'." >&2
  unset -f _fixed_hook_remove_prefix_from_var
  return 1
fi

echo "Fixed-hook real environment ready:"
echo "  ROS_DISTRO=${ROS_DISTRO}"
echo "  ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}"
echo "  RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}"
echo "  px4_msgs=${_fixed_hook_active_prefix} (ACK/Status ${_fixed_hook_versions})"

unset -f _fixed_hook_remove_prefix_from_var
unset _fixed_hook_ros_setup
unset _fixed_hook_control_setup
unset _fixed_hook_px4_prefix
unset _fixed_hook_active_prefix
unset _fixed_hook_versions
