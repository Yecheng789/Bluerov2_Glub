# Glub Fixed-Hook Experiment: Operator Guide

Use this procedure to record a hook-engagement pose and run a supervised
fixed-target retrieval experiment with Glub. Commands assume the lab workspace
at `/home/yecheng/bluerov_ws` and ROS 2 Jazzy. For Splash, use the
[Splash operator guide](SPLASH_FIXED_HOOK_ZH.md).

The robot approaches pre-hook, aligns to the recorded attitude, moves forward
to Hook, holds for operator confirmation, and reverses to pre-hook:

```text
PLAN_TO_PREHOOK -> TRACK_TO_PREHOOK -> PREHOOK_REACHED
-> GO_FORWARD -> WAIT_HOOK -- operator H --> GO_BACK -> COMPLETE
```

## Before you start

Keep a trained operator at QGroundControl (QGC), with working manual takeover
and an emergency stop. Keep people clear of the thrusters and travel corridor.
Use a supervised pool setup and check buoyancy, ballast, and tether clearance
before enabling motion. Zero thrust is not position holding: a positively
buoyant robot can rise after DISARM or loss of control output.

Verify these settings against the actual installation:

| Setting | Glub profile |
| --- | --- |
| PX4 namespace / MoCap rigid body | `/glub` / `glub` |
| Raw pose topic | `/mocap/glub/pose` |
| World / body convention | NED / FRD |
| Vehicle model | `fossen_real`, `standard` |
| MAVLink target system / component | `3` / `1`; verify on the vehicle |
| Raw MoCap pool bounds | `0 9 -2.5 2.5 0 3` in `xmin xmax ymin ymax zmin zmax` order |
| Pool safety margin | `0.25 m` inward from each transformed boundary |
| Pre-hook offset | `0.50 m` behind Hook along the recorded horizontal heading |
| Approach / forward / retreat reference speed | `0.06 / 0.09 / 0.09 m/s` |
| Attitude-reference rate limit | `8 deg/s` |
| Pre-hook arrival | Position `0.05 m`, depth `0.03 m`, full attitude `5 deg`, forward axis `5 deg`, yaw `3 deg`, continuously for `1.0 s` |
| Hook arrival | Position `0.05 m`, depth `0.03 m`, full attitude `5 deg` |
| Thrust / torque saturation | `0.12 / 0.02` |
| NMPC / path-safety monitor | Nominally `25 Hz / 2 Hz` |
| Hook confirmation | H key in a dedicated terminal; no automatic wait timeout |

Pool size alone does not establish its origin or axis signs. Verify coordinate
conversion, the measured boundaries, and the robot's body-forward direction.
Repeat the checks after changes to the marker definition, rigid-body axes,
mounting, ballast, pool frame, or flight-controller configuration. Do not copy
a target quaternion from Splash or change a target's provenance fields to
make it pass validation.

Only one control stack and one robot profile may run at a time. Do not run
this launch alongside `stabilized_control_real.launch.py`,
`offboard_enable_real.launch.py`, `fixed_hook_mpc_june23.launch.py`, a Splash
profile, or another PID/MPC/Offboard heartbeat publisher. QGC, rosbag, and
subscribe-only monitors may run alongside it.

## 1. Build and prepare the environment

With the robot DISARMED and control launches stopped, build from a fresh Bash
terminal. Confirm that `src/px4_msgs` matches the deployed PX4 firmware first.

```bash
cd /home/yecheng/bluerov_ws
source /opt/ros/jazzy/setup.bash

export ACADOS_SOURCE_DIR=/home/yecheng/acados
export LD_LIBRARY_PATH=/home/yecheng/acados/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

colcon build --packages-select px4_msgs --symlink-install --cmake-clean-cache
source install/local_setup.bash
colcon build --packages-select bluerov2_control --symlink-install

source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash
```

Stop if either build or the environment check fails. Source the helper in
every ROS terminal and again after rebuilding. Do not source another PX4
workspace afterward. The helper configures ROS/acados paths and RMW; it does
not configure the robot network, start Offboard, or arm the vehicle.

## 2. Start the DDS Agent

Check for an existing Agent before starting another:

```bash
pgrep -af 'micro-xrce-dds-agent|MicroXRCEAgent'
ss -lunp | grep -E ':8888'
```

If none is running and PX4 is configured to connect to this computer on UDP
port 8888, keep the following running in its own terminal:

```bash
micro-xrce-dds-agent udp4 -p 8888
```

Use the actual connection settings if the installation uses a different
host, port, or serial connection. Do not start a duplicate Agent.

## 3. Connect QGroundControl

Keep QGC open throughout the experiment:

```bash
cd /home/yecheng/Downloads
./QGroundControl-x86_64.AppImage
```

Confirm connection to the intended vehicle. Test joystick axes, mode
switching, DISARM, and the emergency stop under safe conditions. Verify the
configured Offboard-loss actions, including `COM_OF_LOSS_T` and
`COM_OBL_RC_ACT`, with the on-site safety lead. Do not assume that an aircraft
fallback mode is suitable for an underwater vehicle.

## 4. Check PX4 feedback and MoCap

In a diagnostic terminal:

```bash
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash

ros2 pkg prefix px4_msgs
python3 -c "from px4_msgs.msg import VehicleCommandAck as A, VehicleStatus as S; print('ACK/Status =', A.MESSAGE_VERSION, S.MESSAGE_VERSION)"
ros2 topic list | sort | grep -E '^/glub/fmu/(in|out)/'

STATUS_TOPIC=$(python3 -c "from px4_msgs.msg import VehicleStatus as M; v=int(M.MESSAGE_VERSION); print('/glub/fmu/out/vehicle_status'+(f'_v{v}' if v else ''))")
ros2 topic echo "$STATUS_TOPIC" --once --qos-profile sensor_data
ros2 topic info -v /glub/fmu/out/vehicle_control_mode
ros2 topic echo /glub/fmu/out/vehicle_control_mode --once --qos-profile sensor_data
timeout 5s ros2 topic hz /glub/fmu/out/vehicle_control_mode
```

The expected package prefix is `/home/yecheng/bluerov_ws/install/px4_msgs`,
with ACK/Status versions `0 1`. The status message must identify the intended
vehicle (`system_id: 3`, `component_id: 1` for this profile). DDS topic names
alone are insufficient: feedback must actually arrive and agree with QGC.
Resolve firmware/message or ID mismatches before proceeding; do not disguise
them by changing topic suffixes.

Check raw MoCap in the same terminal:

```bash
ros2 topic info -v /mocap/glub/pose
ros2 topic echo /mocap/glub/pose --once --qos-profile sensor_data
timeout 20s ros2 topic hz /mocap/glub/pose
date +%s.%N
ros2 topic echo /mocap/glub/pose --qos-profile sensor_data
```

Observe the continuous echo for at least five seconds, then press `Ctrl-C`.
The `timeout` commands end their monitors automatically; exit status 124 at
the specified duration is expected.

- Require exactly one pose publisher and a stable update rate.
- Check that `header.stamp` is nonzero and increasing, `frame_id` is stable,
  and pose values are finite with a valid quaternion.
- Synchronize the MoCap and control-computer clocks. Check timestamp age
  against the `0.20 s` freshness limit; do not disable freshness checks.
- Verify that stationary measurements do not jump and that body +X points
  toward the bow after the configured frame/correction transform.
- Resolve DDS deserialization errors before operating. Use a validated ROS
  and RMW setup across machines; do not change `ROS_DOMAIN_ID` without
  preserving the required MoCap connection.

If an orientation correction is required, use a known, level calibration pose:

```bash
ros2 run bluerov2_control calibrate_mocap_orientation_correction \
  --topic /mocap/glub/pose \
  --samples 160
```

Require `Calibration quality: OK` before saving the reported
`orientation_correction_quat_xyzw`. Reject an `UNSTABLE` result even if a
quaternion is printed. Apply the same validated transform to the live pose
and recorded target at launch; do not calibrate away an incorrect rigid-body
identity or tracking failure.

## 5. Record the Hook target

Keep automatic control stopped. Before manual positioning, confirm that all
four PX4 input topics have **zero publishers**; PX4 subscriptions are expected:

```bash
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash
ros2 topic info -v /glub/fmu/in/offboard_control_mode
ros2 topic info -v /glub/fmu/in/vehicle_command
ros2 topic info -v /glub/fmu/in/vehicle_thrust_setpoint
ros2 topic info -v /glub/fmu/in/vehicle_torque_setpoint
```

Manually engage the hook and let the vehicle and box settle at the pose to be
reproduced. Record from Glub's raw MoCap topic:

```bash
cd /home/yecheng/bluerov_ws
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash

TARGET_CFG=/home/yecheng/bluerov_ws/src/bluerov2_control/experiments/payload_retrieval/config/hooked_box_target_pose_glub_$(date +%Y%m%d_%H%M%S).json

if ros2 run bluerov2_control record_mocap_target_pose \
  --topic /mocap/glub/pose \
  --message-type pose \
  --samples 160 \
  --timeout-sec 20 \
  --max-message-age-sec 0.20 \
  --output-file "$TARGET_CFG"
then
  python3 -m json.tool "$TARGET_CFG"
  echo "Glub target file: $TARGET_CFG"
else
  unset TARGET_CFG
  echo "Target failed validation; do not launch with this recording."
fi
```

Require `validation.passed: true` and `source_topic: /mocap/glub/pose`.
Check the reported sampling duration, timestamp freshness, and position and
attitude dispersion. Save the printed absolute path; new terminals do not
inherit `TARGET_CFG`. Use a new file rather than overwriting a recording.

Do not move the box or handle after recording. Re-record if the target,
marker geometry, rigid-body frame, or mounting changes. Do not edit
`source_topic` or validation results to reuse an incompatible file.

## 6. Start data recording and position the robot

Before launching control, start rosbag in a dedicated terminal:

```bash
cd /home/yecheng/bluerov_ws
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash

BAG_DIR=/home/yecheng/bluerov_ws/fixed_hook_bags/fixed_hook_glub_$(date +%Y%m%d_%H%M%S)
mkdir -p /home/yecheng/bluerov_ws/fixed_hook_bags

ACK_TOPIC=$(python3 -c "from px4_msgs.msg import VehicleCommandAck as M; v=int(M.MESSAGE_VERSION); print('/glub/fmu/out/vehicle_command_ack'+(f'_v{v}' if v else ''))")
STATUS_TOPIC=$(python3 -c "from px4_msgs.msg import VehicleStatus as M; v=int(M.MESSAGE_VERSION); print('/glub/fmu/out/vehicle_status'+(f'_v{v}' if v else ''))")
ros2 topic info -v "$ACK_TOPIC"
ros2 topic info -v "$STATUS_TOPIC"

ros2 bag record -o "$BAG_DIR" --topics \
  /mocap/glub/pose \
  /mocap/glub/odom_ekf_fixed_hook \
  /mocap/glub/vehicle_odometry_fixed_hook \
  /glub/fmu/in/offboard_control_mode \
  /glub/fmu/in/vehicle_command \
  /glub/fmu/in/vehicle_thrust_setpoint \
  /glub/fmu/in/vehicle_torque_setpoint \
  "$ACK_TOPIC" \
  /glub/fmu/out/vehicle_control_mode \
  /glub/fmu/out/vehicle_odometry \
  "$STATUS_TOPIC" \
  /glub/fmu/out/failsafe_flags \
  /bluerov2/fixed_hook/offboard_request_enable \
  /bluerov2/fixed_hook/mission_enable \
  /bluerov2/fixed_hook/controller_heartbeat \
  /bluerov2/trial_event \
  /rosout \
  /parameter_events
```

Controller topics appear when the launch starts; leave the recorder running
through shutdown. Also record external video showing the robot, box, approach
direction, Hook confirmation, retreat, and safe stop. Note the bag path and
video time reference for the trial.

Manually disengage without moving the box, move the robot to a safe starting
point, and DISARM in QGC. Keep the start, pre-hook, Hook, and travel corridor
inside the operating bounds. Check the tether and clearance from all people
and equipment before launching control.

## 7. Launch control with motion disabled

In a dedicated launch terminal, enter the absolute path printed in Section 5
when prompted. The launch requires an explicit target; it does not select one
automatically:

```bash
cd /home/yecheng/bluerov_ws
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash

read -r -p 'Absolute path to the validated Glub target JSON: ' TARGET_CFG
if [[ "$TARGET_CFG" == /* && -r "$TARGET_CFG" ]]; then
  ros2 launch bluerov2_control fixed_hook_pose_validation.launch.py \
    target_config:="$TARGET_CFG"
else
  echo "Target file is missing or is not an absolute path; launch not started."
fi
```

Review the printed target, pre-hook coordinates, pool bounds, frame transform,
and model settings. Pre-hook is computed in NED as
`goal - 0.50 * [cos(yaw), sin(yaw), 0]`, with the same depth as Hook. Both the
target and live robot center must remain inside the reduced pool bounds.

If the installation requires overrides, add them to this single launch
command before running it; do not launch a second instance:

- `orientation_correction_quat_xyzw`: the verified calibration in `x y z w`
  order; retain `target_orientation_correction_mode=auto` for raw recordings.
- `prehook_static_obstacles_ned_xyxy`: measured NED horizontal rectangles in
  `xmin xmax ymin ymax` groups, separated by semicolons. Each is inflated by
  the `0.20 m` robot radius and `0.05 m` obstacle margin.
- Pool/frame/model/MAVLink settings: use only values verified on site.

The obstacle list is empty by default. A* checks operating bounds and the
configured static rectangles, not unobserved boxes, moving obstacles, people,
or tethers. Do not substitute simulation geometry or the recorded ROV pose
for measured box geometry. The intentional hook-contact corridor must remain
valid; resolve any obstacle/corridor conflict before enabling motion.

Launch starts the EKF, odometry adapter, NMPC, CSV logger, and Offboard manager.
It does **not** arm, request Offboard, or enable motion. Thrust/torque remain
zero until the separate permission and health gates pass. Do not reduce
`thrust_sat` below the `standard` model's allowed minimum of `0.075`; the
default `0.12` still requires a supervised check of actual depth-hold authority.

## 8. Complete the post-launch safety checks

Keep the vehicle DISARMED. In the diagnostic terminal:

```bash
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash

ros2 node list | sort | grep -E 'fixed_hook|mocap_ekf|offboard_enable'
ros2 topic info -v /glub/fmu/in/offboard_control_mode
ros2 topic info -v /glub/fmu/in/vehicle_command
ros2 topic info -v /glub/fmu/in/vehicle_thrust_setpoint
ros2 topic info -v /glub/fmu/in/vehicle_torque_setpoint

timeout 5s ros2 topic hz /mocap/glub/odom_ekf_fixed_hook
timeout 5s ros2 topic hz /mocap/glub/vehicle_odometry_fixed_hook
timeout 5s ros2 topic hz /glub/fmu/out/vehicle_odometry
ros2 topic echo /glub/fmu/out/vehicle_odometry --once --qos-profile sensor_data

ros2 param get /mpc_fixed_hook_pose_validation model_type
ros2 param get /mpc_fixed_hook_pose_validation robot_type
ros2 param get /mpc_fixed_hook_pose_validation goal_x
ros2 param get /mpc_fixed_hook_pose_validation goal_y
ros2 param get /mpc_fixed_hook_pose_validation goal_z
ros2 param get /mpc_fixed_hook_pose_validation pre_approach_x
ros2 param get /mpc_fixed_hook_pose_validation pre_approach_y
ros2 param get /mpc_fixed_hook_pose_validation pre_approach_z
ros2 param get /mpc_fixed_hook_pose_validation fixed_hook_line_position_mode
ros2 param get /mpc_fixed_hook_pose_validation require_mission_enable
ros2 param get /mpc_fixed_hook_pose_validation operating_bounds_enable
ros2 service type /bluerov2/fixed_hook/confirm_hook
```

Require exactly one publisher per PX4 input: the Offboard manager owns mode
commands/heartbeat, and NMPC owns thrust/torque. Confirm `fossen_real`,
`standard`, the intended goal/pre-hook coordinates, and enabled mission and
operating-boundary guards plus Position-like line mode. The confirmation
service must be `std_srvs/srv/Trigger`.

MoCap supplies position, attitude, and linear velocity; PX4 supplies BODY_FRD
angular rates through `vehicle_odometry`. Both streams and the adapted MPC
odometry must remain fresh. Stop on stale state, pose jumps, invalid tilt,
solver failure, duplicate publishers, or an unexpected mode change. Never
bypass these checks to force movement.

## 9. Request Offboard

With the vehicle still DISARMED, clear both permission gates and monitor ACKs
in a dedicated terminal:

```bash
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash

ros2 topic pub --once /bluerov2/fixed_hook/mission_enable \
  std_msgs/msg/Bool '{data: false}'
ros2 topic pub --once /bluerov2/fixed_hook/offboard_request_enable \
  std_msgs/msg/Bool '{data: false}'

ACK_TOPIC=$(python3 -c "from px4_msgs.msg import VehicleCommandAck as M; v=int(M.MESSAGE_VERSION); print('/glub/fmu/out/vehicle_command_ack'+(f'_v{v}' if v else ''))")
ros2 topic echo "$ACK_TOPIC" --qos-profile sensor_data
```

After all checks pass and manual takeover is ready, request Offboard from
another terminal:

```bash
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash

ros2 topic pub --once /bluerov2/fixed_hook/offboard_request_enable \
  std_msgs/msg/Bool '{data: true}'
ros2 topic echo /glub/fmu/out/vehicle_control_mode --once --qos-profile sensor_data
```

Require a matching accepted mode-command ACK and actual Offboard feedback in
both QGC and `flag_control_offboard_enabled: true`. An ACK alone is not proof
of the active mode. The manager does not arm the vehicle automatically.

## 10. Prepare Hook confirmation, then arm and enable motion

Open a separate interactive terminal for Hook confirmation and leave it in
the foreground:

```bash
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash
ros2 run bluerov2_control confirm_hook_keyboard
```

Do not press H yet. It must be pressed in this terminal, not in the launch
terminal, only after `WAIT_HOOK` and visual confirmation of engagement.

ARM manually in QGC with personnel clear and the operator ready. In the
motion-permission terminal, verify both flags before proceeding:

```bash
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash
ros2 topic echo /glub/fmu/out/vehicle_control_mode --once --qos-profile sensor_data
```

Only if `flag_armed: true` and `flag_control_offboard_enabled: true`, all
health checks pass, data recording is active, and the travel corridor is
clear, mark the trial start and grant motion permission:

```bash
ros2 run bluerov2_control mark_trial_event start \
  --note 'Glub fixed-hook mission enabled'
ros2 topic pub --once /bluerov2/fixed_hook/mission_enable \
  std_msgs/msg/Bool '{data: true}'
```

## 11. Supervise approach, engagement, and retreat

- **Approach:** A* plans to pre-hook and NMPC tracks the result. The `2 Hz`
  monitor checks safety/deviation and requests searches as needed; it does
  not detect moving obstacles. Planning failure does not authorize an
  unchecked straight-line fallback.
- **Alignment:** position, depth, and all three attitude gates must pass
  continuously for `1.0 s`. Alignment has no default elapsed-time timeout;
  health and boundary checks remain active. Inspect the reported errors and
  target/frame validity if alignment does not converge. Do not widen gates
  simply to force entry.
- **GO_FORWARD:** A* is inactive. The reference advances from pre-hook to
  Hook along the recorded horizontal heading at `0.09 m/s`, with fixed NED
  depth, zero vertical velocity reference, and the recorded attitude.
  Ordinary tracking errors do not rewind reference progress. These are
  reference constraints, not a guarantee of perfectly straight physical
  motion; intervene if the actual trajectory is unsafe.
- **WAIT_HOOK:** the robot holds the recorded pose without an automatic
  timeout. Once the log confirms entry, allow at least `0.25 s`, visually
  confirm engagement, and press a fresh H in the confirmation terminal.
- **GO_BACK / COMPLETE:** accepted confirmation starts reverse travel along
  the same line at `0.09 m/s`. The controller holds at pre-hook after
  completion; it does not automatically DISARM.

Arrival is latched on entry to `WAIT_HOOK`; later pose drift alone does not
invalidate H confirmation. Mission permission, fresh Armed/Offboard feedback,
valid odometry/state, solver availability, and fresh MPC commands are still
required. The keyboard client prints acceptance or rejection and exits. If
rejected, resolve the stated condition and rerun it; early keypresses are not
queued for later use.

Mark success only after observing the intended physical outcome and checking
controller completion. Note any difference between physical retrieval and
the reported controller state:

```bash
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash
ros2 run bluerov2_control mark_trial_event success \
  --note 'Operator confirmed Glub retrieval and return to pre-hook'
```

## 12. Stop safely

### Normal stop

With the robot under supervision, DISARM in QGC and verify that the thrusters
have stopped. Plan for buoyancy and tether motion; DISARM is not depth hold.
Then clear permissions while the launch and recorder are still running:

```bash
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash

ros2 topic pub --once /bluerov2/fixed_hook/mission_enable \
  std_msgs/msg/Bool '{data: false}'
ros2 topic pub --once /bluerov2/fixed_hook/offboard_request_enable \
  std_msgs/msg/Bool '{data: false}'
ros2 run bluerov2_control mark_trial_event stop \
  --note 'Operator stopped Glub fixed-hook trial'
ros2 topic echo /glub/fmu/out/vehicle_control_mode --once --qos-profile sensor_data
```

Confirm `flag_armed: false`, leave Offboard in QGC, and verify the mode change.
Press `Ctrl-C` in the launch terminal, then in the rosbag terminal and any
remaining monitors/confirmation terminal. Wait for rosbag to finish closing.
`offboard_request_enable=false` stops further mode requests; it does not
switch PX4 out of an already active mode.

### Emergency stop and fault recovery

Use the verified physical emergency stop or QGC DISARM/manual takeover
immediately. Do not wait for ROS commands or terminal output. Once the
vehicle is safe, clear both permissions with the commands above and stop the
control launch. ROS permission messages and subscribe-only watchdogs are
not substitutes for a working emergency stop.

On `Mission enable revoked`, a controller-health latch, tracking loss, an
attitude jump, solver failure, or an unexpected mode change, do not wait for
automatic recovery. DISARM first, clear permissions, investigate the cause,
and repeat the preflight checks. Resume only through the explicit sequence
**Offboard request -> manual ARM -> mission enable**. State recovery alone
does not restore motion permission.

## 13. Check and retain the trial data

The launch automatically creates a timestamped directory under
`/home/yecheng/bluerov_ws/bluerov2_payload_retrieval_trials` containing:

- `samples.csv`: raw MoCap, EKF/controller odometry, control inputs, and
  permission/mode feedback;
- `events.csv`: operator event markers;
- `metadata.json`: target provenance, topics, and experiment configuration.

After shutdown, inspect the directory printed by the logger and the bag path
from Section 6:

```bash
ls -lt /home/yecheng/bluerov_ws/bluerov2_payload_retrieval_trials | head
```

Keep the target JSON, CSV directory, bag, launch logs, and video together in
the experiment record. Check that data covers approach, Hook confirmation,
retreat, and shutdown; distinguish operator-confirmed physical success from
controller `COMPLETE`. Do not overwrite raw recordings. Use the
[logging and analysis guide](README.md) for offline analysis.
