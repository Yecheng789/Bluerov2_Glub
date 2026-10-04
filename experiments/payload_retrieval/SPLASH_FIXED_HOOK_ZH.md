# Splash Fixed-Hook Experiment: Operator Manual

Use this procedure to record Splash's manually hooked pose, approach a point behind it,
move forward to the recorded Hook pose, confirm engagement with `H`, and retreat to the
pre-hook point. This is a fixed-target experiment, with no vision-based target detection,
automatic hook confirmation, return-to-home, or docking.

Splash uses PX4 namespace `/splash`, MoCap rigid body `splash`, and raw pose topic
`/mocap/splash/pose`. The launch requires an explicitly selected target file; it does not
choose one automatically. Do not use a Glub target or copy another robot's quaternion.

## 1. Safety and operating limits

Keep a qualified operator at QGroundControl (QGC) throughout the run. Verify manual control,
all six joystick axes, mode switching, DISARM, and the hardware emergency stop before entering
the water. Keep people clear of the thrusters and keep the box, tether, and travel path in
view. Agree on how the positively buoyant ROV will be safely controlled or recovered when
thrust stops; zero wrench and DISARM are not position holding.

Use only one fixed-hook control chain at a time. Splash and Glub share permission topics,
node names, and the Hook-confirmation service. Do not run this launch alongside a Glub
fixed-hook launch, `fixed_hook_mpc_june23.launch.py`, `stabilized_control_real.launch.py`,
`offboard_enable_real.launch.py`, another PID/MPC, or a standalone heartbeat/wrench publisher.
QGC, rosbag, and subscribe-only monitors may remain running. The wrench watchdog is an
observer, not an emergency stop.

The current Splash profile uses these settings:

| Setting | Value |
| --- | --- |
| MoCap world/body convention; dynamics | NED / FRD; `standard` |
| MAVLink target system/component | `3 / 1`; verify against the real PX4 |
| Raw MoCap pool bounds | `0 9 -2.5 2.5 0 3` m |
| Boundary margin; resulting NED operating bounds | `0.25` m; X `0.25–8.75`, Y `-2.25–2.25`, Z `0.25–2.75` m |
| Pre-hook point | `0.50` m behind the recorded horizontal heading, at Hook depth |
| Reference speeds | Pre-hook `0.06` m/s; forward and retreat `0.09` m/s |
| Reference attitude rate limit | `8` deg/s |
| Pre-hook arrival | Position `0.05` m, depth `0.03` m, full attitude `10` deg, body-X axis `5` deg, yaw `5` deg, continuously for `1.0` s |
| Final Hook arrival | Position `0.05` m, depth `0.03` m, full attitude `5` deg |
| Control and monitoring | NMPC `25` Hz; pre-hook path safety/deviation checks `2` Hz |
| Output limits | `thrust_sat=0.12`, `torque_sat=0.02` |
| Raw MoCap age; maximum EKF coasting | `0.20` s; `2.0` s |

NED Z is positive downward. Confirm the pool origin, axes, rigid-body mounting, ballast,
flight-controller IDs, and physical clearance before using these defaults. Stop and review
the configuration if any differ. Do not enlarge boundaries, loosen attitude gates, or disable
freshness checks to make a failed preflight pass.

A* is used only before pre-hook. The grid is `0.10` m, with a `0.20` m horizontal robot
radius and `0.05` m additional obstacle margin. No interior obstacles are configured by
default: the planner cannot account for unmodelled boxes, people, equipment, or tethers.
For measured static obstacles, use `prehook_static_obstacles_ned_xyxy`, with NED rectangles
in `xmin xmax ymin ymax` order separated by semicolons. Review their clearance and the
fixed Hook corridor before launch. The boundary margin must not be smaller than the robot
radius plus obstacle margin.

## 2. Terminal 0: build and load the environment

Build with the ROV disarmed and all control launches stopped. First verify that
`src/px4_msgs` matches the deployed firmware:

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

Stop if either build or the environment check fails. Every ROS terminal below sources this
helper, which selects the workspace's firmware-matched `px4_msgs` and Fast DDS environment.
Expected prefix: `/home/yecheng/bluerov_ws/install/px4_msgs`; ACK/Status message versions:
`0 1`. Do not source an older PX4 workspace overlay afterward.

## 3. Terminals 1 and 2: DDS Agent and QGC

First check whether an Agent already exists:

```bash
pgrep -af 'micro-xrce-dds-agent|MicroXRCEAgent'
ss -lunp | grep -E ':8888'
```

Only if neither an Agent nor a listener is present, and PX4 is configured to connect to
this computer over UDP port 8888, start one in Terminal 1:

```bash
micro-xrce-dds-agent udp4 -p 8888
```

For a serial connection, different host, or different port, use the verified on-site
configuration instead. Do not start duplicate Agents.

Start QGC in Terminal 2:

```bash
cd /home/yecheng/Downloads
./QGroundControl-x86_64.AppImage
```

Connect to the correct ROV and keep it disarmed. In QGC's MAVLink Console, inspect and save
the following read-only checks:

```text
ver all
uxrce_dds_client status
listener vehicle_control_mode 3
listener vehicle_status 1
param show UXRCE_DDS_CFG
param show UXRCE_DDS_AG_IP
param show UXRCE_DDS_PRT
param show UXRCE_DDS_DOM_ID
param show COM_OF_LOSS_T
param show COM_OBL_RC_ACT
```

The Offboard-loss timeout and action must be approved for this underwater setup. Do not
assume an aircraft Position/Return action is safe or available, and do not change these
parameters blindly. Confirm the manual takeover and emergency-stop tests before proceeding.

## 4. Terminal 3: check live PX4 and MoCap data

```bash
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash

ros2 pkg prefix px4_msgs
python3 -c "from px4_msgs.msg import VehicleCommandAck as A, VehicleStatus as S; print('ACK/Status =', A.MESSAGE_VERSION, S.MESSAGE_VERSION)"
ros2 topic list --no-daemon --spin-time 5 | sort | grep -E '^/splash/fmu/'

STATUS_TOPIC=$(python3 -c "from px4_msgs.msg import VehicleStatus as M; v=int(M.MESSAGE_VERSION); print('/splash/fmu/out/vehicle_status'+(f'_v{v}' if v else ''))")
ACK_TOPIC=$(python3 -c "from px4_msgs.msg import VehicleCommandAck as M; v=int(M.MESSAGE_VERSION); print('/splash/fmu/out/vehicle_command_ack'+(f'_v{v}' if v else ''))")
ros2 topic info -v "$STATUS_TOPIC"
ros2 topic info -v "$ACK_TOPIC"
ros2 topic echo "$STATUS_TOPIC" --once --qos-profile sensor_data
ros2 topic echo /splash/fmu/out/vehicle_control_mode --once --qos-profile sensor_data
timeout 20s ros2 topic hz /splash/fmu/out/vehicle_control_mode
timeout 20s ros2 topic hz /splash/fmu/out/vehicle_odometry
```

Confirm `system_id: 3`, `component_id: 1`, correct message-version suffixes, continuous
feedback, and agreement with the QGC Armed/Offboard state. A listed DDS endpoint alone is
not proof of live data. If PX4 firmware or message versions differ, stop and rebuild against
the correct firmware messages; do not conceal a mismatch by changing topic suffixes.
The `timeout` commands deliberately end after 20 seconds.

Check raw MoCap in the same terminal:

```bash
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash

ros2 topic info -v /mocap/splash/pose
ros2 topic echo /mocap/splash/pose --once --qos-profile sensor_data
timeout 20s ros2 topic hz /mocap/splash/pose
date +%s.%N
timeout 5s ros2 topic echo /mocap/splash/pose --qos-profile sensor_data
```

Require exactly one publisher, stable updates, increasing nonzero timestamps, a stable
nonempty `frame_id`, finite positions, and a valid quaternion. Observe the stationary ROV
for jumps or flips. The MoCap and control computers must use synchronized clocks: accepted
raw poses may be no older than `0.20` s and no more than `0.10` s in the future.
Use a validated, compatible ROS/RMW setup; stop on DDS deserialization errors.

Visually confirm that corrected body +X points toward the bow and that roll/pitch agree
with the physical ROV. If orientation calibration is required, keep automatic control
stopped, hold a known level calibration pose, and run:

```bash
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash
ros2 run bluerov2_control calibrate_mocap_orientation_correction \
  --topic /mocap/splash/pose --samples 160
```

Require `Calibration quality: OK`; never use a quaternion printed with
`UNSTABLE - do not use for MPC`. Save the accepted `orientation_correction_quat_xyzw`
and supply it at launch if needed.
The target and live control must use the same marker definition, mounting, coordinate
conventions, and correction. Do not substitute a Glub quaternion or a startup trim attitude.

## 5. Terminal 4: record the Hook target

Before manual piloting, verify that no ROS controller owns any PX4 control input:

```bash
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash
ros2 topic info -v /splash/fmu/in/offboard_control_mode
ros2 topic info -v /splash/fmu/in/vehicle_command
ros2 topic info -v /splash/fmu/in/vehicle_thrust_setpoint
ros2 topic info -v /splash/fmu/in/vehicle_torque_setpoint
```

Each must have zero publishers; PX4 subscriptions are not controller conflicts. Manually
pilot Splash to engage the hook, let the ROV and box settle, and hold the actual Hook pose
to be reproduced. Keep automatic control stopped while recording:

```bash
cd /home/yecheng/bluerov_ws
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash

TARGET_CFG=/home/yecheng/bluerov_ws/src/bluerov2_control/experiments/payload_retrieval/config/hooked_box_target_pose_splash_$(date +%Y%m%d_%H%M%S).json
if ros2 run bluerov2_control record_mocap_target_pose \
  --topic /mocap/splash/pose \
  --message-type pose \
  --samples 160 \
  --timeout-sec 20 \
  --max-message-age-sec 0.20 \
  --output-file "$TARGET_CFG"
then
  python3 -m json.tool "$TARGET_CFG"
  echo "Selected Splash target: $TARGET_CFG"
else
  unset TARGET_CFG
  echo 'Target recording failed; correct the cause and record again.'
fi
```

Require `validation.passed: true` and `source_topic: /mocap/splash/pose`. The recorder requires
at least `0.75` s of accepted samples, per-axis position standard deviation at most `0.015` m,
orientation RMS dispersion at most `1.5` deg, and pairwise orientation spread at most `5` deg.
Failed recordings are not saved. Keep the printed absolute path; terminals do not share
`TARGET_CFG`. Do not overwrite an existing target or edit its source provenance.

Once recorded, neither the box nor its handle may move. A change to the target pose, marker
definition, rigid-body origin/axes, or mounting requires a new recording.

### Optional: use the unchanged-body rename manifest

The supplied manifest may be selected only if the same Splash rigid body was renamed from
`splash_fb` to `splash`, with marker geometry, origin, axes, mounting, and the physical Hook
target pose all unchanged. If any condition cannot be confirmed, record a new target above.

Inspect the manifest and its unchanged source:

```bash
TARGET_CFG=/home/yecheng/bluerov_ws/src/bluerov2_control/experiments/payload_retrieval/config/hooked_box_target_pose_splash_from_splash_fb_20260819_170318.json
python3 -m json.tool "$TARGET_CFG"
sha256sum /home/yecheng/bluerov_ws/src/bluerov2_control/experiments/payload_retrieval/config/hooked_box_target_pose_splash_20260819_170318.json
```

The source SHA-256 must be exactly:

```text
9237d8460432c0e897713057ac2d3d5c046c9f449b7389a5b35c647341367f3c
```

Do not edit or reformat the source. Launch checks its same-directory filename and SHA-256,
source/destination topics, exact pose snapshot, zero position offset, identity rotation,
and both operator-confirmation fields. Derived-target chains are prohibited. The bare
`splash_fb` source file is not a current direct recording, and the Glub-to-Splash manifest
marked `invalidated_for_control` must never be used.

## 6. Terminal 5: start rosbag and video recording

Start recording before moving from the manually hooked pose:

```bash
cd /home/yecheng/bluerov_ws
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash

BAG_DIR=/home/yecheng/bluerov_ws/fixed_hook_bags/splash_fixed_hook_$(date +%Y%m%d_%H%M%S)
mkdir -p /home/yecheng/bluerov_ws/fixed_hook_bags
echo "Bag directory: $BAG_DIR"
ACK_TOPIC=$(python3 -c "from px4_msgs.msg import VehicleCommandAck as M; v=int(M.MESSAGE_VERSION); print('/splash/fmu/out/vehicle_command_ack'+(f'_v{v}' if v else ''))")
STATUS_TOPIC=$(python3 -c "from px4_msgs.msg import VehicleStatus as M; v=int(M.MESSAGE_VERSION); print('/splash/fmu/out/vehicle_status'+(f'_v{v}' if v else ''))")

ros2 bag record -o "$BAG_DIR" --topics \
  /mocap/splash/pose \
  /mocap/splash/odom_ekf_fixed_hook \
  /mocap/splash/vehicle_odometry_fixed_hook \
  /splash/fmu/in/offboard_control_mode \
  /splash/fmu/in/vehicle_command \
  /splash/fmu/in/vehicle_thrust_setpoint \
  /splash/fmu/in/vehicle_torque_setpoint \
  "$ACK_TOPIC" \
  /splash/fmu/out/vehicle_control_mode \
  /splash/fmu/out/vehicle_odometry \
  "$STATUS_TOPIC" \
  /splash/fmu/out/failsafe_flags \
  /bluerov2/fixed_hook/offboard_request_enable \
  /bluerov2/fixed_hook/mission_enable \
  /bluerov2/fixed_hook/controller_heartbeat \
  /bluerov2/trial_event \
  /rosout \
  /parameter_events
```

Keep this terminal running. Some topics appear only after launch. Record external video
showing the ROV, box, approach direction, manual target recording, QGC state, forward motion,
hook confirmation, retreat, and stop. The launch does not start a camera recorder.

Manually disengage without moving the box, move to a clear starting position within the
operating bounds, then DISARM in QGC. Wait for the ROV and tether to settle. If the box or
handle moved, return to Section 5 and record a new target.

## 7. Terminal 6: launch with motion disabled

Select the exact target you checked; do not automatically choose the newest file:

```bash
cd /home/yecheng/bluerov_ws
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash

read -r -p 'Absolute path of the validated Splash target JSON: ' TARGET_CFG
if [[ "$TARGET_CFG" = /* ]] && test -f "$TARGET_CFG"; then
  ros2 launch bluerov2_control fixed_hook_pose_validation_splash.launch.py \
    target_config:="$TARGET_CFG"
else
  echo 'No launch: select an existing target file using its absolute path.'
fi
```

If a non-default orientation correction or measured obstacle map is required, add the
reviewed launch arguments before running this block. Launch validates provenance, stability,
frames, pool boundaries, and message compatibility. It starts the EKF, odometry adapter,
NMPC, CSV logger, and guarded Offboard manager. Mission permission starts disabled, Offboard
requests require explicit permission, and `auto_arm` is always false.

The startup summary must show the selected target, NED Hook/pre-hook poses, operating bounds,
`recorded_hook` attitude reference, Splash's `10/5/5` degree pre-hook gates, and
`Fixed-hook Position-like translation`. Leave this terminal visible for phase and fault logs.

The logger prints a unique trial directory under
`/home/yecheng/bluerov_ws/bluerov2_payload_retrieval_trials/`. Note that exact directory.
It writes `samples.csv` at a nominal `20` Hz, `events.csv`, and `metadata.json`, including the selected
target/configuration and raw `raw_mocap_*` pose columns. This complements, not replaces,
the rosbag and video. Verify that files are being created and that recording storage is
available before enabling motion.

## 8. Terminal 7: static preflight

```bash
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash

ros2 node list | sort | grep -E 'fixed_hook|mocap_ekf|offboard_enable'
ros2 topic info -v /splash/fmu/in/offboard_control_mode
ros2 topic info -v /splash/fmu/in/vehicle_command
ros2 topic info -v /splash/fmu/in/vehicle_thrust_setpoint
ros2 topic info -v /splash/fmu/in/vehicle_torque_setpoint
timeout 10s ros2 topic hz /mocap/splash/odom_ekf_fixed_hook
timeout 10s ros2 topic hz /mocap/splash/vehicle_odometry_fixed_hook
timeout 10s ros2 topic hz /splash/fmu/out/vehicle_odometry
ros2 topic echo /mocap/splash/vehicle_odometry_fixed_hook --once --qos-profile sensor_data

ros2 param get /mpc_fixed_hook_pose_validation model_type
ros2 param get /mpc_fixed_hook_pose_validation robot_type
ros2 param get /mpc_fixed_hook_pose_validation require_mission_enable
ros2 param get /mpc_fixed_hook_pose_validation fixed_hook_line_position_mode
ros2 param get /mpc_fixed_hook_pose_validation prehook_attitude_reference_mode
ros2 param get /mpc_fixed_hook_pose_validation operating_bounds_enable
ros2 param get /mpc_fixed_hook_pose_validation goal_x
ros2 param get /mpc_fixed_hook_pose_validation goal_y
ros2 param get /mpc_fixed_hook_pose_validation goal_z
ros2 param get /mpc_fixed_hook_pose_validation pre_approach_x
ros2 param get /mpc_fixed_hook_pose_validation pre_approach_y
ros2 param get /mpc_fixed_hook_pose_validation pre_approach_z
ros2 service type /bluerov2/fixed_hook/confirm_hook
```

Require exactly one publisher on each PX4 control input: the Offboard manager owns heartbeat
and vehicle commands; `/mpc_fixed_hook_pose_validation` owns thrust and torque. Stop on
duplicates. The service must exist, the model must be `fossen_real` / `standard`, and mission
gating, operating bounds, and position mode must be enabled. Verify live valid odometry,
expected frames, the recorded attitude, and the printed target/pre-hook geometry against the
physical setup. Position and attitude use MoCap; body angular velocity comes from fresh PX4
`vehicle_odometry`, so both sources are required.

Confirm the ROV, Hook pose, and pre-hook point are inside the reduced bounds, the planned
route and final corridor are clear, and small manual tests agree with the coordinate axes.
Do not enable motion if the solver, logger, DDS feedback, or any safety check reports a fault.

## 9. Terminals 8 and 9: authorize Offboard

With QGC showing DISARMED, clear startup permission latches and monitor ACKs in Terminal 8:

```bash
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash
ros2 topic pub --once /bluerov2/fixed_hook/mission_enable std_msgs/msg/Bool '{data: false}'
ros2 topic pub --once /bluerov2/fixed_hook/offboard_request_enable std_msgs/msg/Bool '{data: false}'

ACK_TOPIC=$(python3 -c "from px4_msgs.msg import VehicleCommandAck as M; v=int(M.MESSAGE_VERSION); print('/splash/fmu/out/vehicle_command_ack'+(f'_v{v}' if v else ''))")
ros2 topic echo "$ACK_TOPIC" --qos-profile sensor_data
```

After preflight passes and the safety operator is ready, request Offboard in Terminal 9:

```bash
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash
ros2 topic pub --once /bluerov2/fixed_hook/offboard_request_enable std_msgs/msg/Bool '{data: true}'
ros2 topic echo /splash/fmu/out/vehicle_control_mode --once --qos-profile sensor_data
```

Wait for the matching `VEHICLE_CMD_DO_SET_MODE` ACK and actual Offboard feedback in both
QGC and `flag_control_offboard_enabled: true`. The manager retries until feedback confirms
the mode; an ACK alone is insufficient. Do not proceed on stale or absent feedback.

## 10. Terminal 10: Hook keyboard; Terminal 9: manual ARM and mission permission

Open a separate terminal for Hook confirmation and leave it in the foreground:

```bash
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash
ros2 run bluerov2_control confirm_hook_keyboard
```

Do not press `H` yet, and do not type it into the launch terminal.

ARM manually in QGC with personnel clear and the operator ready; the launch
never ARMs for you. In Terminal 9, recheck the actual state:

```bash
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash
ros2 topic echo /splash/fmu/out/vehicle_control_mode --once --qos-profile sensor_data
```

Proceed only with both `flag_armed: true` and `flag_control_offboard_enabled: true`.
Mission permission is still false: the ROV is not yet under a position-holding mission.
Enable motion only after the final safety check and confirmation that recording is active:

```bash
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash
ros2 run bluerov2_control mark_trial_event start --note 'Splash fixed-hook mission enabled'
ros2 topic pub --once /bluerov2/fixed_hook/mission_enable std_msgs/msg/Bool '{data: true}'
```

Monitor the physical ROV and launch logs through this sequence:

1. Travel to pre-hook and align to the full recorded Hook attitude. Position, depth, full
   attitude, body-X forward axis, and yaw must all meet the Section 1 limits for `1.0` s.
2. `GO_FORWARD`: follow a horizontal NED reference toward Hook at `0.09` m/s, keeping the
   recorded depth and attitude as references.
3. `WAIT_HOOK`: after actual final position/depth/attitude pass, hold the recorded Hook pose.
   There is no automatic retreat timer.
4. Visually confirm engagement, wait at least `0.25` s after entry into `WAIT_HOOK`, then
   press a fresh `H` once in Terminal 10. Accepted confirmation starts `GO_BACK` on the next
   healthy update. Retreat uses the original line at `0.09` m/s with the recorded attitude,
   then holds at pre-hook.

The forward/retreat reference advances monotonically with time at fixed NED depth; its
vertical velocity reference is zero. Ordinary lateral, depth, and yaw errors do not freeze
or rewind that reference. NMPC corrects errors while translating, with constant straight-line
velocity weighting. This describes the commanded reference, not a guarantee that the ROV
will move straight, maintain depth, or reach it. Stop for unsafe actual motion.

Pre-hook alignment has no default elapsed-time timeout (`prehook_attitude_alignment_timeout_s=0`).
It continues while health checks pass, with `1.5` position/depth exit hysteresis and ongoing
`2` Hz path-safety checks. Persistent failure is not permission to relax the gates. Check
the physical pose, target, and MoCap calibration before another attempt.

Arrival is latched on entering `WAIT_HOOK`; later pose drift does not return the mission to
`GO_FORWARD` or reapply the arrival gates to `H`. Continue watching the actual ROV. Confirmation
still requires a valid mission, fresh Armed/Offboard feedback and odometry, an available
solver, and fresh MPC commands. The keyboard prints acceptance/rejection and exits. If
rejected, inspect the reason and restore safe conditions before restarting the keyboard
and pressing a new `H`; early keypresses are not saved for later.

After observing completed retreat and a stable hold, mark the result:

```bash
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash
ros2 run bluerov2_control mark_trial_event success --note 'Splash hook confirmed; retreat completed'
```

## 11. Normal stop

Prepare for the loss of active thrust and supervise the buoyant ROV. DISARM in QGC first
and verify the thrusters have stopped. Then clear both permissions and mark the stop:

```bash
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash
ros2 topic pub --once /bluerov2/fixed_hook/mission_enable std_msgs/msg/Bool '{data: false}'
ros2 topic pub --once /bluerov2/fixed_hook/offboard_request_enable std_msgs/msg/Bool '{data: false}'
ros2 run bluerov2_control mark_trial_event stop --note 'Operator stopped Splash fixed-hook mission'
```

Leave Offboard in QGC and verify the disarmed state. `mission_enable=false` removes mission
control and commands zero wrench; it is not a hold command. `offboard_request_enable=false`
stops further mode requests but does not switch PX4 out of an already active mode.

Press `Ctrl-C` in the launch terminal, then in the rosbag terminal and wait for recording
to finish. Stop the keyboard/monitors and video recorder. Keep the exact target JSON,
any rename manifest and source JSON, bag directory, CSV trial directory, and video together
in the run record. Check that `samples.csv`, `events.csv`, and `metadata.json` exist and
that the bag contains messages from the required live topics. Do not identify a run solely
by whichever directory is newest.

## 12. Emergency stop and fault recovery

For unsafe motion, loss of tracking/control, or anyone entering the thruster area, use the
verified hardware stop or QGC DISARM immediately. Do not wait for ROS commands or a terminal
response. Stop the launch as soon as it is safe to do so. ROS permission messages are
additional protection, not a replacement for the physical/manual stop.

For `Mission enable revoked`, `controller-health latch`, stale MoCap/odom/PX4 feedback,
attitude jumps, or solver faults, DISARM first and verify stopped thrusters. Then clear the
permissions if the nodes are still running:

```bash
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash
ros2 topic pub --once /bluerov2/fixed_hook/mission_enable std_msgs/msg/Bool '{data: false}'
ros2 topic pub --once /bluerov2/fixed_hook/offboard_request_enable std_msgs/msg/Bool '{data: false}'
```

Do not wait for automatic motion recovery. The EKF rejects excessive position/attitude
innovations and body-Z tilt above `0.55` rad; repeated rejected samples do not reset it to
an untrusted pose. After at most `2.0` s of coasting it stops odometry, retaining its last
trusted anchor. Restored data must pass the checks again, and recovered data alone does
not restore mission permission. Zero output after a fault can allow the ROV to rise.

Identify and correct the cause while disarmed. Re-record the target if anything defining
the target or rigid body changed. Repeat live-data and static preflight checks, then use
the full sequence again: Offboard request true, verified Offboard, manual QGC ARM, verified
Armed/Offboard, and finally mission true. Do not restart by repeatedly sending true to
latched gates or bypassing safety parameters.
