# Fixed-Target Payload Retrieval Experiments

This directory contains procedures, metadata templates, and analysis tools for
fixed-target payload-retrieval experiments. The current pool experiment uses
a recorded hook pose, an A*/NMPC approach, straight-line engagement, an
operator-confirmed hold, and straight-line retreat.

The experiments support the thesis *Controller Design for Autonomous Underwater
Payload Retrieval with a Passive Tool and ROVs*. The aim is to produce
reproducible trials, event timelines, and quantitative results. Perception,
autonomous hook detection, return-to-home, and docking are outside the current
fixed-target experiment.

## Experiment Workflow

Follow the complete [Glub procedure](FIXED_HOOK_POSE_VALIDATION_ZH.md) or
[Splash procedure](SPLASH_FIXED_HOOK_ZH.md) for preflight checks, launch
commands, operator permissions, and shutdown. The sequence is:

1. Manually engage the hook and record a stable raw MoCap target pose.
2. Use A* and NMPC to reach the pre-approach point behind the target.
3. After the robot-specific position, depth, and attitude gates pass, advance
   along a horizontal straight line to the recorded pose.
4. Hold in `WAIT_HOOK` until the operator visually confirms engagement and
   presses `H`.
5. Retreat along the same line to the pre-approach point and save the trial.

Build and source the workspace before running the data-collection tools. For
real-robot builds, also follow the message-version and environment checks in
the relevant operator guide.

```bash
cd ~/bluerov_ws
colcon build --packages-select bluerov2_control
source install/setup.bash
```

## Target Validation

Record Glub targets directly from `/mocap/glub/pose`, not from EKF odometry,
which can briefly continue through a raw MoCap dropout:

```bash
ros2 run bluerov2_control record_mocap_target_pose \
  --topic /mocap/glub/pose \
  --message-type pose \
  --samples 160 \
  --timeout-sec 20 \
  --max-message-age-sec 0.20 \
  --output-file /absolute/path/to/a/new_hooked_target.json
```

Do not relabel files recorded under `/mocap/glub_fb/...`,
`/mocap/glub_4/...`, or an earlier rigid-body definition. Keep their original
provenance and record a new validated target from the current topic.

Splash requires its own validated `/mocap/splash/pose` recording. Matching
MAVLink IDs, dynamics presets, and nominal body-frame conventions do not make
Glub's target quaternion valid for Splash. The only permitted derived target
is the hash-locked
`hooked_box_target_pose_splash_from_splash_fb_20260819_170318.json`
manifest for the operator-confirmed rename of the same Splash rigid body.
It is invalid if the marker definition, rigid-body origin or axes, mounting,
or fixed hook pose changes; record a new target in that case.

Never use
`config/hooked_box_target_pose_splash_from_glub_20260802_195146.json`
as an experiment input. It is an invalidated cross-robot transfer record.
See the [Splash guide](SPLASH_FIXED_HOOK_ZH.md) for target validation and the
robot-specific attitude gates.

## Running a Fixed-Hook Trial

The guarded real-robot launches use NED/FRD frames, the standard robot model,
pool bounds, command limits, and target MAVLink IDs as defaults. Verify these
against the physical setup before each deployment. Neither robot has a default
target file. For Glub, set `TARGET_CFG` to a newly validated
`/mocap/glub/pose` recording and pass it explicitly:

```bash
ros2 launch bluerov2_control fixed_hook_pose_validation.launch.py \
  target_config:="$TARGET_CFG"
```

Use the dedicated launch in the [Splash guide](SPLASH_FIXED_HOOK_ZH.md) for
Splash. Starting a launch does not arm the vehicle, request Offboard, or
permit MPC motion; these are separate operator actions.

During `WAIT_HOOK`, visually verify that the hook has engaged, then press `H`
in a separate terminal running:

```bash
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash
ros2 run bluerov2_control confirm_hook_keyboard
```

`WAIT_HOOK` has no automatic timeout. Entry latches that the recorded hook-pose
tolerances passed; later pose drift neither returns to `GO_FORWARD` nor
invalidates confirmation. Wait at least `0.25 s` after the `WAIT_HOOK` entry
log before pressing `H`. Confirmation starts `GO_BACK` only while mission,
Armed/Offboard, odometry, solver, state, and command freshness remain healthy.
The client calls `/bluerov2/fixed_hook/confirm_hook` once and exits. If the
request is rejected, resolve the reported cause and run the client again.
No timer starts retreat.

Run only one controller stack at a time. Do not combine this launch with
`stabilized_control_real.launch.py`, another Offboard controller, or the other
robot's fixed-hook launch: they share command topics or operator permissions.
`fixed_hook_mpc_june23.launch.py` is a compatibility wrapper for the guarded
Glub launch.

## Data Logging

Use the logging instructions in the robot's operator guide for guarded
fixed-hook trials. The generic launches below are for simulation and legacy
experiments, not the current `/glub` fixed-hook control path. Do not run them
alongside the guarded launch.

For a standard `/itrl_rov_1` simulation experiment:

```bash
ros2 launch bluerov2_control payload_retrieval_data_collection.launch.py \
  controller_name:=stabilized_control \
  environment:=gazebo_tank \
  metadata_file:=/home/yecheng/bluerov_ws/src/bluerov2_control/experiments/payload_retrieval/config/trial_metadata_template.json
```

For a legacy pool experiment:

```bash
ros2 launch bluerov2_control payload_retrieval_data_collection.launch.py \
  controller_name:=mpc_track_trajectory_acados \
  environment:=kth_pool \
  mocap_odom_topic:=/mocap/itrl_rov_1/odom
```

If the current topic names are different, override them at launch:

```bash
ros2 launch bluerov2_control payload_retrieval_data_collection.launch.py \
  odom_topic:=/fmu/out/vehicle_odometry \
  thrust_sp_topic:=/fmu/in/vehicle_thrust_setpoint \
  torque_sp_topic:=/fmu/in/vehicle_torque_setpoint \
  control_mode_topic:=/fmu/out/vehicle_control_mode
```

Mark events from another terminal as they occur. The commands below show the
broader retrieval event vocabulary; record only stages actually performed,
and mark failures as well as successful outcomes.

```bash
ros2 run bluerov2_control mark_trial_event start --note "mission started"
ros2 run bluerov2_control mark_trial_event first_detection
ros2 run bluerov2_control mark_trial_event hook_attempt
ros2 run bluerov2_control mark_trial_event hooked
ros2 run bluerov2_control mark_trial_event docked
ros2 run bluerov2_control mark_trial_event success
```

Logs are written by default to
`/home/yecheng/bluerov_ws/bluerov2_payload_retrieval_trials`, not the local
`data/` placeholder. Each trial directory contains:

- `metadata.json`
- `samples.csv`
- `events.csv`

Use [the metadata template](config/trial_metadata_template.json) to record the
controller, parameter file, environment, payload mass and shape, hook version,
calibration, lighting and water conditions, operator, and notes. Collect the
following data where applicable:

- Robot state: PX4 position, attitude, linear velocity, and angular velocity;
  also record MoCap odometry in pool trials.
- Control: `cmd_vel`, thrust and torque setpoints, armed/Offboard/control mode,
  and attitude setpoints if published.
- Geometry: payload, handle, and dock poses, plus tank bounds. Use simulation
  ground truth or measured real-pool geometry, as appropriate.
- Events: stage transitions, hook attempts, operator-confirmed engagement,
  success, and failure, with notes that define the trial's outcome.
- Perception, when tested: handle pose, confidence, and validity. Save raw RGB
  and depth images in a separate rosbag for qualitative and failure analysis.

The tools live in
[`trial_data_logger.py`](../../bluerov2_control/research/trial_data_logger.py),
[`mark_trial_event.py`](../../bluerov2_control/research/mark_trial_event.py), and
[`analyze_trial.py`](../../bluerov2_control/research/analyze_trial.py).

## Offline Analysis

Analyze one trial:

```bash
ros2 run bluerov2_control analyze_payload_retrieval_trial \
  /home/yecheng/bluerov_ws/bluerov2_payload_retrieval_trials/retrieval_YYYYMMDD_HHMMSS
```

If a simulation experiment did not publish payload / dock poses, pass static
target points manually. The coordinates below are simulation examples and are
not measurements of the 9 x 5 x 3 m real pool:

```bash
ros2 run bluerov2_control analyze_payload_retrieval_trial \
  /home/yecheng/bluerov_ws/bluerov2_payload_retrieval_trials/retrieval_YYYYMMDD_HHMMSS \
  --payload-target=-1.0,-2.0,95.7 \
  --dock-target=0.0,0.0,95.7
```

For simulation safety-boundary analysis, pass the matching simulated bounds.
Never reuse this example as real-pool bounds:

```bash
ros2 run bluerov2_control analyze_payload_retrieval_trial \
  /home/yecheng/bluerov_ws/bluerov2_payload_retrieval_trials/retrieval_YYYYMMDD_HHMMSS \
  --tank-bounds=-4.5,4.5,-2.5,2.5,94.2,97.2
```

Analysis results are written to the trial directory under `analysis/`:

- `summary_metrics.json`
- `summary_metrics.csv`
- `thesis_results_summary.md`
- If `matplotlib` is installed, plan-view, target-distance, and control plots
  are also generated.

For repeated trials, compare task and stage duration, approach error, path
length, speed, angular velocity, normalized thrust/torque cost, and distance
to tank bounds. Include detection or docking metrics only when those stages
were tested and the required data were recorded.

Report the number of trials, the success criterion, and all failed or aborted
runs. Summarize repeated measurements with mean, standard deviation, and
median, and discuss failure cases separately. Reaching a pose or receiving
operator confirmation alone does not establish an end-to-end retrieval
success rate.

## Scope and Safety Limits

- The current experiment validates motion relative to a fixed recorded pose.
  It does not detect the payload online, verify hook engagement automatically,
  return home, or dock. The full detection-to-handoff mission remains a
  broader research objective.
- Pre-hook attitude alignment has no timeout by default. The robot waits for
  the configured gates while path-safety and fail-closed checks remain active.
  Splash's pre-hook gates differ from Glub's; both retain the recorded hook
  quaternion as the reference and a `5 deg` final hook attitude gate.
- In the current straight-line mode, ordinary yaw, lateral, or depth errors
  do not freeze or rewind the reference. Endpoint completion still requires
  measured position, depth, and attitude tolerances. Command limits, operating
  bounds, state continuity, and failure checks remain active.
- MoCap supplies position and attitude; PX4 supplies the BODY_FRD angular
  velocity used by MPC. Missing or stale PX4 rates prevent the adapter from
  publishing controller odometry.
- Simulation coordinates and bounds are not real-pool measurements. Verify
  frames, target provenance, bounds, message versions, and vehicle IDs using
  the [Glub](FIXED_HOOK_POSE_VALIDATION_ZH.md) or
  [Splash](SPLASH_FIXED_HOOK_ZH.md) procedure before operating the robot.
