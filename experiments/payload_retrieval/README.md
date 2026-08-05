# Underwater Autonomous Payload Retrieval Experiments: Data Collection and Thesis Analysis Plan

This directory supports the thesis topic:

`Controller Design for Autonomous Underwater Payload Retrieval with a Passive Tool and ROVs`

The goal is to make each simulation or pool experiment reproducible, and to
generate the tables, event timelines, and plots needed for the thesis
Results / Analysis chapters.

## Data Required for Thesis Results

Each complete trial should cover the task chain: payload detection, approach,
line threading / hooking, return to the docking location, and handoff to the
operator. Repeat each controller and each experimental condition multiple times.

Data that should be collected:

- Robot state: position, attitude quaternion, linear velocity, and angular
  velocity from PX4 odometry. In pool experiments with Qualisys / MoCap, also
  record MoCap odometry.
- Control inputs and outputs: `cmd_vel`, thrust setpoint, torque setpoint,
  offboard / armed / control mode, and attitude setpoints if a controller
  publishes them.
- Perception results: handle pose, detection confidence, and whether the
  detection is valid. RGB / depth raw images should preferably be saved in a
  separate rosbag for qualitative figures and failure-case analysis.
- Task geometry: payload pose, handle pose, dock pose, and tank bounds. In
  simulation these can come from Gazebo ground truth; in the real pool they can
  come from MoCap, calibration points, or manually measured metadata.
- Task events: `start`, `first_detection`, `approach_start`, `hook_attempt`,
  `hooked`, `return_start`, `docked`, `success`, or `failure`.
- Experiment metadata: controller name, parameter file, environment, payload
  mass / shape, hook version, camera calibration version, water / lighting
  conditions, operator, and notes.

These data support the following thesis metrics:

- Task success rate, total duration, and per-stage duration.
- Time to first detection, detection availability, and detection confidence
  statistics.
- Approach accuracy: minimum / final distance to the handle or payload.
- Return accuracy: final docking error or distance to the handoff position.
- Motion quality: path length, mean / RMS / maximum speed, and maximum angular
  velocity.
- Control cost: RMS, peak value, and time integral of normalized thrust /
  torque.
- Safety: minimum distance to tank bounds and number of out-of-bounds samples.

When writing the thesis, report the mean, standard deviation, and median for
each set of repeated experiments, and discuss failure cases separately.

## Directory Structure

- `config/trial_metadata_template.json`: metadata template to copy or edit
  before each experiment.
- `data/`: placeholder directory. By default, run logs are written to
  `/home/yecheng/bluerov_ws/bluerov2_payload_retrieval_trials`.
- `../../bluerov2_control/research/trial_data_logger.py`: ROS 2 CSV data
  collection node.
- `../../bluerov2_control/research/mark_trial_event.py`: command for manually
  marking task events.
- `../../bluerov2_control/research/analyze_trial.py`: offline analysis script.

## Running Data Collection

Build and source the workspace first:

```bash
cd ~/bluerov_ws
colcon build --packages-select bluerov2_control
source install/setup.bash
```

## Recording a Fixed Hook Target in the Pool

The current real-pool milestone is deliberately narrower than the complete
retrieval mission: manually hook the target, record that stable pose from raw
MoCap, then track a slow trajectory back to the recorded pose and hold it.

Use the complete guarded procedure and terminal commands in
[`FIXED_HOOK_POSE_VALIDATION_ZH.md`](FIXED_HOOK_POSE_VALIDATION_ZH.md).

The target must be recorded directly from `/mocap/glub_fb/pose`, not from EKF
odometry that can briefly coast through a raw MoCap dropout:

```bash
ros2 run bluerov2_control record_mocap_target_pose \
  --topic /mocap/glub_fb/pose \
  --message-type pose \
  --samples 160 \
  --timeout-sec 20 \
  --max-message-age-sec 0.20 \
  --output-file /absolute/path/to/a/new_hooked_target.json
```

Archived target JSON files that name `/mocap/glub/...` or
`/mocap/glub_4/...` retain their original provenance and must not be
relabelled. Record a new validated target after the rigid-body rename.

## Automatic Fixed-Hook MPC Test

The real experiment uses a fail-closed launch. The current `glub_fb` target,
NED/FRD frames, standard real-robot model, pool bounds, command limits, and
target MAV IDs are stored as launch defaults, so the validated pipeline starts
with one command:

```bash
ros2 launch bluerov2_control fixed_hook_pose_validation.launch.py
```

The guarded sequence is now pre-approach, straight final approach, a
continuous five-second hold at the recorded pose, then a straight retreat to
the pre-approach pose. MoCap remains the position/attitude source, while the
MPC body-rate state is overridden by the low-latency BODY_FRD angular velocity
from `/glub/fmu/out/vehicle_odometry`. The adapter withholds controller
odometry if that PX4 rate is missing or stale.

`fixed_hook_mpc_june23.launch.py` remains only as a compatibility wrapper for
this guarded launch. Do not run `stabilized_control_real.launch.py` or another
Offboard/controller launch at the same time: they publish to the same heartbeat,
thrust, and torque topics. Starting the fixed-hook launch does not arm the
vehicle, request Offboard, or permit MPC motion. Those are separate operator
actions documented in the Chinese procedure.

Standard `/itrl_rov_1` simulation experiment:

```bash
ros2 launch bluerov2_control payload_retrieval_data_collection.launch.py \
  controller_name:=stabilized_control \
  environment:=gazebo_tank \
  metadata_file:=/home/yecheng/bluerov_ws/src/bluerov2_control/experiments/payload_retrieval/config/trial_metadata_template.json
```

The generic data-collection launch below is retained for legacy experiments;
it is not the current `/glub` fixed-hook control path and must not be run in
parallel with the guarded launch:

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

Mark key events from another terminal:

```bash
ros2 run bluerov2_control mark_trial_event start --note "mission started"
ros2 run bluerov2_control mark_trial_event first_detection
ros2 run bluerov2_control mark_trial_event hook_attempt
ros2 run bluerov2_control mark_trial_event hooked
ros2 run bluerov2_control mark_trial_event docked
ros2 run bluerov2_control mark_trial_event success
```

Each run creates a trial directory containing:

- `metadata.json`
- `samples.csv`
- `events.csv`

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
