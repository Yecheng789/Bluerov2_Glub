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

After starting the MoCap EKF, hold the robot at the pose where the tool is
already hooked into the box handle. Then record an averaged target pose directly
from `/mocap/glub/odom_ekf`:

```bash
ros2 run bluerov2_control record_mocap_target_pose \
  --topic /mocap/glub/odom_ekf \
  --message-type odom \
  --samples 80 \
  --output-file /home/yecheng/bluerov_ws/src/bluerov2_control/experiments/payload_retrieval/config/hooked_box_target_pose_from_ekf.json
```

This generates a new target JSON file. For later fixed-target validation and
data logging, prefer this file recorded directly from the EKF topic instead of
manually copying values from a terminal screenshot.

## Automatic Fixed-Hook MPC Test

The current June 23 baseline is connected to MPC trajectory tracking. After
confirming that the MoCap software on the lab computer publishes
`/mocap/glub/pose` stably, calibrate and verify the corrected EKF first.

If markers were reattached, the Qualisys rigid body was redefined, or the
corrected EKF is still rejected because of `base_link_z_axis_angle`, keep the
robot physically level and still, then run:

```bash
ros2 run bluerov2_control calibrate_mocap_orientation_correction \
  --topic /mocap/glub/pose \
  --samples 120
```

The command prints a new `orientation_correction_quat_xyzw`. Use that value for
the EKF test below and for the fixed-hook launch.

Terminal A:

```bash
ros2 launch bluerov2_control mocap_ekf_odom.launch.py \
  rigid_body_name:=glub \
  orientation_correction_quat_xyzw:="0.04430086214711217 -0.001252015250325171 -5.551990616173286e-05 0.9990174487907484"
```

Terminal B:

```bash
ros2 topic hz /mocap/glub/pose
ros2 topic hz /mocap/glub/odom_ekf
ros2 topic hz /mocap/glub/vehicle_odometry_ekf
```

After the EKF is stable, stop the standalone EKF in Terminal A and start the
complete fixed-hook MPC launch:

```bash
ros2 launch bluerov2_control fixed_hook_mpc_june23.launch.py
```

This launch starts the MoCap EKF, the `nav_msgs/Odometry` to
`px4_msgs/VehicleOdometry` adapter, the offboard heartbeat, MPC trajectory
tracking, and data logging. It also sets the `/home/yecheng/acados` related
`ACADOS_SOURCE_DIR` and `LD_LIBRARY_PATH` for the MPC process. The MPC includes
an odometry timeout guard; if MoCap / EKF odometry is not updated for more than
0.30 s, it publishes zero thrust / torque.

By default, `fixed_hook_mpc_june23.launch.py` applies a fixed rotation
correction to the `glub` MoCap orientation and applies the same correction to
the orientation in older target JSON files. The current correction comes from a
2026-06-28 level-and-still calibration; after correction, the maximum
base-link z-axis angle is about 0.34 deg. If the rigid-body axes are redefined
in Qualisys, recalibrate or set `orientation_correction_quat_xyzw` back to an
empty string.

Keep low saturation limits for the first real pool test:

```bash
ros2 launch bluerov2_control fixed_hook_mpc_june23.launch.py \
  thrust_sat:=0.04 \
  torque_sat:=0.05
```

If the robot keeps spinning or MoCap warnings persist, stop the launch
immediately and do not increase the saturation limits yet.

If the real PX4 topics use `/itrl_rov_1` instead of `/glub`:

```bash
ros2 launch bluerov2_control fixed_hook_mpc_june23.launch.py \
  robot_namespace:=/itrl_rov_1
```

Standard `/itrl_rov_1` simulation experiment:

```bash
ros2 launch bluerov2_control payload_retrieval_data_collection.launch.py \
  controller_name:=stabilized_control \
  environment:=gazebo_tank \
  metadata_file:=/home/yecheng/bluerov_ws/src/bluerov2_control/experiments/payload_retrieval/config/trial_metadata_template.json
```

Pool experiment with MoCap recording:

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

If the experiment did not publish payload / dock poses, pass static target
points manually:

```bash
ros2 run bluerov2_control analyze_payload_retrieval_trial \
  /home/yecheng/bluerov_ws/bluerov2_payload_retrieval_trials/retrieval_YYYYMMDD_HHMMSS \
  --payload-target -1.0,-2.0,95.7 \
  --dock-target 0.0,0.0,95.7
```

For safety-boundary analysis, pass tank bounds:

```bash
ros2 run bluerov2_control analyze_payload_retrieval_trial \
  /home/yecheng/bluerov_ws/bluerov2_payload_retrieval_trials/retrieval_YYYYMMDD_HHMMSS \
  --tank-bounds -4.5,4.5,-2.5,2.5,94.2,97.2
```

Analysis results are written to the trial directory under `analysis/`:

- `summary_metrics.json`
- `summary_metrics.csv`
- `thesis_results_summary.md`
- If `matplotlib` is installed, plan-view, target-distance, and control plots
  are also generated.
