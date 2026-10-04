# BlueROV2 Control

ROS 2 controllers and experiment tools for underwater payload retrieval with
a BlueROV2 and a passive hook. The real-robot system combines motion-capture
state estimation, A* path planning, and Fossen-model nonlinear model predictive
control (NMPC).

The vehicle approaches a recorded target pose, follows a straight engagement
path, holds for operator confirmation, and retreats along the same line.
Camera-derived targets, automatic engagement detection, docking, and
heavy-payload lifting are outside the evaluated scope.

## Contents

- [Features and architecture](#features-and-architecture)
- [Requirements](#requirements)
- [Build and environment](#build-and-environment)
- [Real-robot usage](#real-robot-usage)
- [Simulation and other controllers](#simulation-and-other-controllers)
- [Data and paper figures](#data-and-paper-figures)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Project structure](#project-structure)
- [Contributing and support](#contributing-and-support)
- [License](#license)

## Features and architecture

- Target-pose recording from raw MoCap with frame and provenance validation.
- MoCap/EKF state estimation with PX4 body-rate feedback.
- Background A* planning with path simplification, smoothing, and timed
  position/attitude references for the approach to pre-hook.
- Fossen-model NMPC implemented in acados, with bounded force and moment
  commands, state-validity checks, and operator permission gates.
- Robot-specific Glub and Splash profiles, CSV logging, and keyboard-based
  hook confirmation.

```text
PLAN_TO_PREHOOK -> TRACK_TO_PREHOOK -> PREHOOK_REACHED
  -> GO_FORWARD -> WAIT_HOOK -- operator H --> GO_BACK -> COMPLETE
```

| Setting | Fixed-target real-robot configuration |
| --- | --- |
| NMPC solve rate | Nominally 25 Hz |
| Prediction interval / horizon | 0.04 s / 25 intervals (1.0 s) |
| Model integration | acados ERK, four stages, one step per interval |
| A* path monitor | Nominally 2 Hz; searches are requested as needed |
| Pre-hook offset | 0.50 m behind Hook along the recorded horizontal heading |
| Approach / final forward / retreat speed | 0.06 / 0.09 / 0.09 m/s |
| Hook confirmation | H key in a dedicated terminal; no automatic hold timeout |

A* is used only for the approach to pre-hook. Engagement and retreat follow
a time-parameterized straight line at constant NED depth, with the recorded
attitude as the reference. Tracking errors are corrected without reversing
reference progress; endpoint tolerances and safety checks remain active.
The table lists nominal settings, not measured timing or tracking guarantees.

## Requirements

The lab configuration uses:

- ROS 2 Jazzy, Python 3.12, colcon, and ROS message dependencies declared in
  [`package.xml`](package.xml).
- NumPy, CasADi, and a built acados installation with its Python interface,
  shared libraries, and code-generation tools available to the ROS Python
  interpreter.
- PX4 firmware with the appropriate UUV support and matching `px4_msgs`.
- Micro XRCE-DDS Agent, QGroundControl/manual takeover, and working MoCap
  for real-robot operation.
- Matplotlib and pandas for the offline paper-figure script, plus pytest
  for regression tests.

Install PX4, the required Gazebo worlds/models, and acados separately using
their upstream instructions. Keep the PX4 firmware and ROS message definitions
matched; an import error is not a reason to switch message versions.

## Build and environment

The commands below assume ROS and acados are installed and the workspace is
located at `/home/yecheng/bluerov_ws`. For another installation, update the
path-dependent launch and environment settings before proceeding.

```bash
cd /home/yecheng/bluerov_ws
source /opt/ros/jazzy/setup.bash

export ACADOS_SOURCE_DIR=/home/yecheng/acados
export LD_LIBRARY_PATH=/home/yecheng/acados/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# Build only after checking that src/px4_msgs matches the deployed firmware.
colcon build --packages-select px4_msgs --symlink-install
source install/local_setup.bash
colcon build --packages-select bluerov2_control --symlink-install

source src/bluerov2_control/scripts/source_fixed_hook_real.bash
```

The source script checks that this workspace's `px4_msgs` takes precedence.
The current real profile expects VehicleCommandAck version 0 and VehicleStatus
version 1. Do not source an older `px4_ws` overlay afterward. The script does
not configure the robot network or start Offboard/ARM.

## Real-robot usage

> **Read the full operator procedure before enabling thrusters.** Keep a
> trained operator and manual takeover available. Do not start multiple
> controllers or Offboard heartbeat/setpoint publishers for one vehicle.

| Robot | PX4 namespace | Raw MoCap topic | Launch | Operator guide |
| --- | --- | --- | --- | --- |
| Glub | `/glub` | `/mocap/glub/pose` | `fixed_hook_pose_validation.launch.py` | [Glub procedure](experiments/payload_retrieval/FIXED_HOOK_POSE_VALIDATION_ZH.md) |
| Splash | `/splash` | `/mocap/splash/pose` | `fixed_hook_pose_validation_splash.launch.py` | [Splash procedure](experiments/payload_retrieval/SPLASH_FIXED_HOOK_ZH.md) |

Both profiles require a validated target file supplied through `target_config`.
Follow the robot-specific guide to record or validate the target, start the
DDS agent and logging, and complete the preflight checks. Starting a launch
does not arm the vehicle, request Offboard, or grant permission to move.

Glub and Splash share permission-topic and node names, so run only one profile
at a time. Their MoCap target quaternions are not interchangeable, even though
the vehicle hardware matches. Splash accepts a direct recording or a validated,
hash-locked transfer from its former `splash_fb` name, provided the marker
geometry, body frame, mounting, and target pose are unchanged. The archived
Glub-to-Splash transfer is invalid and must not be used.

After the controller enters `WAIT_HOOK` and the operator visually confirms
engagement, use a separate terminal:

```bash
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash
ros2 run bluerov2_control confirm_hook_keyboard
```

Press `H` in this terminal, not in the launch terminal. Once the controller
enters `WAIT_HOOK`, pose drift alone does not invalidate confirmation. Mission
permission, Armed/Offboard feedback, state and command freshness, and solver
health must still be valid. The client sends one request, prints the result,
and exits. If the request is rejected, resolve the reported condition before
trying again.

### Safety and interpretation limits

- Pool dimensions alone do not determine the coordinate origin or axis signs.
  Verify NED/FRD conversion, measured pool bounds, target pose, and MAVLink IDs.
- With no configured internal obstacles, A* checks only the operating bounds.
  Dynamic obstacles, cables, and people are not sensed by the planner.
- Measured static obstacle rectangles must not intersect the intentional
  Hook contact corridor. Do not classify the contacted target as a no-contact
  obstacle without revising the task geometry.
- Planning failure does not fall back to an unchecked straight line.
- Pre-hook alignment has no default elapsed-time timeout, but health,
  operating-boundary, and operator safety gates remain active.
- Do not run `stabilized_control_real.launch.py` alongside this pipeline.

## Simulation and other controllers

Simulation uses `position_payload_retrieval`, an outer position P-D loop
coupled to PX4's UUV attitude controller. Despite its name,
`mpc_track_trajectory_acados_sim.launch.py` starts this position controller,
not the real-robot Fossen NMPC.

With the matching PX4 Gazebo setup and DDS agent already running, use a
separate simulation session:

```bash
source /opt/ros/jazzy/setup.bash
source /home/yecheng/bluerov_ws/install/local_setup.bash

# Default un-namespaced SITL topics: /fmu/...
ros2 launch bluerov2_control mpc_track_trajectory_acados_sim.launch.py

# Alternatively, for SITL started with PX4_UXRCE_DDS_NS=itrl_rov_1:
# ros2 launch bluerov2_control mpc_track_trajectory_acados_sim.launch.py \
#   px4_namespace:=itrl_rov_1
```

The launch supplies its own attitude-mode Offboard heartbeat and begins
tracking when the simulated vehicle reports Armed and Offboard. Do not start
another heartbeat or MPC executable in parallel. Because simulation and
hardware use different controllers, their results do not establish
same-controller simulation-to-reality transfer.

Additional launch files support stabilized control, PID position control,
and MPC position holding. `mpc_hold_position_acados_real.py` is a separate
holding controller; fixed-hook retrieval uses
`mpc_track_trajectory_acados.py`. Check the relevant launch configuration
before using these alternative controllers.

## Data and paper figures

The guarded real launch includes `payload_retrieval_data_logger`. Each trial
contains `samples.csv`, `events.csv`, and `metadata.json`; the guides also
provide rosbag commands. Read the
[logging and analysis guide](experiments/payload_retrieval/README.md) for
field semantics and experiment provenance.

[`scripts/plot_three_success_trials.py`](scripts/plot_three_success_trials.py)
reproduces the figures for the three selected trials when the workspace's
trial data is available. Pass `--output-dir` to avoid overwriting original
figures. The plots distinguish operator-confirmed physical retrieval from
controller completion: Direct-front lost MoCap/odometry before `COMPLETE`.
One selected trial per starting condition is not a repeated success-rate
study.

## Testing

After building and sourcing the environment, run the offline functional
tests from the workspace root:

```bash
cd /home/yecheng/bluerov_ws
source src/bluerov2_control/scripts/source_fixed_hook_real.bash
python3 -m pytest -q src/bluerov2_control/test \
  --ignore=src/bluerov2_control/test/test_copyright.py \
  --ignore=src/bluerov2_control/test/test_flake8.py \
  --ignore=src/bluerov2_control/test/test_pep257.py
```

This runs the functional tests; copyright and style checks are separate.
Coverage includes planning, frame conversion, mission gates, hook confirmation,
logging, and target validation. The recorded-baseline audit requires the
archived Direct-front dataset and is skipped when that data is unavailable.
Validate hardware behavior separately in supervised pool trials.

## Troubleshooting

- **Wrong PX4 message version:** use the checked environment script and the
  matching firmware/messages; do not stack incompatible overlays.
- **No motion:** inspect mission permission, Armed/Offboard feedback, and
  state-health gates before changing gains or limits.
- **Odometry stale or mode fallback:** inspect raw MoCap, EKF, PX4 body-rate
  feedback, and the Offboard failsafe configuration. Do not bypass freshness
  checks to force movement.
- **Pre-hook attitude not accepted:** verify the robot-specific target and
  frame transform. Use the profile's documented thresholds, not a quaternion
  copied from another rigid body.
- **Missing acados library or renderer:** check `ACADOS_SOURCE_DIR`,
  `LD_LIBRARY_PATH`, and the acados Python/code-generation installation.

## Project structure

| Path | Responsibility |
| --- | --- |
| `bluerov2_control/planner_astar.py` | Grid search, collision checking, simplification, and smoothing |
| `bluerov2_control/mpc_track_trajectory_acados.py` | Real NMPC, mission state machine, and replanning coordination |
| `bluerov2_control/models/fossen_bluerov2_model_real.py` | Real-vehicle Fossen dynamics |
| `bluerov2_control/mocap_ekf_odom.py` | MoCap state estimation |
| `bluerov2_control/nav_odom_to_vehicle_odometry.py` | Controller odometry adapter |
| `bluerov2_control/confirm_hook_keyboard.py` | One-shot operator confirmation |
| `bluerov2_control/research/` | Trial logging and analysis |
| `launch/` | Robot profiles and simulation/legacy entry points |
| `experiments/payload_retrieval/` | Operator guides, target recordings, and provenance |
| `scripts/` and `test/` | Environment setup, plotting, and regression tests |

## Contributing and support

Open a repository issue with a minimal reproduction, launch arguments,
software/firmware versions, and relevant logs. Do not include credentials or
private network information. Add regression tests when changing control,
frames, safety gates, or mission transitions. Preserve raw experimental
evidence and document changes to target geometry or calibration.

This package builds on KTH-DHSG BlueROV2 control work and the PX4, ROS 2,
CasADi, and acados projects. Maintainer details are in
[`package.xml`](package.xml); experiment-specific additions and evidence are
documented in the operator guides.

## License

[MIT](LICENSE). Preserve the original copyright and license notices.
External dependencies retain their own licenses.
