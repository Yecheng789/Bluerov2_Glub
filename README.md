## For Stabilized Control:
For it to work remember to use the appropiate namespace (same as the real BlueROV2 in the tank):
```bash
PX4_UXRCE_DDS_NS=itrl_rov_1 make px4_sitl_uuv gz_uuv_bluerov2_heavy
```

You can also try it in the KTH tank environment with:
```bash
PX4_UXRCE_DDS_NS=itrl_rov_1 PX4_GZ_WORLD=kth_marinarium make px4_sitl_uuv gz_uuv_bluerov2_heavy
```

Remember to run the Micro-XRCE-DDS-Agent:
```bash
micro-xrce-dds-agent udp4 -p 8888
```

To launch the controller and the keyboard teleop, make sure you are in Offboard mode in QGC and:
```bash
cd ~/px4_ws
colcon build
source install/setup.bash


# 1) Launch the nodes (heartbeat + stabilized)
ros2 launch bluerov2_control stabilized_control.launch.py

# 2) In a second terminal run the custom keyboard teleop 
ros2 run bluerov2_control wasd_teleop
```

Also, make sure to arm the vehicle adter you run the controller node.

## For PID Position Control:

Same setup as Stabilized Control but run:
```bash
cd ~/px4_ws
colcon build
source install/setup.bash


# 1) Launch the nodes (heartbeat + PID)
ros2 launch bluerov2_control position_control_pid.launch.py

# 2) In a second terminal run the custom keyboard teleop 
ros2 run bluerov2_control wasd_teleop
```

## For 6DoF MPC Holding controller:

You will need to install casadi:
```bash
pip install casadi
```

And then run:
```bash
cd ~/px4_ws
colcon build
source install/setup.bash


# 1) Launch the nodes (heartbeat + MPC)
ros2 launch bluerov2_control mpc_hold_position.launch.py
```

## For 6DoF MPC Holding controller (acados):

Acados is a fast nonlinear optimization library designed for embedded applications. To use it first install it following the official documentation:
```bash
git clone https://github.com/acados/acados.git
cd acados
git submodule update --recursive --init

mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON ..
make install -j4

pip install -e ../interfaces/acados_template
```

Then you also need to add environment variables (if you have it in a folder inside ```$HOME``` make sure to point to it below):
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/acados/lib
export ACADOS_SOURCE_DIR=$HOME/acados
```

To make them permanent, you can add those two lines to ```~/.bashrc```. You will also probably have to install the ```t_renderer```, for that, download the latest one for your system [here](https://github.com/acados/tera_renderer/releases/), rename it to just ```t_renderer``` and then inside your acados repository folder do:
```bash
mkdir -p bin
```
and place the renderer file there, then give it executable permissions:
```bash
chmod +x bin/t_renderer
```
and verify:
```bash
ls -l bin/t_renderer
/bin/t_renderer --help
```

If that last command runs, the renderer part is fixed.

Then build and launch:
```bash
cd ~/px4_ws
colcon build
source install/setup.bash

ros2 launch bluerov2_control mpc_hold_position_acados.launch.py
```

## Gazebo fixed-hook retrieval with Position behavior

For Gazebo, use the simulation launch instead of running the wrench MPC
executable directly. The historical launch filename is kept for command
compatibility, but it now starts a single position-outer-loop controller. That
node publishes `VehicleAttitudeSetpoint` and an `attitude=True` Offboard
heartbeat; PX4's UUV attitude controller supplies the roll, pitch, and yaw
inner loop.

```bash
cd ~/bluerov_ws
colcon build --packages-select bluerov2_control
source install/setup.bash

# PX4 topics are /fmu/... (the default used by an un-namespaced SITL).
ros2 launch bluerov2_control mpc_track_trajectory_acados_sim.launch.py

# If SITL was started with PX4_UXRCE_DDS_NS=itrl_rov_1:
ros2 launch bluerov2_control mpc_track_trajectory_acados_sim.launch.py \
  px4_namespace:=itrl_rov_1
```

Start the launch before selecting Offboard in QGC, then arm the simulated
vehicle. The simulation launch intentionally begins tracking immediately after
PX4 confirms both `armed` and `offboard`.

The controller holds commanded roll and pitch at zero, moves horizontally to
the alignment point at `0.06 m/s` while slewing yaw at no more than
`0.10 rad/s`, and then descends vertically at `0.06 m/s`. Translation is active
during the yaw change;
the ROS position P-D loop rotates navigation-frame position correction into
body FRD thrust while PX4 independently stabilizes attitude. The hook's SDF
visual/collision pose stays at `z=-0.08`. `hook_alignment_raise_m=0.04` instead
subtracts 40 mm from the vehicle's NED-z targets for ALIGN, FORWARD_PASS, and
BACKWARD_PASS. At ALIGN_HOLD the vehicle then lowers another 10 mm and keeps
that engagement depth through FORWARD_PASS and BACKWARD_PASS.

This is the same outer-position/inner-attitude control structure as PX4 UUV
Position mode. It is implemented on the ROS side because this PX4 tree's
native `uuv_pos_control` consumes the custom uORB
`trajectory_setpoint6dof`, which is not exposed by the current DDS topic map.
The lower PX4 UUV attitude loop and allocator are still used unchanged.

Do not run `offboard_heartbeat_wrench`, `mpc_track_trajectory_acados`, or any
other Offboard heartbeat/setpoint publisher at the same time as this launch.
