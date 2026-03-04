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

Same setup as stabilized Control but run:
```bash
cd ~/px4_ws
colcon build
source install/setup.bash


# 1) Launch the nodes (heartbeat + PID)
ros2 launch bluerov2_control position_control_pid.launch.py

# 2) In a second terminal run the custom keyboard teleop 
ros2 run bluerov2_control wasd_teleop
```

## For MPC Holding controller:

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
