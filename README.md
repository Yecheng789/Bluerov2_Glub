## For Position Control PID:
```bash
sudo apt update
sudo apt install ros-humble-teleop-twist-keyboard


cd ~/px4_ws
colcon build
source install/setup.bash


# 1) Launch the nodes (heartbeat + PID)
ros2 launch bluerov2_control position_control_pid.launch.py

# 2) In a second terminal: run keyboard teleop publishing to your namespaced topic
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=/itrl_rov_1/cmd_vel
```


## For Stabilized Control:
```bash
cd ~/px4_ws
colcon build
source install/setup.bash


# 1) Launch the nodes (heartbeat + PID)
ros2 launch bluerov2_control stabilized_control.launch.py

# 2) In a second terminal: run keyboard teleop publishing to your namespaced topic
ros2 run bluerov2_control wasd_teleop
```