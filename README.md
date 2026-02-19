## For Stabilized Control:
For it to work remember to use the appropiate namespace:
```bas
PX4_UXRCE_DDS_NS=itrl_rov_1 make px4_sitl_uuv gz_uuv_bluerov2_heavy
```

```bash
cd ~/px4_ws
colcon build
source install/setup.bash


# 1) Launch the nodes (heartbeat + PID)
ros2 launch bluerov2_control stabilized_control.launch.py

# 2) In a second terminal run the custom keyboard teleop 
ros2 run bluerov2_control wasd_teleop
```
