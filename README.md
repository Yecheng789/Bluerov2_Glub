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

# STILL IN PROGRESS
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

To make them permanent, you can add those two lines to ```~/.bashrc```. You will also probably have to install the ```t_renderer```, for that download the latest one for your system [here](https://github.com/acados/tera_renderer/releases/), rename it to just ```t_renderer``` and then inside your acados repository folder do:
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

## For 6DoF MPC Tracking controller: