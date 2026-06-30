from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'bluerov2_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    include_package_data=True,
    package_data={
        'bluerov2_control': ['models/weights/*.npz'],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/position_control_pid.launch.py']),
        ('share/' + package_name + '/launch', ['launch/stabilized_control.launch.py']),
        ('share/' + package_name + '/launch', ['launch/mpc_hold_position.launch.py']),
        ('share/' + package_name + '/launch', ['launch/mpc_hold_position_acados.launch.py']),
        ('share/' + package_name + '/launch', ['launch/mpc_hold_position_acados_real.launch.py']),
        ('share/' + package_name + '/launch', ['launch/offboard_enable_real.launch.py']),
        ('share/' + package_name + '/launch', ['launch/stabilized_control_real.launch.py']),
        ('share/' + package_name + '/launch', ['launch/payload_retrieval_data_collection.launch.py']),
        ('share/' + package_name + '/launch', ['launch/mocap_ekf_odom.launch.py']),
        ('share/' + package_name + '/launch', ['launch/fixed_hook_mpc_june23.launch.py']),
        # --- install weights into share/<pkg>/models/weights ---
        ('share/' + package_name + '/models/weights',
            glob('bluerov2_control/models/weights/*.npz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Victor Nan Fernandez-Ayala',
    maintainer_email='viktornfa@gmail.com',
    description='BlueROV2 controllers with mocap tracking in tank or Gazebo simulator.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'offboard_heartbeat_actuator = bluerov2_control.offboard_heartbeat_actuator:main',
            'offboard_heartbeat_wrench = bluerov2_control.offboard_heartbeat_wrench:main',
            'offboard_heartbeat_attitude = bluerov2_control.offboard_heartbeat_attitude:main',
            'position_control_pid = bluerov2_control.position_control_pid:main',
            'stabilized_control = bluerov2_control.stabilized_control:main',
            'stabilized_control_real = bluerov2_control.stabilized_control_real:main',
            'real_wrench_watchdog = bluerov2_control.real_wrench_watchdog:main',
            'mpc_hold_position = bluerov2_control.mpc_hold_position:main',
            'mpc_hold_position_acados = bluerov2_control.mpc_hold_position_acados:main',
            'mpc_hold_position_acados_real = bluerov2_control.mpc_hold_position_acados_real:main',
            'mpc_track_trajectory_acados = bluerov2_control.mpc_track_trajectory_acados:main',
            'payload_retrieval_data_logger = bluerov2_control.research.trial_data_logger:main',
            'mark_trial_event = bluerov2_control.research.mark_trial_event:main',
            'analyze_payload_retrieval_trial = bluerov2_control.research.analyze_trial:main',
            'mocap_ekf_odom = bluerov2_control.mocap_ekf_odom:main',
            'calibrate_mocap_orientation_correction = bluerov2_control.calibrate_mocap_orientation_correction:main',
            'nav_odom_to_vehicle_odometry = bluerov2_control.nav_odom_to_vehicle_odometry:main',
            'record_mocap_target_pose = bluerov2_control.record_mocap_target_pose:main',
            'offboard_enable = bluerov2_control.offboard_enable:main',
            'wasd_teleop = bluerov2_control.wasd_teleop:main',
            "keyboard_cmd_vel = bluerov2_control.keyboard_cmd_vel:main",
        ],
    },
)
