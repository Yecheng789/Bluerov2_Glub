from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'bluerov2_mpc'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    include_package_data=True,
    package_data={
        # optional: also ship inside site-packages for importlib.resources fallback
        'bluerov2_mpc': ['models/weights/*.npz'],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/mpc.launch.py']),
        ('share/' + package_name + '/launch', ['launch/mpc_wrench.launch.py']),
        ('share/' + package_name + '/launch', ['launch/single_int_mpc_wrench.launch.py']),
        ('share/' + package_name + '/launch', ['launch/cascaded_pid_wrench.launch.py']),
        ('share/' + package_name + '/launch', ['launch/pid_only.launch.py']),
        # --- install weights into share/<pkg>/models/weights ---
        ('share/' + package_name + '/models/weights',
            glob('bluerov2_mpc/models/weights/*.npz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Victor Nan Fernandez-Ayala',
    maintainer_email='viktornfa@gmail.com',
    description='BlueROV2 MPC (Fossen/DI/Koopman) with mocap tracking in tank.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'mpc_node = bluerov2_mpc.mpc_node:main',
            'mpc_wrench_node = bluerov2_mpc.mpc_wrench_node:main',
            'single_int_mpc_node = bluerov2_mpc.single_int_mpc_node:main',
            'vel_pid_thrust_node = bluerov2_mpc.vel_pid_thrust_node:main',
            'pose_vel_pid_node = bluerov2_mpc.pose_vel_pid_node:main',
            'offboard_heartbeat = bluerov2_mpc.offboard_heartbeat:main',
            'offboard_heartbeat_wrench = bluerov2_mpc.offboard_heartbeat_wrench:main',
            'soft_dive_pid = bluerov2_mpc.soft_dive_pid:main',
        ],
    },
)