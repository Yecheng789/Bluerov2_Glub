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
            'position_control_pid = bluerov2_control.position_control_pid:main',
        ],
    },
)