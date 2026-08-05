#!/usr/bin/env python3
"""
Gazebo payload retrieval using a position outer loop and PX4 attitude loop.

The historical filename is retained so existing simulation commands keep
working.  This launch intentionally does not start the wrench MPC or a second
heartbeat publisher.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


CONTROL_MODE_TIMEOUT_SEC = 1.25


def _px4_topic(namespace, suffix):
    namespace = str(namespace).strip().strip('/')
    prefix = f'/{namespace}' if namespace else ''
    return f'{prefix}/fmu/{suffix}'


def _launch_setup(context):
    px4_namespace = LaunchConfiguration('px4_namespace').perform(context)

    return [
        Node(
            package='bluerov2_control',
            executable='position_payload_retrieval',
            name='position_payload_retrieval',
            output='screen',
            parameters=[{
                'odom_topic': _px4_topic(
                    px4_namespace,
                    'out/vehicle_odometry',
                ),
                'control_mode_topic': _px4_topic(
                    px4_namespace,
                    'out/vehicle_control_mode',
                ),
                # VehicleAttitudeSetpoint is MESSAGE_VERSION=1 in this
                # workspace, so PX4 appends the required _v1 suffix.
                'attitude_setpoint_topic': _px4_topic(
                    px4_namespace,
                    'in/vehicle_attitude_setpoint_v1',
                ),
                'offboard_control_mode_topic': _px4_topic(
                    px4_namespace,
                    'in/offboard_control_mode',
                ),
                'rates_setpoint_topic': _px4_topic(
                    px4_namespace,
                    'in/vehicle_rates_setpoint',
                ),
                'require_mission_enable': False,
                'control_mode_timeout_s': CONTROL_MODE_TIMEOUT_SEC,
                # Physical hook stays at its SDF pose.  NED +z is down, so
                # the controller subtracts this value from ALIGN/PASS/BACK.
                'hook_alignment_raise_m': 0.04,
                'align_hold_lower_m': 0.01,
                'align_speed_mps': 0.06,
                'descent_speed_mps': 0.06,
                'forward_pass_speed_mps': 0.05,
                'backward_pass_speed_mps': 0.06,
                'yaw_reference_rate_rad_s': 0.10,
                # Stay below PX4's default UUV_THRUST_SAT=0.1.
                'thrust_limit_xy': 0.08,
                'thrust_limit_z': 0.08,
            }],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'px4_namespace',
            default_value='',
            description=(
                'PX4 DDS namespace without /fmu; leave empty for /fmu topics '
                'or use itrl_rov_1 for /itrl_rov_1/fmu topics.'
            ),
        ),
        OpaqueFunction(function=_launch_setup),
    ])
