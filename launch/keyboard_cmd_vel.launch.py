from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="bluerov2_control",
            executable="keyboard_cmd_vel",
            name="keyboard_cmd_vel",
            output="screen",
            emulate_tty=True,
            parameters=[{
                "cmd_vel_topic": "/itrl_rov_1/cmd_vel",
                "loop_hz": 30.0,
                "hold_timeout": 0.25,
                "linear_scale": 0.25,
                "angular_scale": 0.50,
                "linear_step": 0.05,
                "angular_step": 0.10,
                "max_linear_scale": 0.80,
                "max_angular_scale": 1.50,
                "debug_keys": False,  # Set to True to see detailed key logs
                "slew_rate": 5.0,
            }],
        ),
    ])
