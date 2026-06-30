from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="bluerov2_control",
            executable="offboard_enable",
            name="offboard_enable_real",
            output="screen",
            emulate_tty=True,
            parameters=[{
                "offboard_mode_topic": "/glub/fmu/in/offboard_control_mode",
                "vehicle_cmd_topic": "/glub/fmu/in/vehicle_command",
                "auto_arm": False,
            }],
        ),
    ])
