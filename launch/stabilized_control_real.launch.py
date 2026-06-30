from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    cmd_vel_topic = LaunchConfiguration("cmd_vel_topic")
    torque_z_sign = ParameterValue(LaunchConfiguration("torque_z_sign"), value_type=float)
    thrust_scale = ParameterValue(LaunchConfiguration("thrust_scale"), value_type=float)
    yaw_torque_scale = ParameterValue(LaunchConfiguration("yaw_torque_scale"), value_type=float)
    thrust_min = ParameterValue(LaunchConfiguration("thrust_min"), value_type=float)
    torque_min = ParameterValue(LaunchConfiguration("torque_min"), value_type=float)
    thrust_sat = ParameterValue(LaunchConfiguration("thrust_sat"), value_type=float)
    torque_sat = ParameterValue(LaunchConfiguration("torque_sat"), value_type=float)
    cmd_timeout = ParameterValue(LaunchConfiguration("cmd_timeout"), value_type=float)

    heartbeat = Node(
        package="bluerov2_control",
        executable="offboard_heartbeat_wrench",
        name="offboard_heartbeat_wrench",
        output="screen",
        parameters=[
            {
                "topic": "/glub/fmu/in/offboard_control_mode"
            }
        ]
    )

    controller = Node(
        package="bluerov2_control",
        executable="stabilized_control_real",
        name="stabilized_control_real",
        output="screen",
        parameters=[
            {
                "control_mode_topic": "/glub/fmu/out/vehicle_control_mode",
                "thrust_sp_topic": "/glub/fmu/in/vehicle_thrust_setpoint",
                "torque_sp_topic": "/glub/fmu/in/vehicle_torque_setpoint",
                "cmd_vel_topic": cmd_vel_topic,
                "torque_z_sign": torque_z_sign,
                "thrust_scale": thrust_scale,
                "yaw_torque_scale": yaw_torque_scale,
                "thrust_min": thrust_min,
                "torque_min": torque_min,
                "thrust_sat": thrust_sat,
                "torque_sat": torque_sat,
                "require_control_mode_gate": False,
                "cmd_timeout": cmd_timeout,
            }
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument("cmd_vel_topic", default_value="/itrl_rov_1/cmd_vel"),
        DeclareLaunchArgument("torque_z_sign", default_value="1.0"),
        DeclareLaunchArgument("thrust_scale", default_value="0.45"),
        DeclareLaunchArgument("yaw_torque_scale", default_value="0.45"),
        DeclareLaunchArgument("thrust_min", default_value="0.18"),
        DeclareLaunchArgument("torque_min", default_value="0.18"),
        DeclareLaunchArgument("thrust_sat", default_value="0.60"),
        DeclareLaunchArgument("torque_sat", default_value="0.60"),
        DeclareLaunchArgument("cmd_timeout", default_value="0.5"),
        heartbeat,
        controller,
    ])
