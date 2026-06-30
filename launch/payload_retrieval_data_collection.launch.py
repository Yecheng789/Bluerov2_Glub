from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    sample_period_s = ParameterValue(LaunchConfiguration("sample_period_s"), value_type=float)

    logger = Node(
        package="bluerov2_control",
        executable="payload_retrieval_data_logger",
        name="payload_retrieval_data_logger",
        output="screen",
        parameters=[
            {
                "trial_id": LaunchConfiguration("trial_id"),
                "output_dir": LaunchConfiguration("output_dir"),
                "sample_period_s": sample_period_s,
                "metadata_file": LaunchConfiguration("metadata_file"),
                "controller_name": LaunchConfiguration("controller_name"),
                "environment": LaunchConfiguration("environment"),
                "operator": LaunchConfiguration("operator"),
                "notes": LaunchConfiguration("notes"),
                "odom_topic": LaunchConfiguration("odom_topic"),
                "mocap_odom_topic": LaunchConfiguration("mocap_odom_topic"),
                "cmd_vel_topic": LaunchConfiguration("cmd_vel_topic"),
                "thrust_sp_topic": LaunchConfiguration("thrust_sp_topic"),
                "torque_sp_topic": LaunchConfiguration("torque_sp_topic"),
                "attitude_sp_topic": LaunchConfiguration("attitude_sp_topic"),
                "control_mode_topic": LaunchConfiguration("control_mode_topic"),
                "handle_pose_topic": LaunchConfiguration("handle_pose_topic"),
                "handle_confidence_topic": LaunchConfiguration("handle_confidence_topic"),
                "handle_detected_topic": LaunchConfiguration("handle_detected_topic"),
                "payload_pose_topic": LaunchConfiguration("payload_pose_topic"),
                "dock_pose_topic": LaunchConfiguration("dock_pose_topic"),
                "task_event_topic": LaunchConfiguration("task_event_topic"),
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("trial_id", default_value=""),
            DeclareLaunchArgument(
                "output_dir",
                default_value="/home/yecheng/bluerov_ws/bluerov2_payload_retrieval_trials",
            ),
            DeclareLaunchArgument("sample_period_s", default_value="0.05"),
            DeclareLaunchArgument("metadata_file", default_value=""),
            DeclareLaunchArgument("controller_name", default_value="unknown_controller"),
            DeclareLaunchArgument("environment", default_value="sim_or_pool"),
            DeclareLaunchArgument("operator", default_value=""),
            DeclareLaunchArgument("notes", default_value=""),
            DeclareLaunchArgument("odom_topic", default_value="/itrl_rov_1/fmu/out/vehicle_odometry"),
            DeclareLaunchArgument("mocap_odom_topic", default_value=""),
            DeclareLaunchArgument("cmd_vel_topic", default_value="/itrl_rov_1/cmd_vel"),
            DeclareLaunchArgument(
                "thrust_sp_topic",
                default_value="/itrl_rov_1/fmu/in/vehicle_thrust_setpoint",
            ),
            DeclareLaunchArgument(
                "torque_sp_topic",
                default_value="/itrl_rov_1/fmu/in/vehicle_torque_setpoint",
            ),
            DeclareLaunchArgument("attitude_sp_topic", default_value=""),
            DeclareLaunchArgument(
                "control_mode_topic",
                default_value="/itrl_rov_1/fmu/out/vehicle_control_mode",
            ),
            DeclareLaunchArgument("handle_pose_topic", default_value="/payload/handle_pose"),
            DeclareLaunchArgument("handle_confidence_topic", default_value="/payload/handle_confidence"),
            DeclareLaunchArgument("handle_detected_topic", default_value="/payload/handle_detected"),
            DeclareLaunchArgument("payload_pose_topic", default_value="/payload/pose"),
            DeclareLaunchArgument("dock_pose_topic", default_value="/dock/pose"),
            DeclareLaunchArgument("task_event_topic", default_value="/bluerov2/trial_event"),
            logger,
        ]
    )
