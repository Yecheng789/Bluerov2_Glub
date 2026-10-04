from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    publish_rate_hz = ParameterValue(
        LaunchConfiguration("publish_rate_hz"),
        value_type=float,
    )
    publish_tf = ParameterValue(
        LaunchConfiguration("publish_tf"),
        value_type=bool,
    )
    publish_static_flu_tf = ParameterValue(
        LaunchConfiguration("publish_static_flu_tf"),
        value_type=bool,
    )
    use_imu_gyro = ParameterValue(
        LaunchConfiguration("use_imu_gyro"),
        value_type=bool,
    )
    max_coast_sec = ParameterValue(
        LaunchConfiguration("max_coast_sec"),
        value_type=float,
    )
    max_rejected_samples = ParameterValue(
        LaunchConfiguration("max_rejected_samples"),
        value_type=int,
    )
    max_position_innovation_m = ParameterValue(
        LaunchConfiguration("max_position_innovation_m"),
        value_type=float,
    )
    max_orientation_innovation_rad = ParameterValue(
        LaunchConfiguration("max_orientation_innovation_rad"),
        value_type=float,
    )
    max_base_link_z_axis_angle_rad = ParameterValue(
        LaunchConfiguration("max_base_link_z_axis_angle_rad"),
        value_type=float,
    )

    mocap_ekf = Node(
        package="bluerov2_control",
        executable="mocap_ekf_odom",
        name="mocap_ekf_odom",
        output="screen",
        parameters=[
            {
                "rigid_body_name": LaunchConfiguration("rigid_body_name"),
                "pose_topic": LaunchConfiguration("pose_topic"),
                "odom_topic": LaunchConfiguration("odom_topic"),
                "imu_topic": LaunchConfiguration("imu_topic"),
                "parent_frame": LaunchConfiguration("parent_frame"),
                "child_frame": LaunchConfiguration("child_frame"),
                "child_frame_flu": LaunchConfiguration("child_frame_flu"),
                "publish_rate_hz": publish_rate_hz,
                "publish_tf": publish_tf,
                "publish_static_flu_tf": publish_static_flu_tf,
                "use_imu_gyro": use_imu_gyro,
                "orientation_correction_quat_xyzw": LaunchConfiguration(
                    "orientation_correction_quat_xyzw"
                ),
                "max_coast_sec": max_coast_sec,
                "max_rejected_samples": max_rejected_samples,
                "max_position_innovation_m": max_position_innovation_m,
                "max_orientation_innovation_rad": max_orientation_innovation_rad,
                "max_base_link_z_axis_angle_rad": max_base_link_z_axis_angle_rad,
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("rigid_body_name", default_value="glub"),
            DeclareLaunchArgument("pose_topic", default_value=""),
            DeclareLaunchArgument("odom_topic", default_value=""),
            DeclareLaunchArgument(
                "imu_topic",
                default_value="/mavros/imu/data",
            ),
            DeclareLaunchArgument("parent_frame", default_value="mocap"),
            DeclareLaunchArgument("child_frame", default_value=""),
            DeclareLaunchArgument("child_frame_flu", default_value=""),
            DeclareLaunchArgument("publish_rate_hz", default_value="80.0"),
            DeclareLaunchArgument("publish_tf", default_value="true"),
            DeclareLaunchArgument(
                "publish_static_flu_tf",
                default_value="true",
            ),
            DeclareLaunchArgument("use_imu_gyro", default_value="false"),
            DeclareLaunchArgument(
                "orientation_correction_quat_xyzw",
                default_value="",
            ),
            DeclareLaunchArgument("max_coast_sec", default_value="1.0"),
            DeclareLaunchArgument("max_rejected_samples", default_value="200"),
            DeclareLaunchArgument(
                "max_position_innovation_m",
                default_value="0.25",
            ),
            DeclareLaunchArgument(
                "max_orientation_innovation_rad",
                default_value="0.75",
            ),
            DeclareLaunchArgument(
                "max_base_link_z_axis_angle_rad",
                default_value="0.4",
            ),
            mocap_ekf,
        ]
    )
