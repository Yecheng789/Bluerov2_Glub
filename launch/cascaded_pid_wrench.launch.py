#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    ns = "itrl_rov_1"

    # Trajectory / rate args
    rate_arg = DeclareLaunchArgument("rate", default_value="20.0")
    traj_mode_arg = DeclareLaunchArgument("traj_mode", default_value="waypoint")
    refx_arg = DeclareLaunchArgument("ref_x", default_value="4.5")
    refy_arg = DeclareLaunchArgument("ref_y", default_value="0.0")
    refz_arg = DeclareLaunchArgument("ref_z", default_value="1.2")

    # Offboard heartbeat (must set thrust_and_torque=True in that node)
    heartbeat = Node(
        package="bluerov2_mpc",
        executable="offboard_heartbeat_wrench",
        name="offboard_heartbeat",
        parameters=[{
            "topic": f"/{ns}/fmu/in/offboard_control_mode",
        }],
        output="screen"
    )

    # Optional: SoftDive PID that publishes /bluerov2_mpc/start_mpc
    soft_dive = Node(
        package="bluerov2_mpc",
        executable="soft_dive_pid",
        name="soft_dive_pid",
        parameters=[{
            "odom_topic": f"/mocap/{ns}/odom",
            "motors_topic": f"/{ns}/fmu/in/actuator_motors",
            "z_ref": 1.2,
            "soft_bias": 0.8,
            "start_mpc_on_ready": True,   # will publish /bluerov2_mpc/start_mpc
            "handoff_seconds": 0.5,
        }],
        output="screen"
    )

    # Inner PID: body velocity -> thrust/torque
    vel_pid = Node(
        package="bluerov2_mpc",
        executable="vel_pid_thrust_node",
        name="vel_pid_thrust_node",
        parameters=[{
            "odom_topic": f"/mocap/{ns}/odom",
            "cmd_vel_topic": "/bluerov2_mpc/body_vel_cmd",
            "thrust_topic": f"/{ns}/fmu/in/vehicle_thrust_setpoint",
            "torque_topic": f"/{ns}/fmu/in/vehicle_torque_setpoint",
            # initial gains / limits, tune in practice
            "kp_lin": 40.0,
            "ki_lin": 5.0,
            "kd_lin": 0.0,
            "kp_ang": 10.0,
            "ki_ang": 2.0,
            "kd_ang": 0.0,
            "f_max": 40.0,
            "tau_max": 8.0,
        }],
        output="screen"
    )

    # Outer PID: pose tracking -> body velocity setpoints
    pose_pid = Node(
        package="bluerov2_mpc",
        executable="pose_vel_pid_node",
        name="pose_vel_pid_node",
        output="screen",
        parameters=[{
            "rate": LaunchConfiguration("rate"),
            "odom_topic": f"/mocap/{ns}/odom",
            "cmd_vel_topic": "/bluerov2_mpc/body_vel_cmd",
            "traj_mode": LaunchConfiguration("traj_mode"),
            "ref_x": LaunchConfiguration("ref_x"),
            "ref_y": LaunchConfiguration("ref_y"),
            "ref_z": LaunchConfiguration("ref_z"),
            "require_start_signal": True,
            "start_signal_topic": "/bluerov2_mpc/start_mpc",
            # PID gains and limits can be overridden at launch
            "kp_pos": 0.6,
            "ki_pos": 0.05,
            "kd_pos": 0.0,
            "kp_yaw": 1.5,
            "ki_yaw": 0.05,
            "kd_yaw": 0.0,
            "v_max_xy": 0.3,
            "v_max_z": 0.3,
            "w_max_yaw": 0.5,
        }]
    )

    return LaunchDescription([
        rate_arg, traj_mode_arg, refx_arg, refy_arg, refz_arg,
        heartbeat,
        soft_dive,
        vel_pid,
        pose_pid,
    ])