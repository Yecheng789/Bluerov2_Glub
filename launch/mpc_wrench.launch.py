#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    ns = "itrl_rov_1"

    # ---- Common args for Wrench MPC ----
    model_arg = DeclareLaunchArgument(
        "model", default_value="koopman_wrench",
        description="Wrench model backend: fossen_wrench | di_wrench | di_ana_wrench | koopman_wrench"
    )
    di_weights_arg = DeclareLaunchArgument(
        "di_weights", default_value="models/weights/double_integrator_wrench_weights.npz"
    )
    koop_weights_arg = DeclareLaunchArgument(
        "koopman_weights", default_value="models/weights/koopman_edmdc_wrench_weights.npz"
    )
    rate_arg = DeclareLaunchArgument("rate", default_value="20.0")
    dt_arg = DeclareLaunchArgument("dt", default_value="0.01")
    horizon_arg = DeclareLaunchArgument("horizon", default_value="15")
    traj_mode_arg = DeclareLaunchArgument("traj_mode", default_value="waypoint")
    refx_arg = DeclareLaunchArgument("ref_x", default_value="4.5")
    refy_arg = DeclareLaunchArgument("ref_y", default_value="0.0")
    refz_arg = DeclareLaunchArgument("ref_z", default_value="1.2")  # positive down

    force_max_arg = DeclareLaunchArgument("force_max", default_value="40.0")
    torque_max_arg = DeclareLaunchArgument("torque_max", default_value="10.0")

    # ---- Offboard heartbeat ----
    # NOTE: for wrench mode PX4 usually needs thrust_and_torque=True and direct_actuator=False.
    heartbeat = Node(
        package="bluerov2_mpc",
        executable="offboard_heartbeat_wrench",
        name="offboard_heartbeat",
        parameters=[{
            "topic": f"/{ns}/fmu/in/offboard_control_mode",
        }],
        output="screen"
    )

    # ---- SoftDive PID (still drives /actuator_motors and sends start_mpc) ----
    pid = Node(
        package="bluerov2_mpc",
        executable="soft_dive_pid",
        name="soft_dive_pid",
        parameters=[{
            "odom_topic": f"/mocap/{ns}/odom",
            "motors_topic": f"/{ns}/fmu/in/actuator_motors",
            "z_ref": 1.2,
            "soft_bias": 0.8,
            "start_mpc_on_ready": True,
            "handoff_seconds": 0.5,
        }],
        output="screen"
    )

    # ---- Wrench MPC Node (waits for /bluerov2_mpc/start_mpc) ----
    wrench_mpc = Node(
        package="bluerov2_mpc",
        executable="mpc_wrench_node",
        name="bluerov2_mpc_wrench",
        output="screen",
        parameters=[{
            "model": LaunchConfiguration("model"),
            "di_weights": LaunchConfiguration("di_weights"),
            "koopman_weights": LaunchConfiguration("koopman_weights"),
            "rate": LaunchConfiguration("rate"),
            "dt": LaunchConfiguration("dt"),
            "horizon": LaunchConfiguration("horizon"),
            "traj_mode": LaunchConfiguration("traj_mode"),
            "ref_x": LaunchConfiguration("ref_x"),
            "ref_y": LaunchConfiguration("ref_y"),
            "ref_z": LaunchConfiguration("ref_z"),
            "force_max": LaunchConfiguration("force_max"),
            "torque_max": LaunchConfiguration("torque_max"),
            "require_start_signal": True,
            "start_signal_topic": "/bluerov2_mpc/start_mpc",
        }]
    )

    return LaunchDescription([
        model_arg, di_weights_arg, koop_weights_arg,
        rate_arg, dt_arg, horizon_arg,
        traj_mode_arg, refx_arg, refy_arg, refz_arg,
        force_max_arg, torque_max_arg,
        heartbeat, pid, wrench_mpc
    ])