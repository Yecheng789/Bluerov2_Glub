from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ns = "/itrl_rov_1"

    return LaunchDescription([
        Node(
            package="bluerov2_control",
            executable="offboard_heartbeat_wrench",
            name="offboard_heartbeat_wrench",
            output="screen",
            parameters=[{
                "topic": f"{ns}/fmu/in/offboard_control_mode",
            }],
        ),

        Node(
            package="bluerov2_control",
            executable="mpc_hold_position",
            name="mpc_hold_position",
            output="screen",
            parameters=[{
                # Topics
                "odom_topic": f"{ns}/fmu/out/vehicle_odometry",
                # "odom_topic": "/mocap/itrl_rov_1/odom", # real mocap odom
                "control_mode_topic": f"{ns}/fmu/out/vehicle_control_mode",
                "thrust_sp_topic": f"{ns}/fmu/in/vehicle_thrust_setpoint",
                "torque_sp_topic": f"{ns}/fmu/in/vehicle_torque_setpoint",

                # Goal (PX4 local NED tank center)
                "goal_x": -1.15,
                "goal_y": -2.175,
                "goal_z": 95.7,
                "goal_roll": 0.0,
                "goal_pitch": 0.0,
                "goal_yaw": 0.0,
                "hold_attitude": True,

                # MPC settings
                "Ts": 0.04,
                "N": 25,
                "solve_rate_hz": 25.0,

                # Model selector
                # options: "double_integrator", "fossen" or "koopman"
                "model_type": "fossen",

                # Model (only for double_integrator)
                "mass": 13.0,
                "Ix": 0.25,
                "Iy": 0.221,
                "Iz": 0.356,

                # Weights
                "w_pos": 50.0,
                "w_vel": 15.0,
                "w_att": 8.0,
                "w_omega": 1.0,
                "w_u_force": 0.1,
                "w_u_torque": 0.05,

                # Physical force bounds (Newtons/Nm) used by MPC + normalization
                "Fx_max_N": 88.0,
                "Fy_max_N": 88.0,
                "Fz_max_N": 137.0,
                "Mx_max_Nm": 30.0,
                "My_max_Nm": 16.5,
                "Mz_max_Nm": 21.0,

                # Published normalized thrust and torque clamp
                "thrust_sat": 0.2,
                "torque_sat": 0.2,

                # Publish rate
                "publish_dt": 0.02,
            }],
        ),
    ])