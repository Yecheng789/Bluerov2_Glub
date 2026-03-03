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
                "hold_yaw": True,
                "yaw_goal": 0.0,

                # MPC settings
                "Ts": 0.10,
                "N": 15,
                "solve_rate_hz": 20.0,

                # Model
                "mass": 13.5,
                "Ix": 0.26,
                "Iy": 0.23,
                "Iz": 0.37,

                # Weights
                "w_pos": 50.0,
                "w_vel": 5.0,
                "w_yaw": 5.0,
                "w_omega": 0.2,
                "w_u_force": 0.1,
                "w_u_torque": 0.0,

                # Physical force bounds (Newtons/Nm) used by MPC + normalization
                "Fx_max_N": 88.0,
                "Fy_max_N": 88.0,
                "Fz_max_N": 137.0,
                "Mz_max_Nm": 0.2,
                "Mx_max_Nm": 0.0,
                "My_max_Nm": 0.0,

                # Published normalized thrust clamp (keep small initially)
                "thrust_sat": 0.15,

                # Publish rate
                "publish_dt": 0.02,
            }],
        ),
    ])