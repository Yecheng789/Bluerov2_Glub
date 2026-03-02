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
                "control_mode_topic": f"{ns}/fmu/out/vehicle_control_mode",
                "thrust_sp_topic": f"{ns}/fmu/in/vehicle_thrust_setpoint",
                "torque_sp_topic": f"{ns}/fmu/in/vehicle_torque_setpoint",

                # Goal (PX4 local NED tank center)
                "goal_x": -1.15,
                "goal_y": -2.175,
                "goal_z": 95.7,
                "hold_yaw": False,
                "yaw_goal": 0.0,

                # If sway/depth are inverted, flip these (only affects THIS node)
                # examples:
                #   right goes left -> set sign_y := -1.0
                #   up/down inverted -> set sign_z := -1.0 (remember: z is DOWN-positive in NED)
                "sign_x": 1.0,
                "sign_y": 1.0,
                "sign_z": 1.0,

                # MPC settings
                "Ts": 0.10,
                "N": 15,
                "solve_rate_hz": 10.0,

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

                # Physical force bounds (Newtons) used by MPC + normalization
                "Fx_max_N": 20.0,
                "Fy_max_N": 20.0,
                "Fz_max_N": 30.0,

                # Published normalized thrust clamp (keep small initially)
                "thrust_sat": 0.15,

                # Publish rate
                "publish_dt": 0.02,

                # Axis test (set True to validate directions)
                "axis_test_enable": False,
                "axis_test_axis": "x",
                "axis_test_force_N": 5.0,
            }],
        ),
    ])