from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Adjust namespace/topic prefixes if needed
    ns = "/itrl_rov_1"

    return LaunchDescription([
        # OffboardControlMode heartbeat
        Node(
            package="bluerov2_control",
            executable="offboard_heartbeat_wrench",
            name="offboard_heartbeat_wrench",
            output="screen",
            parameters=[{
                "topic": f"{ns}/fmu/in/offboard_control_mode",
            }],
        ),

        # Position PID node
        Node(
            package="bluerov2_control",
            executable="position_control_pid",
            name="position_control_pid",
            output="screen",
            parameters=[{
                "odom_topic": f"{ns}/fmu/out/vehicle_odometry",
                # "odom_topic": "/mocap/itrl_rov_1/odom", # real mocap odom
                "thrust_sp_topic": f"{ns}/fmu/in/vehicle_thrust_setpoint",
                "torque_sp_topic": f"{ns}/fmu/in/vehicle_torque_setpoint",
                "control_mode_topic": f"{ns}/fmu/out/vehicle_control_mode",

                # PX4-like gains
                "gain_x_p": 1.0,
                "gain_y_p": 1.0,
                "gain_z_p": 1.0,
                "gain_x_d": 0.2,
                "gain_y_d": 0.2,
                "gain_z_d": 0.2,

                # Setpoint generation (PX4-like)
                "pos_stick_db": 0.1,
                "pgm_vel": 0.5,
                "sgm_yaw": 0.8,
                "pos_mode": 1,   # 1 = body frame, 0 = world frame

                # Output limits (tune)
                "max_thrust_xy": 0.6,
                "max_thrust_z": 0.6,
                "max_torque_z": 0.4,

                # Yaw torque
                "yaw_kp": 0.3,
                "yaw_kd": 0.2,

                "cmd_vel_topic": f"{ns}/cmd_vel",
                "cmd_timeout": 0.3,
                "yaw_sign": -1.0,
                "z_sign": -1.0,
            }],
        ),
    ])