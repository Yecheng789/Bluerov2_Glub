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
            executable="stabilized_control",
            name="stabilized_control",
            output="screen",
            parameters=[{
                "odom_topic": f"{ns}/fmu/out/vehicle_odometry",
                # "odom_topic": "/mocap/itrl_rov_1/odom", # real mocap odom
                "control_mode_topic": f"{ns}/fmu/out/vehicle_control_mode",
                "thrust_sp_topic": f"{ns}/fmu/in/vehicle_thrust_setpoint",
                "torque_sp_topic": f"{ns}/fmu/in/vehicle_torque_setpoint",
                "cmd_vel_topic": f"{ns}/cmd_vel",

                # PX4 defaults
                "UUV_ROLL_P": 4.0,
                "UUV_ROLL_D": 1.5,
                "UUV_PITCH_P": 4.0,
                "UUV_PITCH_D": 2.0,
                "UUV_YAW_P": 2.0,
                "UUV_YAW_D": 0.5,

                "UUV_SGM_YAW": 0.7,
                "UUV_SGM_THRTL": 0.15,
                "UUV_TORQUE_SAT": 0.4,
                "UUV_THRUST_SAT": 0.15,

                # 0 = XYZ thrust (heave/sway enabled), 1 = surge-only
                "UUV_STICK_MODE": 0,

                # Convention knobs
                "yaw_sign": -1.0,
                "z_sign": -1.0,
                "yaw_rate_lpf_hz": 8.0,

                "cmd_timeout": 0.3,
                "cmd_deadband": 0.05,
                "loop_dt": 0.02,
            }],
        ),
    ])