from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="bluerov2_control",
            executable="offboard_heartbeat_wrench",
            name="offboard_heartbeat_wrench",
            output="screen",
            parameters=[{
                "topic": "/fmu/in/offboard_control_mode",
            }],
        ),

        Node(
            package="bluerov2_control",
            executable="stabilized_control",
            name="stabilized_control",
            output="screen",
            parameters=[{
                "odom_topic": "/fmu/out/vehicle_odometry",
                "control_mode_topic": "/fmu/out/vehicle_control_mode",
                "thrust_sp_topic": "/fmu/in/vehicle_thrust_setpoint",
                "torque_sp_topic": "/fmu/in/vehicle_torque_setpoint",
                "cmd_vel_topic": "/itrl_rov_1/cmd_vel",

                "UUV_ROLL_P": 4.0,
                "UUV_ROLL_D": 1.5,
                "UUV_PITCH_P": 4.0,
                "UUV_PITCH_D": 2.0,
                "UUV_YAW_P": 4.0,
                "UUV_YAW_D": 2.0,

                "UUV_SGM_YAW": 0.7,
                "UUV_SGM_THRTL": 0.15,
                "UUV_TORQUE_SAT": 1.2,
                "UUV_THRUST_SAT": 0.25,

                "UUV_STICK_MODE": 0,

                "yaw_sign": 1.0,
                "z_sign": -1.0,
                "yaw_rate_lpf_hz": 8.0,

                "cmd_timeout": 0.3,
                "cmd_deadband": 0.05,
                "loop_dt": 0.02,
            }],
        ),
    ])