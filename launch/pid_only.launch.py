from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ns = "itrl_rov_1"  # change if needed

    heartbeat = Node(
        package="bluerov2_mpc",
        executable="offboard_heartbeat",
        name="offboard_heartbeat",
        parameters=[{
            "topic": f"/{ns}/fmu/in/offboard_control_mode",
        }],
        output="screen"
    )

    pid = Node(
        package="bluerov2_mpc",
        executable="soft_dive_pid",
        name="soft_dive_pid",
        parameters=[{
            "odom_topic": f"/mocap/{ns}/odom",
            "motors_topic": f"/{ns}/fmu/in/actuator_motors",
            "z_ref": 1.2,
            "soft_bias": 0.8,
            "start_mpc_on_ready": False,  # PID-only test
        }],
        output="screen"
    )

    return LaunchDescription([heartbeat, pid])