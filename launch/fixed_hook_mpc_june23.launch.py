import json
import math
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


DEFAULT_ORIENTATION_CORRECTION_QUAT_XYZW = (
    "0.04430086214711217 -0.001252015250325171 "
    "-5.551990616173286e-05 0.9990174487907484"
)


def _quat_normalize(q):
    x, y, z, w = [float(v) for v in q]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12 or not math.isfinite(norm):
        return (0.0, 0.0, 0.0, 1.0)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    return (x, y, z, w)


def _quat_multiply_xyzw(q1, q2):
    x1, y1, z1, w1 = _quat_normalize(q1)
    x2, y2, z2, w2 = _quat_normalize(q2)
    return _quat_normalize(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )
    )


def _parse_quat_xyzw(text):
    text = str(text).strip()
    if not text:
        return (0.0, 0.0, 0.0, 1.0)
    values = [float(value) for value in text.replace(",", " ").split()]
    if len(values) != 4:
        raise ValueError(
            "orientation_correction_quat_xyzw must contain four values: "
            "x y z w"
        )
    return _quat_normalize(values)


def _parse_bool(text):
    return str(text).strip().lower() in ("1", "true", "yes", "on")


def _quat_xyzw_to_rpy(q):
    x, y, z, w = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _load_target(path, orientation_correction=None):
    with Path(path).expanduser().open("r", encoding="utf-8") as src:
        payload = json.load(src)
    target = payload["target_pose"]
    position = target["position"]
    quat = target["orientation_xyzw"]
    quat_xyzw = (
        float(quat["x"]),
        float(quat["y"]),
        float(quat["z"]),
        float(quat["w"]),
    )
    if orientation_correction is not None:
        quat_xyzw = _quat_multiply_xyzw(quat_xyzw, orientation_correction)
    roll, pitch, yaw = _quat_xyzw_to_rpy(
        quat_xyzw
    )
    return {
        "x": float(position["x"]),
        "y": float(position["y"]),
        "z": float(position["z"]),
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
    }


def _make_nodes(context, *args, **kwargs):
    target_config = LaunchConfiguration("target_config").perform(context)
    rigid_body_name = LaunchConfiguration("rigid_body_name").perform(context)
    robot_ns = LaunchConfiguration("robot_namespace").perform(context).rstrip("/")
    orientation_correction_text = LaunchConfiguration(
        "orientation_correction_quat_xyzw"
    ).perform(context)
    apply_target_correction = _parse_bool(
        LaunchConfiguration("apply_orientation_correction_to_target").perform(
            context
        )
    )
    orientation_correction = _parse_quat_xyzw(orientation_correction_text)
    target = _load_target(
        target_config,
        orientation_correction if apply_target_correction else None,
    )

    nav_odom_topic = f"/mocap/{rigid_body_name}/odom_ekf"
    vehicle_odom_topic = f"/mocap/{rigid_body_name}/vehicle_odometry_ekf"

    heartbeat = Node(
        package="bluerov2_control",
        executable="offboard_heartbeat_wrench",
        name="offboard_heartbeat_wrench",
        output="screen",
        parameters=[
            {
                "topic": f"{robot_ns}/fmu/in/offboard_control_mode",
            }
        ],
    )

    mocap_ekf = Node(
        package="bluerov2_control",
        executable="mocap_ekf_odom",
        name="mocap_ekf_odom",
        output="screen",
        parameters=[
            {
                "rigid_body_name": rigid_body_name,
                "pose_topic": f"/mocap/{rigid_body_name}/pose",
                "odom_topic": nav_odom_topic,
                "publish_rate_hz": 80.0,
                "publish_tf": True,
                "use_imu_gyro": False,
                "orientation_correction_quat_xyzw": orientation_correction_text,
                "max_coast_sec": 1.0,
                "max_rejected_samples": 200,
            }
        ],
    )

    odom_adapter = Node(
        package="bluerov2_control",
        executable="nav_odom_to_vehicle_odometry",
        name="nav_odom_to_vehicle_odometry",
        output="screen",
        parameters=[
            {
                "input_odom_topic": nav_odom_topic,
                "output_vehicle_odometry_topic": vehicle_odom_topic,
                "pose_frame": "frd",
                "velocity_frame": "body_frd",
            }
        ],
    )

    mpc = Node(
        package="bluerov2_control",
        executable="mpc_track_trajectory_acados",
        name="mpc_track_trajectory_acados_fixed_hook",
        output="screen",
        parameters=[
            {
                "odom_topic": vehicle_odom_topic,
                "control_mode_topic": f"{robot_ns}/fmu/out/vehicle_control_mode",
                "thrust_sp_topic": f"{robot_ns}/fmu/in/vehicle_thrust_setpoint",
                "torque_sp_topic": f"{robot_ns}/fmu/in/vehicle_torque_setpoint",
                "goal_x": target["x"],
                "goal_y": target["y"],
                "goal_z": target["z"],
                "goal_roll": target["roll"],
                "goal_pitch": target["pitch"],
                "goal_yaw": target["yaw"],
                "hold_attitude": True,
                "traj_mode": "linear",
                "traj_speed_mps": float(
                    LaunchConfiguration("traj_speed_mps").perform(context)
                ),
                "min_traj_duration_s": 5.0,
                "goal_reached_tol_m": 0.05,
                "regenerate_on_goal_change": False,
                "planner_mode": "none",
                "use_box_recovery_mission": False,
                "Ts": 0.04,
                "N": 25,
                "solve_rate_hz": 25.0,
                "model_type": "fossen",
                "w_pos": 50.0,
                "w_vel": 15.0,
                "w_att": 20.0,
                "w_omega": 4.0,
                "w_u_force": 0.1,
                "w_u_torque": 0.05,
                "Fx_max_N": 88.0,
                "Fy_max_N": 88.0,
                "Fz_max_N": 137.0,
                "Mx_max_Nm": 30.0,
                "My_max_Nm": 16.5,
                "Mz_max_Nm": 21.0,
                "thrust_sat": float(
                    LaunchConfiguration("thrust_sat").perform(context)
                ),
                "torque_sat": float(
                    LaunchConfiguration("torque_sat").perform(context)
                ),
                "publish_dt": 0.02,
                "odom_timeout_s": 0.30,
                "codegen_dir": "/tmp/bluerov2_acados_fixed_hook",
                "rebuild_solver": False,
            }
        ],
    )

    logger = Node(
        package="bluerov2_control",
        executable="payload_retrieval_data_logger",
        name="payload_retrieval_data_logger_fixed_hook",
        output="screen",
        parameters=[
            {
                "controller_name": "mpc_fixed_hook_june23",
                "environment": "kth_pool",
                "metadata_file": target_config,
                "mocap_odom_topic": nav_odom_topic,
                "odom_topic": vehicle_odom_topic,
                "control_mode_topic": f"{robot_ns}/fmu/out/vehicle_control_mode",
                "thrust_sp_topic": f"{robot_ns}/fmu/in/vehicle_thrust_setpoint",
                "torque_sp_topic": f"{robot_ns}/fmu/in/vehicle_torque_setpoint",
                "cmd_vel_topic": "",
            }
        ],
    )

    return [heartbeat, mocap_ekf, odom_adapter, mpc, logger]


def generate_launch_description():
    acados_source_dir = "/home/yecheng/acados"
    acados_lib_dir = f"{acados_source_dir}/lib"
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if ld_library_path:
        ld_library_path = f"{acados_lib_dir}:{ld_library_path}"
    else:
        ld_library_path = acados_lib_dir

    return LaunchDescription(
        [
            SetEnvironmentVariable("ACADOS_SOURCE_DIR", acados_source_dir),
            SetEnvironmentVariable("LD_LIBRARY_PATH", ld_library_path),
            DeclareLaunchArgument("rigid_body_name", default_value="glub"),
            DeclareLaunchArgument("robot_namespace", default_value="/glub"),
            DeclareLaunchArgument(
                "target_config",
                default_value=(
                    "/home/yecheng/bluerov_ws/src/bluerov2_control/"
                    "experiments/payload_retrieval/config/"
                    "hooked_box_target_pose_trial_baseline.json"
                ),
            ),
            DeclareLaunchArgument("traj_speed_mps", default_value="0.04"),
            DeclareLaunchArgument("thrust_sat", default_value="0.04"),
            DeclareLaunchArgument("torque_sat", default_value="0.05"),
            DeclareLaunchArgument(
                "orientation_correction_quat_xyzw",
                default_value=DEFAULT_ORIENTATION_CORRECTION_QUAT_XYZW,
            ),
            DeclareLaunchArgument(
                "apply_orientation_correction_to_target",
                default_value="true",
            ),
            OpaqueFunction(function=_make_nodes),
        ]
    )
