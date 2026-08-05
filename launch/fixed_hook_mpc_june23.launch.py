#!/usr/bin/env python3
"""Compatibility wrapper for the guarded fixed-hook reality validation."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


FORWARDED_ARGUMENTS = (
    'rigid_body_name',
    'robot_namespace',
    'target_config',
    'mocap_world_frame',
    'pool_bounds_mocap',
    'pool_safety_margin_m',
    'mocap_body_frame',
    'robot_type',
    'orientation_correction_quat_xyzw',
    'target_orientation_correction_mode',
    'traj_speed_mps',
    'pre_approach_speed_mps',
    'final_approach_speed_mps',
    'min_traj_duration_s',
    'pre_approach_distance_m',
    'fixed_hook_depth_tolerance_m',
    'traj_angular_speed_deg_s',
    'final_pose_hold_s',
    'retreat_speed_mps',
    'px4_angular_velocity_timeout_sec',
    'w_att',
    'w_omega',
    'w_u_torque',
    'position_integral_gain_N_per_m_s',
    'position_integral_force_limit_fraction',
    'position_integral_activation_error_m',
    'thrust_sat',
    'torque_sat',
    'max_initial_goal_distance_m',
    'max_initial_goal_orientation_error_deg',
    'max_odom_position_jump_m',
    'max_odom_orientation_jump_deg',
    'max_raw_mocap_message_age_sec',
    'max_mocap_coast_sec',
    'target_system_id',
    'target_component_id',
    'source_system_id',
    'source_component_id',
    'trial_output_dir',
    'trial_id',
    'acados_source_dir',
    'rebuild_solver',
)


DEFAULTS = {
    'rigid_body_name': 'glub_fb',
    'robot_namespace': '/glub',
    'target_config': (
        '/home/yecheng/bluerov_ws/src/bluerov2_control/'
        'experiments/payload_retrieval/config/'
        'hooked_box_target_pose_20260802_195146.json'
    ),
    'mocap_world_frame': 'ned',
    'pool_bounds_mocap': '0 9 -2.5 2.5 0 3',
    'pool_safety_margin_m': '0.25',
    'mocap_body_frame': 'frd',
    'robot_type': 'standard',
    'orientation_correction_quat_xyzw': '',
    'target_orientation_correction_mode': 'auto',
    'traj_speed_mps': '0.05',
    'pre_approach_speed_mps': '0.06',
    'final_approach_speed_mps': '0.04',
    'min_traj_duration_s': '5.0',
    'pre_approach_distance_m': '0.50',
    'fixed_hook_depth_tolerance_m': '0.03',
    'traj_angular_speed_deg_s': '8.0',
    'final_pose_hold_s': '5.0',
    'retreat_speed_mps': '0.04',
    'px4_angular_velocity_timeout_sec': '0.10',
    'w_att': '10.0',
    'w_omega': '20.0',
    'w_u_torque': '0.5',
    'position_integral_gain_N_per_m_s': '3.0',
    'position_integral_force_limit_fraction': '0.07',
    'position_integral_activation_error_m': '0.50',
    'thrust_sat': '0.12',
    'torque_sat': '0.02',
    'max_initial_goal_distance_m': '0',
    'max_initial_goal_orientation_error_deg': '0',
    'max_odom_position_jump_m': '0.20',
    'max_odom_orientation_jump_deg': '20.0',
    'max_raw_mocap_message_age_sec': '0.20',
    'max_mocap_coast_sec': '2.0',
    'target_system_id': '3',
    'target_component_id': '1',
    'source_system_id': '1',
    'source_component_id': '191',
    'trial_output_dir': (
        '/home/yecheng/bluerov_ws/bluerov2_payload_retrieval_trials'
    ),
    'trial_id': '',
    'acados_source_dir': '/home/yecheng/acados',
    'rebuild_solver': 'false',
}


def generate_launch_description():
    """Forward the old launch name to the new fail-closed experiment."""
    launch_path = (
        get_package_share_directory('bluerov2_control')
        + '/launch/fixed_hook_pose_validation.launch.py'
    )
    declarations = [
        DeclareLaunchArgument(name, default_value=DEFAULTS[name])
        for name in FORWARDED_ARGUMENTS
    ]
    included = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_path),
        launch_arguments={
            name: LaunchConfiguration(name)
            for name in FORWARDED_ARGUMENTS
        }.items(),
    )
    return LaunchDescription([*declarations, included])
