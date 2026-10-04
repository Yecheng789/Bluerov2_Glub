#!/usr/bin/env python3
"""Run guarded fixed-hook validation with the Splash robot profile."""

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


SPLASH_PROFILE = {
    'rigid_body_name': 'splash',
    'robot_namespace': '/splash',
    'allow_identity_derived_target': 'true',
    'prehook_attitude_reference_mode': 'recorded_hook',
    # Retain the last audited compatibility thresholds for explicit
    # position_mode=false runs. The active real profile below does not freeze
    # its line reference at these ordinary-error thresholds.
    'prehook_reached_orientation_tol_deg': '10.0',
    'prehook_reached_forward_axis_tol_deg': '5.0',
    'prehook_reached_yaw_tol_deg': '5.0',
    # The 2026-08-29 trial proved that freezing at the 3 cm corridor boundary
    # both brakes the vehicle and removes vertical damping. Use the common
    # real-robot Position-like line: monotonic time reference, constant NED
    # depth, and no ordinary-error freeze/reverse.
    'fixed_hook_line_position_mode': 'true',
    'fixed_hook_line_yaw_tol_deg': '7.0',
    'fixed_hook_line_cross_track_tol_m': '0.03',
    # Compatibility-only measured-progress lead. Position-like mode does not
    # use this value, but keeping the last audited 4 cm setting makes an
    # explicit compatibility run reproducible.
    'fixed_hook_line_max_reference_lead_m': '0.04',
    'fixed_hook_line_velocity_weight_multiplier': '20.0',
}

# The operator confirmed that Splash matches Glub in every non-network,
# non-MoCap physical/control setting. The pose itself must still be recorded
# from the current Splash rigid body: the historical
# retrieval_20260818_190143_613458 trial (recorded under the former splash_fb
# name) disproved the assumption that the Glub and Splash quaternions were
# interchangeable.
SPLASH_DEFAULTS = {
    'target_system_id': '3',
    'target_component_id': '1',
    'robot_type': 'standard',
    'mocap_body_frame': 'frd',
}
REQUIRED_ARGUMENTS = ('target_config',)


def generate_launch_description():
    """Include the common fail-closed launch with Splash-specific names."""
    launch_path = (
        get_package_share_directory('bluerov2_control')
        + '/launch/fixed_hook_pose_validation.launch.py'
    )
    declarations = [
        DeclareLaunchArgument(
            'rigid_body_name',
            default_value=SPLASH_PROFILE['rigid_body_name'],
            description='Fixed by this profile to splash.',
        ),
        DeclareLaunchArgument(
            'robot_namespace',
            default_value=SPLASH_PROFILE['robot_namespace'],
            description='Fixed by this profile to /splash.',
        ),
        DeclareLaunchArgument(
            'target_config',
            description=(
                'Required direct /mocap/splash/pose target or the audited '
                'hash-locked manifest for the name-only splash_fb-to-splash '
                'change. Cross-rigid-body Glub targets are rejected.'
            ),
        ),
        DeclareLaunchArgument(
            'allow_identity_derived_target',
            default_value=SPLASH_PROFILE['allow_identity_derived_target'],
            description=(
                'Fixed true only for an audited, hash-locked identity '
                'transfer from the former splash_fb name. The invalidated '
                'Glub-to-Splash transfer remains rejected.'
            ),
        ),
        DeclareLaunchArgument(
            'prehook_attitude_reference_mode',
            default_value=SPLASH_PROFILE['prehook_attitude_reference_mode'],
            description=(
                'Fixed recorded_hook for Splash: pre-hook and Hook use the '
                'same pose from the validated Splash source recording.'
            ),
        ),
        DeclareLaunchArgument(
            'prehook_reached_orientation_tol_deg',
            default_value=SPLASH_PROFILE[
                'prehook_reached_orientation_tol_deg'
            ],
            description=(
                'Fixed 10-degree Splash pre-hook full-attitude safety '
                'envelope. This relaxation applies only before the final '
                'straight Hook corridor.'
            ),
        ),
        DeclareLaunchArgument(
            'prehook_reached_forward_axis_tol_deg',
            default_value=SPLASH_PROFILE[
                'prehook_reached_forward_axis_tol_deg'
            ],
            description=(
                'Fixed 5-degree Splash body-X forward-axis gate before '
                'GO_FORWARD.'
            ),
        ),
        DeclareLaunchArgument(
            'prehook_reached_yaw_tol_deg',
            default_value=SPLASH_PROFILE['prehook_reached_yaw_tol_deg'],
            description=(
                'Fixed 5-degree Splash heading gate before GO_FORWARD. The '
                'in-transit line envelope is separately fixed at 7 degrees.'
            ),
        ),
        DeclareLaunchArgument(
            'fixed_hook_line_position_mode',
            default_value=SPLASH_PROFILE[
                'fixed_hook_line_position_mode'
            ],
            description=(
                'Fixed true for Splash real trials: use monotonic '
                'Position-like NED translation without ordinary-error '
                'freeze, braking, or rewind.'
            ),
        ),
        DeclareLaunchArgument(
            'fixed_hook_line_yaw_tol_deg',
            default_value=SPLASH_PROFILE['fixed_hook_line_yaw_tol_deg'],
            description=(
                'Compatibility-only 7-degree Splash in-transit heading '
                'threshold. Position-like real translation does not freeze '
                'at this threshold; endpoint acceptance remains 5 degrees.'
            ),
        ),
        DeclareLaunchArgument(
            'fixed_hook_line_cross_track_tol_m',
            default_value=SPLASH_PROFILE[
                'fixed_hook_line_cross_track_tol_m'
            ],
            description=(
                'Compatibility-only 0.03 m Splash cross-track threshold; '
                'the active Position-like line corrects without freezing.'
            ),
        ),
        DeclareLaunchArgument(
            'fixed_hook_line_max_reference_lead_m',
            default_value=SPLASH_PROFILE[
                'fixed_hook_line_max_reference_lead_m'
            ],
            description=(
                'Compatibility-only 0.04 m measured-progress lead; the '
                'active Position-like line uses monotonic trajectory time.'
            ),
        ),
        DeclareLaunchArgument(
            'fixed_hook_line_velocity_weight_multiplier',
            default_value=SPLASH_PROFILE[
                'fixed_hook_line_velocity_weight_multiplier'
            ],
            description=(
                'Fixed 20x Splash NED/world velocity-tracking weight for the '
                'whole GO_FORWARD/GO_BACK line, including zero depth speed.'
            ),
        ),
        DeclareLaunchArgument(
            'target_system_id',
            default_value=SPLASH_DEFAULTS['target_system_id'],
            description=(
                'Splash MAVLink system_id; confirmed equal to Glub.'
            ),
        ),
        DeclareLaunchArgument(
            'target_component_id',
            default_value=SPLASH_DEFAULTS['target_component_id'],
            description=(
                'Splash MAVLink component_id; confirmed equal to Glub.'
            ),
        ),
        DeclareLaunchArgument(
            'robot_type',
            default_value=SPLASH_DEFAULTS['robot_type'],
            description=(
                'Splash dynamics preset; confirmed equal to Glub standard.'
            ),
        ),
        DeclareLaunchArgument(
            'mocap_body_frame',
            default_value=SPLASH_DEFAULTS['mocap_body_frame'],
            description=(
                'Splash body convention; confirmed equal to Glub FRD.'
            ),
        ),
    ]
    included = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_path),
        launch_arguments={
            **SPLASH_PROFILE,
            **{
                name: LaunchConfiguration(name)
                for name in (*REQUIRED_ARGUMENTS, *SPLASH_DEFAULTS)
            },
        }.items(),
    )
    return LaunchDescription([*declarations, included])
