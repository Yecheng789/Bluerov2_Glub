"""Safety tests for the Splash real fixed-hook launch profile."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription

from launch_ros.actions import Node as LaunchNode
from launch_ros.utilities import evaluate_parameters

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = (
    PACKAGE_ROOT / 'experiments' / 'payload_retrieval' / 'config'
)
GLUB_TARGET_NAME = 'hooked_box_target_pose_20260802_195146.json'
SPLASH_DERIVED_TARGET_NAME = (
    'hooked_box_target_pose_splash_from_glub_20260802_195146.json'
)
SPLASH_RENAME_SOURCE_NAME = (
    'hooked_box_target_pose_splash_20260819_170318.json'
)
SPLASH_RENAME_MANIFEST_NAME = (
    'hooked_box_target_pose_splash_from_splash_fb_20260819_170318.json'
)
GLUB_TARGET_SHA256 = (
    'e61b678f52c9bfe58ebe4b01610a7939a9355c91f2a8f0e81b9699f598316cb7'
)
SPLASH_RENAME_SOURCE_SHA256 = (
    '9237d8460432c0e897713057ac2d3d5c046c9f449b7389a5b35c647341367f3c'
)


def _load_launch_module(filename, module_name):
    path = PACKAGE_ROOT / 'launch' / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _main_launch_context(module):
    context = LaunchContext()
    for entity in module.generate_launch_description().entities:
        if isinstance(entity, DeclareLaunchArgument):
            entity.execute(context)
    return context


def _write_splash_target(tmp_path):
    roll = 0.12
    pitch = -0.18
    yaw = 0.31
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    payload = {
        'frame_id': 'mocap',
        'message_type': 'geometry_msgs/PoseStamped',
        'source_topic': '/mocap/splash/pose',
        'target_pose': {
            'position': {'x': 4.0, 'y': 0.0, 'z': 1.5},
            'orientation_xyzw': {
                'x': sr * cp * cy - cr * sp * sy,
                'y': cr * sp * cy + sr * cp * sy,
                'z': cr * cp * sy - sr * sp * cy,
                'w': cr * cp * cy + sr * sp * sy,
            },
        },
        'validation': {
            'passed': True,
            'settings': {'max_message_age_sec': 0.20},
            'metrics': {
                'accepted_sample_count': 160,
                'header_timestamped_sample_count': 160,
                'distinct_header_timestamp_count': 160,
                'sampling_span_sec': 1.5,
                'message_age_sec': {
                    'min': 0.005,
                    'mean': 0.010,
                    'max': 0.020,
                },
                'max_axis_position_standard_deviation_m': 0.004,
                'orientation': {
                    'angular_std_deg': 0.5,
                    'pairwise_spread_deg': 1.5,
                },
            },
        },
    }
    path = tmp_path / 'hooked_box_target_pose_splash.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _write_derived_fixture(tmp_path):
    """Copy the real source/manifest pair into an isolated mutable folder."""
    source_path = CONFIG_ROOT / GLUB_TARGET_NAME
    manifest_path = CONFIG_ROOT / SPLASH_DERIVED_TARGET_NAME
    copied_source = tmp_path / source_path.name
    copied_manifest = tmp_path / manifest_path.name
    copied_source.write_bytes(source_path.read_bytes())
    copied_manifest.write_bytes(manifest_path.read_bytes())
    # The committed manifest is intentionally invalidated by real trial data.
    # Re-enable only this isolated fixture to retain generic resolver tests.
    manifest = _load_json(copied_manifest)
    manifest['derived_target'][
        'operator_confirmed_same_target_pose'
    ] = True
    manifest.pop('status', None)
    copied_manifest.write_text(json.dumps(manifest), encoding='utf-8')
    return copied_source, copied_manifest


def _node_parameters(context, actions):
    result = {}
    for action in actions:
        if not isinstance(action, LaunchNode):
            continue
        executable = vars(action)['_Node__node_executable']
        parameters = vars(action)['_Node__parameters']
        result[executable] = evaluate_parameters(context, parameters)[0]
    return result


def test_splash_wrapper_requires_a_direct_target_and_keeps_robot_defaults():
    """Require Splash MoCap data while retaining confirmed hardware setup."""
    module = _load_launch_module(
        'fixed_hook_pose_validation_splash.launch.py',
        'fixed_hook_pose_validation_splash_launch',
    )

    assert module.SPLASH_PROFILE == {
        'rigid_body_name': 'splash',
        'robot_namespace': '/splash',
        'allow_identity_derived_target': 'true',
        'prehook_attitude_reference_mode': 'recorded_hook',
        'prehook_reached_orientation_tol_deg': '10.0',
        'prehook_reached_forward_axis_tol_deg': '5.0',
        'prehook_reached_yaw_tol_deg': '5.0',
        'fixed_hook_line_position_mode': 'true',
        'fixed_hook_line_yaw_tol_deg': '7.0',
        'fixed_hook_line_cross_track_tol_m': '0.03',
        'fixed_hook_line_max_reference_lead_m': '0.04',
        'fixed_hook_line_velocity_weight_multiplier': '20.0',
    }
    assert module.SPLASH_DEFAULTS == {
        'target_system_id': '3',
        'target_component_id': '1',
        'robot_type': 'standard',
        'mocap_body_frame': 'frd',
    }
    assert module.REQUIRED_ARGUMENTS == ('target_config',)

    description = module.generate_launch_description()
    declarations = {
        vars(entity)['_DeclareLaunchArgument__name']: entity
        for entity in description.entities
        if isinstance(entity, DeclareLaunchArgument)
    }
    assert set(declarations) == (
        set(module.REQUIRED_ARGUMENTS)
        | set(module.SPLASH_PROFILE)
        | set(module.SPLASH_DEFAULTS)
    )
    context = LaunchContext()
    for name, declaration in declarations.items():
        if name != 'target_config':
            declaration.execute(context)
    assert 'target_config' not in context.launch_configurations
    assert vars(declarations['target_config'])[
        '_DeclareLaunchArgument__default_value'
    ] is None
    assert all(
        vars(declarations[name])['_DeclareLaunchArgument__default_value']
        is not None
        for name in (
            *module.SPLASH_PROFILE,
            *module.SPLASH_DEFAULTS,
        )
    )

    includes = [
        entity
        for entity in description.entities
        if isinstance(entity, IncludeLaunchDescription)
    ]
    assert len(includes) == 1
    include_arguments = dict(
        vars(includes[0])['_IncludeLaunchDescription__launch_arguments']
    )
    assert include_arguments['rigid_body_name'] == 'splash'
    assert include_arguments['robot_namespace'] == '/splash'
    assert include_arguments['allow_identity_derived_target'] == 'true'
    assert include_arguments['prehook_attitude_reference_mode'] == (
        'recorded_hook'
    )
    assert include_arguments['prehook_reached_orientation_tol_deg'] == '10.0'
    assert include_arguments[
        'prehook_reached_forward_axis_tol_deg'
    ] == '5.0'
    assert include_arguments['prehook_reached_yaw_tol_deg'] == '5.0'
    assert include_arguments['fixed_hook_line_position_mode'] == 'true'
    assert include_arguments['fixed_hook_line_yaw_tol_deg'] == '7.0'
    assert include_arguments['fixed_hook_line_cross_track_tol_m'] == '0.03'
    assert include_arguments[
        'fixed_hook_line_max_reference_lead_m'
    ] == '0.04'
    assert include_arguments[
        'fixed_hook_line_velocity_weight_multiplier'
    ] == '20.0'


def test_latest_splash_prehook_sample_uses_recorded_hook_not_start_trim():
    """Regress the 2026-08-26 stall caused by the wrong attitude target."""
    recorded_hook_wxyz = (
        0.00787666,
        -0.08546681,
        -0.05861794,
        0.99458399,
    )
    captured_start_trim_wxyz = (
        0.01503468,
        -0.05018161,
        0.04059689,
        0.99780141,
    )
    measured_prehook_wxyz = (
        0.024430390108745075,
        -0.06545973478420077,
        -0.05168343263750955,
        0.9962163429456077,
    )

    def normalized(quaternion):
        norm = math.sqrt(sum(value * value for value in quaternion))
        return tuple(value / norm for value in quaternion)

    def angular_error_deg(lhs, rhs):
        lhs = normalized(lhs)
        rhs = normalized(rhs)
        dot = abs(sum(a * b for a, b in zip(lhs, rhs)))
        return math.degrees(2.0 * math.acos(min(1.0, dot)))

    def yaw_rad(quaternion):
        w, x, y, z = normalized(quaternion)
        return math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

    recorded_error_deg = angular_error_deg(
        measured_prehook_wxyz,
        recorded_hook_wxyz,
    )
    obsolete_trim_error_deg = angular_error_deg(
        measured_prehook_wxyz,
        captured_start_trim_wxyz,
    )
    yaw_error_deg = abs(math.degrees(
        math.remainder(
            yaw_rad(measured_prehook_wxyz)
            - yaw_rad(recorded_hook_wxyz),
            2.0 * math.pi,
        )
    ))

    assert recorded_error_deg < 5.0
    assert yaw_error_deg < 3.0
    assert obsolete_trim_error_deg > 8.0


def test_real_splash_derivation_is_archived_and_invalidated_for_control():
    """Retain provenance but reject the disproved cross-rigid-body pose."""
    module = _load_launch_module(
        'fixed_hook_pose_validation.launch.py',
        'fixed_hook_pose_validation_launch_real_derivation',
    )
    source_path = CONFIG_ROOT / GLUB_TARGET_NAME
    manifest_path = CONFIG_ROOT / SPLASH_DERIVED_TARGET_NAME
    source = _load_json(source_path)
    manifest = _load_json(manifest_path)
    derivation = manifest['derived_target']

    assert _sha256(source_path) == GLUB_TARGET_SHA256
    assert derivation['source_config'] == GLUB_TARGET_NAME
    assert derivation['source_config_sha256'] == GLUB_TARGET_SHA256
    assert derivation['source_pose_topic'] == '/mocap/glub_fb/pose'
    assert derivation['destination_pose_topic'] == '/mocap/splash_fb/pose'
    assert derivation['position_offset_m'] == [0.0, 0.0, 0.0]
    assert derivation['orientation_offset_xyzw'] == [0.0, 0.0, 0.0, 1.0]
    assert derivation['operator_confirmed_equivalent_geometry'] is True
    assert derivation['operator_confirmed_same_target_pose'] is False
    assert manifest['status'] == 'invalidated_for_control'
    assert derivation['invalidation_trial_id'] == (
        'retrieval_20260818_190143_613458'
    )
    assert manifest['frame_id'] == source['frame_id']
    assert manifest['message_type'] == source['message_type']
    assert manifest['source_topic'] == source['source_topic']
    assert manifest['target_pose'] == source['target_pose']

    with pytest.raises(RuntimeError, match='invalidated_for_control'):
        module._resolve_target_payload(
            manifest_path,
            '/mocap/splash/pose',
            True,
        )


def test_real_splash_name_rename_manifest_is_hash_locked_and_resolves():
    """Accept only the audited identity transfer for the name-only change."""
    module = _load_launch_module(
        'fixed_hook_pose_validation.launch.py',
        'fixed_hook_pose_validation_launch_real_splash_name_transfer',
    )
    source_path = CONFIG_ROOT / SPLASH_RENAME_SOURCE_NAME
    manifest_path = CONFIG_ROOT / SPLASH_RENAME_MANIFEST_NAME
    source = _load_json(source_path)
    manifest = _load_json(manifest_path)
    derivation = manifest['derived_target']

    assert _sha256(source_path) == SPLASH_RENAME_SOURCE_SHA256
    assert derivation == {
        'schema_version': 1,
        'type': 'identity_pose_transfer',
        'source_config': SPLASH_RENAME_SOURCE_NAME,
        'source_config_sha256': SPLASH_RENAME_SOURCE_SHA256,
        'source_pose_topic': '/mocap/splash_fb/pose',
        'destination_pose_topic': '/mocap/splash/pose',
        'position_offset_m': [0.0, 0.0, 0.0],
        'orientation_offset_xyzw': [0.0, 0.0, 0.0, 1.0],
        'operator_confirmed_equivalent_geometry': True,
        'operator_confirmed_same_target_pose': True,
    }
    for field_name in (
        'frame_id',
        'message_type',
        'source_topic',
        'target_pose',
    ):
        assert manifest[field_name] == source[field_name]

    resolved, provenance = module._resolve_target_payload(
        manifest_path,
        '/mocap/splash/pose',
        True,
    )
    assert resolved == source
    assert provenance == {
        'kind': 'identity_pose_transfer',
        'manifest_config': str(manifest_path.resolve()),
        'source_config': str(source_path.resolve()),
        'source_config_sha256': SPLASH_RENAME_SOURCE_SHA256,
        'source_pose_topic': '/mocap/splash_fb/pose',
        'destination_pose_topic': '/mocap/splash/pose',
        'position_offset_m': [0.0, 0.0, 0.0],
        'orientation_offset_xyzw': [0.0, 0.0, 0.0, 1.0],
        'operator_confirmed_equivalent_geometry': True,
        'operator_confirmed_same_target_pose': True,
    }


def test_splash_launch_accepts_the_audited_name_rename_manifest():
    """Route live Splash topics while retaining the old source provenance."""
    module = _load_launch_module(
        'fixed_hook_pose_validation.launch.py',
        'fixed_hook_pose_validation_launch_splash_name_transfer',
    )
    source = _load_json(CONFIG_ROOT / SPLASH_RENAME_SOURCE_NAME)
    context = _main_launch_context(module)
    context.launch_configurations.update({
        'rigid_body_name': 'splash',
        'robot_namespace': '/splash',
        'target_config': str(CONFIG_ROOT / SPLASH_RENAME_MANIFEST_NAME),
        'allow_identity_derived_target': 'true',
        'target_system_id': '3',
        'target_component_id': '1',
        'robot_type': 'standard',
    })

    parameters = _node_parameters(context, module._launch_setup(context))
    mocap = parameters['mocap_ekf_odom']
    assert mocap['rigid_body_name'] == 'splash'
    assert mocap['pose_topic'] == '/mocap/splash/pose'
    goal = source['target_pose']['position']
    mpc = parameters['mpc_track_trajectory_acados']
    assert mpc['goal_x'] == pytest.approx(goal['x'])
    assert mpc['goal_y'] == pytest.approx(goal['y'])
    assert mpc['goal_z'] == pytest.approx(goal['z'])

    logger_notes = json.loads(
        parameters['payload_retrieval_data_logger']['notes']
    )
    assert logger_notes['target_live_pose_topic'] == '/mocap/splash/pose'
    assert logger_notes['target_source_topic'] == '/mocap/splash_fb/pose'
    assert logger_notes['target_provenance']['kind'] == (
        'identity_pose_transfer'
    )
    assert logger_notes['target_provenance'][
        'source_config_sha256'
    ] == SPLASH_RENAME_SOURCE_SHA256


def test_splash_launch_uses_direct_splash_pose_and_provenance(tmp_path):
    """Use position and orientation measured from the Splash rigid body."""
    module = _load_launch_module(
        'fixed_hook_pose_validation.launch.py',
        'fixed_hook_pose_validation_launch_real_splash_goal',
    )
    target_path = _write_splash_target(tmp_path)
    source = _load_json(target_path)
    context = _main_launch_context(module)
    context.launch_configurations.update({
        'rigid_body_name': 'splash',
        'robot_namespace': '/splash',
        'target_config': str(target_path),
        'allow_identity_derived_target': 'false',
        'target_system_id': '3',
        'target_component_id': '1',
        'robot_type': 'standard',
        'prehook_attitude_reference_mode': 'recorded_hook',
        'prehook_reached_orientation_tol_deg': '10.0',
        'prehook_reached_forward_axis_tol_deg': '5.0',
        'prehook_reached_yaw_tol_deg': '5.0',
        'fixed_hook_line_position_mode': 'true',
        'fixed_hook_line_yaw_tol_deg': '7.0',
        'fixed_hook_line_max_reference_lead_m': '0.04',
        'fixed_hook_line_velocity_weight_multiplier': '20.0',
    })

    parameters = _node_parameters(context, module._launch_setup(context))
    goal = source['target_pose']['position']
    mpc = parameters['mpc_track_trajectory_acados']
    assert mpc['goal_x'] == pytest.approx(goal['x'])
    assert mpc['goal_y'] == pytest.approx(goal['y'])
    assert mpc['goal_z'] == pytest.approx(goal['z'])
    assert mpc['goal_roll'] == pytest.approx(0.12)
    assert mpc['goal_pitch'] == pytest.approx(-0.18)
    assert mpc['goal_yaw'] == pytest.approx(0.31)
    assert mpc['fixed_hook_line_position_mode'] is True

    logger_notes = json.loads(
        parameters['payload_retrieval_data_logger']['notes']
    )
    assert logger_notes['target_live_pose_topic'] == '/mocap/splash/pose'
    assert logger_notes['target_source_topic'] == '/mocap/splash/pose'
    assert logger_notes['target_provenance']['kind'] == (
        'direct_mocap_recording'
    )


def test_identity_derivation_is_rejected_unless_profile_enables_it(tmp_path):
    """Do not let a generic or Glub launch silently accept a derived target."""
    module = _load_launch_module(
        'fixed_hook_pose_validation.launch.py',
        'fixed_hook_pose_validation_launch_derivation_disabled',
    )
    _source_path, manifest_path = _write_derived_fixture(tmp_path)

    with pytest.raises(RuntimeError, match='does not explicitly allow'):
        module._resolve_target_payload(
            manifest_path,
            '/mocap/splash_fb/pose',
            False,
        )


def test_isolated_confirmed_identity_derivation_still_resolves(tmp_path):
    """Keep the generic audited resolver covered outside Splash defaults."""
    module = _load_launch_module(
        'fixed_hook_pose_validation.launch.py',
        'fixed_hook_pose_validation_launch_confirmed_fixture',
    )
    source_path, manifest_path = _write_derived_fixture(tmp_path)

    resolved, provenance = module._resolve_target_payload(
        manifest_path,
        '/mocap/splash_fb/pose',
        True,
    )

    assert resolved == _load_json(source_path)
    assert provenance['kind'] == 'identity_pose_transfer'
    assert provenance['source_config_sha256'] == GLUB_TARGET_SHA256


@pytest.mark.parametrize(
    ('field_name', 'replacement', 'expected_error'),
    [
        (
            'source_config_sha256',
            '0' * 64,
            'SHA-256 mismatch',
        ),
        (
            'destination_pose_topic',
            '/mocap/not_splash/pose',
            'destination_pose_topic must be /mocap/splash_fb/pose',
        ),
        (
            'position_offset_m',
            [0.001, 0.0, 0.0],
            'position_offset_m must be the exact identity',
        ),
        (
            'orientation_offset_xyzw',
            [0.0, 0.0, 1.0, 0.0],
            'orientation_offset_xyzw must be the exact identity',
        ),
        (
            'operator_confirmed_equivalent_geometry',
            False,
            'operator confirmation',
        ),
        (
            'operator_confirmed_same_target_pose',
            False,
            'operator confirmation',
        ),
    ],
)
def test_identity_derivation_rejects_critical_metadata_tampering(
    tmp_path,
    field_name,
    replacement,
    expected_error,
):
    """Reject changed hashes, routing, transforms, or confirmations."""
    module = _load_launch_module(
        'fixed_hook_pose_validation.launch.py',
        f'fixed_hook_pose_validation_launch_tampered_{field_name}',
    )
    _, manifest_path = _write_derived_fixture(tmp_path)
    manifest = _load_json(manifest_path)
    manifest['derived_target'][field_name] = replacement
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')

    with pytest.raises(RuntimeError, match=expected_error):
        module._resolve_target_payload(
            manifest_path,
            '/mocap/splash_fb/pose',
            True,
        )


@pytest.mark.parametrize(
    ('field_name', 'replacement'),
    [
        ('frame_id', 'wrong_frame'),
        ('message_type', 'nav_msgs/Odometry'),
        ('source_topic', '/mocap/not_glub/pose'),
        (
            'target_pose',
            {
                'position': {'x': 8.0, 'y': 0.0, 'z': 1.0},
                'orientation_xyzw': {
                    'x': 0.0,
                    'y': 0.0,
                    'z': 0.0,
                    'w': 1.0,
                },
            },
        ),
    ],
)
def test_identity_derivation_rejects_snapshot_tampering(
    tmp_path,
    field_name,
    replacement,
):
    """The human-readable snapshot must exactly match the hashed source."""
    module = _load_launch_module(
        'fixed_hook_pose_validation.launch.py',
        f'fixed_hook_pose_validation_launch_snapshot_{field_name}',
    )
    _, manifest_path = _write_derived_fixture(tmp_path)
    manifest = _load_json(manifest_path)
    manifest[field_name] = replacement
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')

    with pytest.raises(
        RuntimeError,
        match=rf'derived target {field_name} snapshot does not exactly match',
    ):
        module._resolve_target_payload(
            manifest_path,
            '/mocap/splash_fb/pose',
            True,
        )


def test_identity_derivation_rejects_source_file_modification(tmp_path):
    """Changing even a validated source recording invalidates its manifest."""
    module = _load_launch_module(
        'fixed_hook_pose_validation.launch.py',
        'fixed_hook_pose_validation_launch_modified_source',
    )
    source_path, manifest_path = _write_derived_fixture(tmp_path)
    source_path.write_bytes(source_path.read_bytes() + b'\n')

    with pytest.raises(RuntimeError, match='SHA-256 mismatch'):
        module._resolve_target_payload(
            manifest_path,
            '/mocap/splash_fb/pose',
            True,
        )


def test_identity_derivation_rejects_a_derived_source_chain(tmp_path):
    """Require a real recorder output as the root, never another manifest."""
    module = _load_launch_module(
        'fixed_hook_pose_validation.launch.py',
        'fixed_hook_pose_validation_launch_derived_chain',
    )
    source_path, manifest_path = _write_derived_fixture(tmp_path)
    source = _load_json(source_path)
    source['derived_target'] = {
        'schema_version': 1,
        'type': 'identity_pose_transfer',
    }
    source_path.write_text(json.dumps(source), encoding='utf-8')
    manifest = _load_json(manifest_path)
    manifest['derived_target']['source_config_sha256'] = _sha256(source_path)
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')

    with pytest.raises(
        RuntimeError,
        match='derived target chains are not allowed',
    ):
        module._resolve_target_payload(
            manifest_path,
            '/mocap/splash_fb/pose',
            True,
        )


def test_direct_splash_recording_resolution_remains_supported(tmp_path):
    """Keep direct Splash recordings valid with derivation disabled."""
    module = _load_launch_module(
        'fixed_hook_pose_validation.launch.py',
        'fixed_hook_pose_validation_launch_direct_splash_regression',
    )
    direct_path = _write_splash_target(tmp_path)
    expected = _load_json(direct_path)

    resolved, provenance = module._resolve_target_payload(
        direct_path,
        '/mocap/splash/pose',
        False,
    )
    assert resolved == expected
    assert provenance['kind'] == 'direct_mocap_recording'
    assert provenance['source_pose_topic'] == '/mocap/splash/pose'
    assert provenance['destination_pose_topic'] == '/mocap/splash/pose'


def test_current_splash_profile_rejects_former_splash_fb_recording(tmp_path):
    """Require a new recording instead of relabelling former provenance."""
    module = _load_launch_module(
        'fixed_hook_pose_validation.launch.py',
        'fixed_hook_pose_validation_launch_reject_former_splash_name',
    )
    target_path = _write_splash_target(tmp_path)
    payload = _load_json(target_path)
    payload['source_topic'] = '/mocap/splash_fb/pose'
    target_path.write_text(json.dumps(payload), encoding='utf-8')

    context = _main_launch_context(module)
    context.launch_configurations.update({
        'rigid_body_name': 'splash',
        'robot_namespace': '/splash',
        'target_config': str(target_path),
        'target_system_id': '3',
        'target_component_id': '1',
        'robot_type': 'standard',
    })

    with pytest.raises(
        RuntimeError,
        match='target source_topic must be /mocap/splash/pose',
    ):
        module._launch_setup(context)


def test_splash_profile_routes_every_real_topic_to_splash(
    tmp_path,
):
    """Route MoCap, PX4 feedback, commands, and IDs to Splash only."""
    module = _load_launch_module(
        'fixed_hook_pose_validation.launch.py',
        'fixed_hook_pose_validation_launch_for_splash',
    )
    target_path = _write_splash_target(tmp_path)
    context = _main_launch_context(module)
    context.launch_configurations.update({
        'rigid_body_name': 'splash',
        'robot_namespace': '/splash',
        'target_config': str(target_path),
        'target_system_id': '3',
        'target_component_id': '1',
        'robot_type': 'standard',
        'prehook_attitude_reference_mode': 'recorded_hook',
        'prehook_reached_orientation_tol_deg': '10.0',
        'prehook_reached_forward_axis_tol_deg': '5.0',
        'prehook_reached_yaw_tol_deg': '5.0',
        'fixed_hook_line_position_mode': 'true',
        'fixed_hook_line_yaw_tol_deg': '7.0',
        'fixed_hook_line_max_reference_lead_m': '0.04',
        'fixed_hook_line_velocity_weight_multiplier': '20.0',
    })

    parameters = _node_parameters(context, module._launch_setup(context))
    assert set(parameters) == {
        'mocap_ekf_odom',
        'nav_odom_to_vehicle_odometry',
        'mpc_track_trajectory_acados',
        'payload_retrieval_data_logger',
        'offboard_enable',
    }

    mocap = parameters['mocap_ekf_odom']
    assert mocap['rigid_body_name'] == 'splash'
    assert mocap['pose_topic'] == '/mocap/splash/pose'
    assert mocap['odom_topic'] == '/mocap/splash/odom_ekf_fixed_hook'

    adapter = parameters['nav_odom_to_vehicle_odometry']
    assert adapter['input_odom_topic'] == (
        '/mocap/splash/odom_ekf_fixed_hook'
    )
    assert adapter['output_vehicle_odometry_topic'] == (
        '/mocap/splash/vehicle_odometry_fixed_hook'
    )
    assert adapter['angular_velocity_override_topic'] == (
        '/splash/fmu/out/vehicle_odometry'
    )

    mpc = parameters['mpc_track_trajectory_acados']
    assert mpc['prehook_attitude_alignment_timeout_s'] == pytest.approx(0.0)
    assert mpc['prehook_attitude_reference_mode'] == 'recorded_hook'
    assert mpc['prehook_reached_orientation_tol_rad'] == pytest.approx(
        math.radians(10.0)
    )
    assert mpc['prehook_reached_forward_axis_tol_rad'] == pytest.approx(
        math.radians(5.0)
    )
    assert mpc['prehook_reached_yaw_tol_rad'] == pytest.approx(
        math.radians(5.0)
    )
    assert mpc['goal_reached_orientation_tol_rad'] == pytest.approx(
        math.radians(5.0)
    )
    assert mpc['fixed_hook_line_position_mode'] is True
    assert mpc['fixed_hook_line_yaw_tol_rad'] == pytest.approx(
        math.radians(7.0)
    )
    assert mpc['fixed_hook_line_cross_track_tol_m'] == pytest.approx(0.03)
    assert mpc['fixed_hook_line_max_reference_lead_m'] == pytest.approx(0.04)
    assert mpc[
        'fixed_hook_line_velocity_weight_multiplier'
    ] == pytest.approx(20.0)
    assert mpc[
        'prehook_attitude_wait_exit_hysteresis_ratio'
    ] == pytest.approx(1.5)
    assert mpc['require_operator_hook_confirmation'] is True
    assert mpc['hook_confirmation_service'] == (
        '/bluerov2/fixed_hook/confirm_hook'
    )
    assert mpc['hook_confirmation_min_wait_s'] == pytest.approx(0.25)
    assert mpc['control_mode_topic'] == (
        '/splash/fmu/out/vehicle_control_mode'
    )
    assert mpc['thrust_sp_topic'] == (
        '/splash/fmu/in/vehicle_thrust_setpoint'
    )
    assert mpc['torque_sp_topic'] == (
        '/splash/fmu/in/vehicle_torque_setpoint'
    )

    logger_notes = json.loads(
        parameters['payload_retrieval_data_logger']['notes']
    )
    prehook_metadata = logger_notes['prehook_planner']
    assert prehook_metadata['attitude_reference_mode'] == (
        'recorded_hook'
    )
    assert prehook_metadata[
        'reached_orientation_tolerance_deg'
    ] == pytest.approx(10.0)
    assert prehook_metadata[
        'reached_forward_axis_tolerance_deg'
    ] == pytest.approx(5.0)
    assert prehook_metadata['attitude_gate_semantics'] == (
        'full_attitude_safety_envelope_and_body_x_forward_axis_and_yaw'
    )
    assert prehook_metadata['reached_yaw_tolerance_deg'] == pytest.approx(5.0)
    assert prehook_metadata[
        'attitude_wait_exit_hysteresis_ratio'
    ] == pytest.approx(1.5)
    assert prehook_metadata[
        'attitude_wait_exit_hysteresis_semantics'
    ] == 'position_depth_exit_gate_ratio'
    assert prehook_metadata[
        'attitude_alignment_timeout_enabled'
    ] is False
    assert prehook_metadata['attitude_timeout_action'] == (
        'continuous_nmpc_alignment_no_elapsed_timeout'
    )
    assert prehook_metadata[
        'attitude_timeout_integral_semantics'
    ] == 'not_applicable_no_timeout_transition'
    line_interlock = logger_notes['fixed_hook_line_interlock']
    assert line_interlock['enabled'] is False
    assert line_interlock['control_mode'] == (
        'position_like_time_parameterized_ned'
    )
    assert line_interlock['active_states'] == []
    assert line_interlock['progress_source'] == 'monotonic_trajectory_time'
    assert line_interlock['progress_semantics'] == (
        'direct_to_endpoint_no_freeze_no_rewind'
    )
    assert line_interlock['yaw_tolerance_deg'] == pytest.approx(7.0)
    assert line_interlock['cross_track_tolerance_m'] == pytest.approx(0.03)
    assert line_interlock['depth_tolerance_m'] == pytest.approx(0.03)
    assert line_interlock['maximum_reference_lead_m'] is None
    assert line_interlock['maximum_reference_lead_semantics'] == (
        'not_used_by_time_parameterized_position_mode'
    )
    assert line_interlock[
        'velocity_weight_multiplier'
    ] == pytest.approx(20.0)
    assert line_interlock['velocity_weight_scope'] == (
        'GO_FORWARD_GO_BACK_world_ned_velocity'
    )
    assert line_interlock['velocity_reference_frame'] == 'ned'
    assert line_interlock['depth_velocity_reference_mps'] == pytest.approx(0.0)
    assert line_interlock[
        'interlocked_velocity_weight_multiplier'
    ] is None
    assert line_interlock[
        'interlocked_velocity_weight_semantics'
    ] == 'not_applicable_interlock_disabled'
    assert line_interlock['outside_tolerance_action'] == (
        'continue_translation_and_correct_all_axes_with_nmpc'
    )
    assert line_interlock['backslide_action'] == (
        'time_reference_continues_toward_endpoint_never_rewinds'
    )
    assert line_interlock['cross_track_integral_semantics'] == (
        'bounded_line_normal_only_during_transit'
    )
    assert line_interlock['along_track_integral_semantics'] == (
        'always_zero_during_transit'
    )

    offboard = parameters['offboard_enable']
    assert offboard['offboard_mode_topic'] == (
        '/splash/fmu/in/offboard_control_mode'
    )
    assert offboard['vehicle_cmd_topic'] == '/splash/fmu/in/vehicle_command'
    assert offboard['vehicle_control_mode_topic'] == (
        '/splash/fmu/out/vehicle_control_mode'
    )
    assert offboard['target_system_id'] == 3
    assert offboard['target_component_id'] == 1

    all_string_values = [
        value
        for node_parameters in parameters.values()
        for value in node_parameters.values()
        if isinstance(value, str)
    ]
    assert not any(value.startswith('/glub/') for value in all_string_values)


def test_splash_profile_rejects_glub_target_provenance():
    """Never relabel or reuse a Glub target for the Splash rigid body."""
    module = _load_launch_module(
        'fixed_hook_pose_validation.launch.py',
        'fixed_hook_pose_validation_launch_reject_glub',
    )
    context = _main_launch_context(module)
    context.launch_configurations.update({
        'rigid_body_name': 'splash',
        'robot_namespace': '/splash',
        'target_config': str(CONFIG_ROOT / GLUB_TARGET_NAME),
        'target_system_id': '3',
        'target_component_id': '1',
        'robot_type': 'standard',
    })

    with pytest.raises(
        RuntimeError,
        match='target source_topic must be /mocap/splash/pose',
    ):
        module._launch_setup(context)


def test_unconfigured_target_has_clear_fail_closed_error():
    """Explain a missing measured target before any nodes are started."""
    module = _load_launch_module(
        'fixed_hook_pose_validation.launch.py',
        'fixed_hook_pose_validation_launch_unconfigured',
    )
    context = _main_launch_context(module)
    context.launch_configurations['target_config'] = 'unconfigured'

    with pytest.raises(RuntimeError, match='target_config is required'):
        module._launch_setup(context)
