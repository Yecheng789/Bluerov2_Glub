"""Focused tests for the guarded fixed-hook reality path."""

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from launch import LaunchContext
from launch_ros.actions import Node as LaunchNode
from std_msgs.msg import Bool

from bluerov2_control.calibrate_mocap_orientation_correction import (
    normalize_quat as normalize_calibration_quat,
)
from bluerov2_control.mpc_track_trajectory_acados import (
    MPCTrackTrajectoryAcados,
    _validated_operating_bounds,
    euler_to_quat_wxyz,
    quat_angular_distance_wxyz,
)
from bluerov2_control.offboard_enable import _build_ack_result_names
from bluerov2_control.offboard_enable import DEFAULT_CONTROL_MODE_TIMEOUT_SEC
from bluerov2_control.offboard_enable import OffboardEnableNode
from bluerov2_control.offboard_enable import versioned_px4_topic
from px4_msgs.msg import VehicleCommandAck, VehicleOdometry


def _load_launch_module():
    path = (
        Path(__file__).resolve().parents[1]
        / 'launch'
        / 'fixed_hook_pose_validation.launch.py'
    )
    spec = importlib.util.spec_from_file_location(
        'fixed_hook_pose_validation_launch', path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message, **_kwargs):
        self.messages.append(('info', message))

    def warning(self, message, **_kwargs):
        self.messages.append(('warning', message))

    def error(self, message, **_kwargs):
        self.messages.append(('error', message))


def test_nwu_flu_raw_target_matches_live_ned_frd_conversion():
    module = _load_launch_module()
    payload = {
        'message_type': 'geometry_msgs/PoseStamped',
        'target_pose': {
            'position': {'x': 1.0, 'y': 2.0, 'z': 0.5},
            'orientation_xyzw': {
                'x': 0.0,
                'y': 0.0,
                'z': 0.0,
                'w': 1.0,
            },
        },
    }
    position, quaternion, applied = module._target_pose_in_ned_frd(
        payload,
        'nwu',
        'flu',
        np.array([0.0, 0.0, 0.0, 1.0]),
        'auto',
    )

    np.testing.assert_allclose(position, [1.0, -2.0, -0.5])
    np.testing.assert_allclose(quaternion, [1.0, 0.0, 0.0, 0.0])
    assert applied is True


def test_target_quaternion_validation_is_fail_closed():
    module = _load_launch_module()
    with pytest.raises(RuntimeError, match='near-zero'):
        module._strict_quaternion_xyzw('0 0 0 0', 'target')
    with pytest.raises(RuntimeError, match='finite'):
        module._strict_quaternion_xyzw('nan 0 0 1', 'target')


def _recording_validation_context():
    context = LaunchContext()
    context.launch_configurations.update({
        'max_raw_mocap_message_age_sec': '0.20',
        'target_min_samples': '80',
        'target_min_sampling_span_sec': '0.75',
        'target_max_position_std_m': '0.015',
        'target_max_orientation_std_deg': '1.5',
        'target_max_orientation_spread_deg': '5.0',
    })
    return context


def _validated_recording_payload(recorder_age_limit=0.20):
    return {
        'validation': {
            'passed': True,
            'settings': {
                'max_message_age_sec': recorder_age_limit,
            },
            'metrics': {
                'accepted_sample_count': 80,
                'header_timestamped_sample_count': 80,
                'distinct_header_timestamp_count': 80,
                'sampling_span_sec': 1.0,
                'message_age_sec': {
                    'min': 0.01,
                    'mean': 0.02,
                    'max': 0.03,
                },
                'max_axis_position_standard_deviation_m': 0.005,
                'orientation': {
                    'angular_std_deg': 0.5,
                    'pairwise_spread_deg': 1.0,
                },
            },
        },
    }


def test_target_and_live_mocap_age_limits_must_match():
    module = _load_launch_module()
    context = _recording_validation_context()
    payload = _validated_recording_payload()
    metrics = module._validate_recording(payload, context)
    assert metrics['message_age_sec']['max'] == pytest.approx(0.03)

    stale_policy_payload = _validated_recording_payload(
        recorder_age_limit=0.50
    )
    with pytest.raises(RuntimeError, match='incompatible'):
        module._validate_recording(stale_policy_payload, context)


@pytest.mark.parametrize(
    ('source_topic', 'expected_error'),
    [
        ('/mocap/glub_fb/pose', None),
        (
            '/mocap/glub_4/pose',
            'target source_topic must be /mocap/glub_fb/pose',
        ),
        (
            '/mocap/glub/pose',
            'target source_topic must be /mocap/glub_fb/pose',
        ),
    ],
)
def test_fixed_hook_launch_accepts_only_current_rigid_body_target(
    tmp_path, source_topic, expected_error
):
    module = _load_launch_module()
    payload = _validated_recording_payload()
    payload.update({
        'frame_id': 'qualisys_world',
        'message_type': 'geometry_msgs/PoseStamped',
        'source_topic': source_topic,
        'target_pose': {
            'position': {'x': 1.0, 'y': 0.0, 'z': 0.5},
            'orientation_xyzw': {
                'x': 0.0,
                'y': 0.0,
                'z': 0.0,
                'w': 1.0,
            },
        },
    })
    target_path = tmp_path / 'validated_target.json'
    target_path.write_text(json.dumps(payload), encoding='utf-8')

    context = LaunchContext()
    for entity in module.generate_launch_description().entities:
        if getattr(entity, 'name', None) is not None:
            entity.execute(context)
    assert context.launch_configurations['rigid_body_name'] == 'glub_fb'
    assert context.launch_configurations['robot_namespace'] == '/glub'
    assert context.launch_configurations['target_config'].endswith(
        'hooked_box_target_pose_20260802_195146.json'
    )
    assert context.launch_configurations['mocap_world_frame'] == 'ned'
    assert (
        context.launch_configurations['pool_bounds_mocap']
        == '0 9 -2.5 2.5 0 3'
    )
    assert context.launch_configurations['mocap_body_frame'] == 'frd'
    assert context.launch_configurations['robot_type'] == 'standard'
    assert context.launch_configurations['target_system_id'] == '3'
    assert context.launch_configurations['target_component_id'] == '1'
    assert context.launch_configurations['pre_approach_distance_m'] == '0.50'
    assert (
        context.launch_configurations['fixed_hook_depth_tolerance_m']
        == '0.03'
    )
    assert context.launch_configurations['traj_speed_mps'] == '0.05'
    assert context.launch_configurations['pre_approach_speed_mps'] == '0.06'
    assert context.launch_configurations['final_approach_speed_mps'] == '0.04'
    assert context.launch_configurations['min_traj_duration_s'] == '5.0'
    assert context.launch_configurations['traj_angular_speed_deg_s'] == '8.0'
    assert context.launch_configurations['final_pose_hold_s'] == '5.0'
    assert context.launch_configurations['retreat_speed_mps'] == '0.04'
    assert (
        context.launch_configurations['px4_angular_velocity_timeout_sec']
        == '0.10'
    )
    assert context.launch_configurations['w_att'] == '10.0'
    assert context.launch_configurations['w_omega'] == '20.0'
    assert context.launch_configurations['w_u_torque'] == '0.5'
    assert (
        context.launch_configurations[
            'position_integral_gain_N_per_m_s'
        ]
        == '3.0'
    )
    assert (
        context.launch_configurations[
            'position_integral_force_limit_fraction'
        ]
        == '0.07'
    )
    assert (
        context.launch_configurations[
            'position_integral_activation_error_m'
        ]
        == '0.50'
    )
    assert context.launch_configurations['max_mocap_coast_sec'] == '2.0'
    assert context.launch_configurations['max_initial_goal_distance_m'] == '0'
    assert (
        context.launch_configurations[
            'max_initial_goal_orientation_error_deg'
        ]
        == '0'
    )
    context.launch_configurations.update({
        'target_config': str(target_path),
        'mocap_world_frame': 'ned',
        'pool_bounds_mocap': '-4.5 4.5 -2.5 2.5 -1.5 1.5',
        'mocap_body_frame': 'frd',
        'robot_type': 'standard',
        'target_system_id': '1',
        'target_component_id': '1',
    })

    if expected_error is not None:
        with pytest.raises(RuntimeError, match=expected_error):
            module._launch_setup(context)
        return

    actions = module._launch_setup(context)
    nodes = [action for action in actions if isinstance(action, LaunchNode)]
    assert len(nodes) == 5


def test_fixed_hook_compatibility_wrapper_uses_current_real_defaults():
    path = (
        Path(__file__).resolve().parents[1]
        / 'launch'
        / 'fixed_hook_mpc_june23.launch.py'
    )
    spec = importlib.util.spec_from_file_location(
        'fixed_hook_mpc_june23_launch', path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.DEFAULTS['rigid_body_name'] == 'glub_fb'
    assert module.DEFAULTS['robot_namespace'] == '/glub'
    assert module.DEFAULTS['target_config'].endswith(
        'hooked_box_target_pose_20260802_195146.json'
    )
    assert module.DEFAULTS['mocap_world_frame'] == 'ned'
    assert module.DEFAULTS['mocap_body_frame'] == 'frd'
    assert module.DEFAULTS['robot_type'] == 'standard'
    assert module.DEFAULTS['pool_bounds_mocap'] == '0 9 -2.5 2.5 0 3'
    assert module.DEFAULTS['target_system_id'] == '3'
    assert module.DEFAULTS['target_component_id'] == '1'
    assert module.DEFAULTS['pre_approach_distance_m'] == '0.50'
    assert module.DEFAULTS['fixed_hook_depth_tolerance_m'] == '0.03'
    assert module.DEFAULTS['traj_speed_mps'] == '0.05'
    assert module.DEFAULTS['pre_approach_speed_mps'] == '0.06'
    assert module.DEFAULTS['final_approach_speed_mps'] == '0.04'
    assert module.DEFAULTS['min_traj_duration_s'] == '5.0'
    assert module.DEFAULTS['traj_angular_speed_deg_s'] == '8.0'
    assert module.DEFAULTS['final_pose_hold_s'] == '5.0'
    assert module.DEFAULTS['retreat_speed_mps'] == '0.04'
    assert module.DEFAULTS['px4_angular_velocity_timeout_sec'] == '0.10'
    assert module.DEFAULTS['w_att'] == '10.0'
    assert module.DEFAULTS['w_omega'] == '20.0'
    assert module.DEFAULTS['w_u_torque'] == '0.5'
    assert module.DEFAULTS['position_integral_gain_N_per_m_s'] == '3.0'
    assert (
        module.DEFAULTS['position_integral_force_limit_fraction']
        == '0.07'
    )
    assert (
        module.DEFAULTS['position_integral_activation_error_m']
        == '0.50'
    )
    assert module.DEFAULTS['thrust_sat'] == '0.12'
    assert module.DEFAULTS['torque_sat'] == '0.02'
    assert module.DEFAULTS['max_mocap_coast_sec'] == '2.0'
    assert module.DEFAULTS['max_initial_goal_distance_m'] == '0'
    assert module.DEFAULTS['max_initial_goal_orientation_error_deg'] == '0'
    assert 'pre_approach_distance_m' in module.FORWARDED_ARGUMENTS
    assert 'fixed_hook_depth_tolerance_m' in module.FORWARDED_ARGUMENTS
    assert 'pre_approach_speed_mps' in module.FORWARDED_ARGUMENTS
    assert 'final_approach_speed_mps' in module.FORWARDED_ARGUMENTS
    assert 'traj_angular_speed_deg_s' in module.FORWARDED_ARGUMENTS
    assert 'final_pose_hold_s' in module.FORWARDED_ARGUMENTS
    assert 'retreat_speed_mps' in module.FORWARDED_ARGUMENTS
    assert 'px4_angular_velocity_timeout_sec' in module.FORWARDED_ARGUMENTS
    assert (
        'position_integral_gain_N_per_m_s'
        in module.FORWARDED_ARGUMENTS
    )
    assert (
        'position_integral_force_limit_fraction'
        in module.FORWARDED_ARGUMENTS
    )
    assert (
        'position_integral_activation_error_m'
        in module.FORWARDED_ARGUMENTS
    )
    assert 'max_mocap_coast_sec' in module.FORWARDED_ARGUMENTS


def test_new_recorded_target_builds_level_half_metre_pre_approach():
    module = _load_launch_module()
    target_path = (
        Path(__file__).resolve().parents[1]
        / 'experiments'
        / 'payload_retrieval'
        / 'config'
        / 'hooked_box_target_pose_20260802_195146.json'
    )
    payload = json.loads(target_path.read_text(encoding='utf-8'))

    target_position, target_quaternion, _applied = (
        module._target_pose_in_ned_frd(
            payload,
            'ned',
            'frd',
            np.array([0.0, 0.0, 0.0, 1.0]),
            'auto',
        )
    )
    _roll, _pitch, goal_yaw = module._quat_wxyz_to_rpy(
        target_quaternion
    )
    horizontal_forward = np.array(
        [np.cos(goal_yaw), np.sin(goal_yaw), 0.0],
        dtype=float,
    )
    pre_approach = target_position - 0.50 * horizontal_forward
    pre_approach[2] = target_position[2]

    np.testing.assert_allclose(
        pre_approach,
        [4.84480064, 0.07768497, 1.72805171],
        atol=1e-7,
    )
    assert pre_approach[2] == pytest.approx(target_position[2])


def test_launch_rejects_pre_approach_waypoint_outside_pool(tmp_path):
    module = _load_launch_module()
    payload = _validated_recording_payload()
    payload.update({
        'frame_id': 'mocap',
        'message_type': 'geometry_msgs/PoseStamped',
        'source_topic': '/mocap/glub_fb/pose',
        'target_pose': {
            # The final target is inside the 0.25 m shrunken pool, but the
            # default 0.50 m waypoint behind a zero-yaw body is outside it.
            'position': {'x': 0.40, 'y': 0.0, 'z': 1.0},
            'orientation_xyzw': {
                'x': 0.0,
                'y': 0.0,
                'z': 0.0,
                'w': 1.0,
            },
        },
    })
    target_path = tmp_path / 'target_with_unsafe_pre_approach.json'
    target_path.write_text(json.dumps(payload), encoding='utf-8')

    context = LaunchContext()
    for entity in module.generate_launch_description().entities:
        if getattr(entity, 'name', None) is not None:
            entity.execute(context)
    context.launch_configurations['target_config'] = str(target_path)

    with pytest.raises(
        RuntimeError,
        match='converted pre-approach position.*outside',
    ):
        module._launch_setup(context)


def test_pool_bounds_convert_all_enu_corners_and_shrink_in_ned():
    module = _load_launch_module()
    bounds = module._pool_bounds_to_ned(
        '-2 3 -4 5 -1 2',
        'enu',
        0.25,
    )

    np.testing.assert_allclose(bounds['raw_min'], [-2.0, -4.0, -1.0])
    np.testing.assert_allclose(bounds['raw_max'], [3.0, 5.0, 2.0])
    np.testing.assert_allclose(bounds['ned_min'], [-4.0, -2.0, -2.0])
    np.testing.assert_allclose(bounds['ned_max'], [5.0, 3.0, 1.0])
    np.testing.assert_allclose(
        bounds['operating_min_ned'], [-3.75, -1.75, -1.75]
    )
    np.testing.assert_allclose(
        bounds['operating_max_ned'], [4.75, 2.75, 0.75]
    )


@pytest.mark.parametrize(
    ('bounds_text', 'margin', 'match'),
    [
        ('unconfigured', 0.25, 'required'),
        ('0 1 0 1 0 nan', 0.25, 'finite'),
        ('0 0 0 1 0 1', 0.25, 'xmin < xmax'),
        ('0 1 0 1 0 1', 0.50, 'no valid NED operating volume'),
    ],
)
def test_pool_bounds_configuration_is_fail_closed(
    bounds_text, margin, match
):
    module = _load_launch_module()
    with pytest.raises(RuntimeError, match=match):
        module._pool_bounds_to_ned(bounds_text, 'nwu', margin)


def test_fixed_target_must_be_inside_margin_shrunk_pool_bounds():
    module = _load_launch_module()
    minimum = np.array([-1.0, -2.0, 0.25])
    maximum = np.array([2.0, 3.0, 1.5])

    # The closed boundary itself is permitted.
    module._require_point_inside_bounds(
        np.array([-1.0, 3.0, 1.5]),
        minimum,
        maximum,
        'converted target position',
    )
    with pytest.raises(RuntimeError, match='outside'):
        module._require_point_inside_bounds(
            np.array([-1.001, 0.0, 1.0]),
            minimum,
            maximum,
            'converted target position',
        )


def test_tracker_operating_bounds_require_finite_ordered_limits():
    minimum, maximum = _validated_operating_bounds(
        -1.0, 2.0, -3.0, 4.0, 0.25, 1.5
    )
    np.testing.assert_allclose(minimum, [-1.0, -3.0, 0.25])
    np.testing.assert_allclose(maximum, [2.0, 4.0, 1.5])

    with pytest.raises(ValueError, match='finite'):
        _validated_operating_bounds(-1.0, np.inf, -3.0, 4.0, 0.0, 1.0)
    with pytest.raises(ValueError, match='xmin < xmax'):
        _validated_operating_bounds(1.0, 1.0, -3.0, 4.0, 0.0, 1.0)


def test_valid_out_of_bounds_odom_zeroes_and_latches_mission_preflight():
    logger = _Logger()
    invalidations = []
    parameters = {
        'require_expected_odom_frames': False,
        'min_odom_quality': 0,
        'require_increasing_odom_timestamp': False,
        'max_tilt_rad': 0.0,
        'require_mission_enable': True,
        'revoke_mission_on_state_failure': False,
    }
    fake = SimpleNamespace(
        operating_bounds_enabled=True,
        operating_bounds_min_ned=np.array([-1.0, -1.0, -1.0]),
        operating_bounds_max_ned=np.array([1.0, 1.0, 1.0]),
        have_odom=True,
        odom_valid=True,
        mission_enable=False,
        mission_rearm_required=False,
        last_odom_timestamp_sample=None,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        get_logger=lambda: logger,
        _invalidate_command=lambda **kwargs: invalidations.append(kwargs),
    )
    fake._mission_allowed = lambda: MPCTrackTrajectoryAcados._mission_allowed(
        fake
    )
    fake._handle_state_failure = lambda reason: (
        MPCTrackTrajectoryAcados._handle_state_failure(fake, reason)
    )
    fake._revoke_mission_enable = lambda reason: (
        MPCTrackTrajectoryAcados._revoke_mission_enable(fake, reason)
    )
    fake._reject_odom = lambda reason, **kwargs: (
        MPCTrackTrajectoryAcados._reject_odom(fake, reason, **kwargs)
    )

    odom = VehicleOdometry()
    odom.position = [1.001, 0.0, 0.0]
    odom.q = [1.0, 0.0, 0.0, 0.0]
    odom.velocity = [0.0, 0.0, 0.0]
    odom.angular_velocity = [0.0, 0.0, 0.0]

    MPCTrackTrajectoryAcados.on_odom(fake, odom)

    assert fake.have_odom is False
    assert fake.odom_valid is False
    assert fake.mission_enable is False
    assert fake.mission_rearm_required is True
    assert invalidations == [
        {'reset_trajectory': True, 'publish_zero': True}
    ]
    assert any('outside the configured operating bounds' in message
               for _level, message in logger.messages)


def test_orientation_calibration_rejects_invalid_quaternions():
    with pytest.raises(ValueError, match='too small'):
        normalize_calibration_quat((0.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match='finite'):
        normalize_calibration_quat((0.0, 0.0, float('nan'), 1.0))


def test_missing_px4_feedback_cannot_issue_heartbeat_or_mode_command():
    calls = []
    fake = SimpleNamespace(
        enforce_control_publisher_exclusivity=False,
        _controller_is_alive=lambda _now: True,
        _control_mode_feedback_is_fresh=lambda _now: False,
        _withdraw_heartbeat_for_health_failure=(
            lambda reason: calls.append(('withdraw', reason))
        ),
        publish_offboard_control_mode=lambda: calls.append(('heartbeat', None)),
        _request_offboard=lambda _now: calls.append(('request', None)),
    )

    OffboardEnableNode.timer_callback(fake)

    assert [kind for kind, _detail in calls] == ['withdraw']


def test_controller_health_failure_requires_false_then_true():
    logger = _Logger()
    fake = SimpleNamespace(
        # Even an early request sent before health prestream begins must latch.
        first_heartbeat_time=None,
        require_request_enable=True,
        request_enabled=True,
        offboard_active=False,
        controller_rearm_required=False,
        offboard_ack_accepted=True,
        arm_accepted=True,
        manual_arm_notice_logged=True,
        last_offboard_command_time=10.0,
        get_logger=lambda: logger,
    )

    OffboardEnableNode._withdraw_heartbeat_for_health_failure(
        fake, 'test failure'
    )
    assert fake.request_enabled is False
    assert fake.controller_rearm_required is True
    assert fake.first_heartbeat_time is None

    enable = Bool(data=True)
    OffboardEnableNode.request_enable_callback(fake, enable)
    assert fake.request_enabled is False
    assert fake.controller_rearm_required is True

    disable = Bool(data=False)
    OffboardEnableNode.request_enable_callback(fake, disable)
    assert fake.controller_rearm_required is False

    OffboardEnableNode.request_enable_callback(fake, enable)
    assert fake.request_enabled is True
    assert fake.last_offboard_command_time is None


def test_terminal_hold_is_stable_and_changed_goal_rebuilds():
    logger = _Logger()
    old_signature = (1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0)
    signature = [old_signature]
    rebuilds = []

    fake = SimpleNamespace(
        have_odom=True,
        mission_state='INIT',
        traj_active=False,
        last_goal_signature=old_signature,
        terminal_hold_goal_signature=old_signature,
        _update_mission=lambda: None,
        _update_fixed_hook_hold_phase=lambda: False,
        _goal_signature=lambda: signature[0],
        get_parameter=lambda name: SimpleNamespace(
            value={'regenerate_on_goal_change': False}[name]
        ),
        get_logger=lambda: logger,
    )

    def rebuild():
        rebuilds.append(signature[0])
        fake.traj_active = True
        fake.last_goal_signature = signature[0]
        fake.terminal_hold_goal_signature = None

    fake._reset_trajectory_from_current_pose = rebuild

    MPCTrackTrajectoryAcados._maybe_refresh_trajectory(fake)
    assert rebuilds == []
    assert fake.traj_active is False
    assert fake.terminal_hold_goal_signature == old_signature

    new_signature = (1.1, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0)
    signature[0] = new_signature
    MPCTrackTrajectoryAcados._maybe_refresh_trajectory(fake)
    assert rebuilds == [new_signature]
    assert fake.traj_active is True
    assert fake.terminal_hold_goal_signature is None


def test_terminal_hold_requires_attitude_when_hold_attitude_is_enabled():
    logger = _Logger()
    goal_position = np.array([1.0, -2.0, 0.5], dtype=float)
    goal_quaternion = np.array(
        euler_to_quat_wxyz(0.0, 0.0, np.pi / 2.0),
        dtype=float,
    )
    goal_signature = tuple(
        np.concatenate([goal_position, goal_quaternion]).tolist()
    )
    parameters = {
        'regenerate_on_goal_change': False,
        'goal_reached_tol_m': 0.05,
        'hold_attitude': True,
        'goal_reached_orientation_tol_rad': np.deg2rad(5.0),
        'use_box_recovery_mission': False,
    }
    rebuilds = []

    fake = SimpleNamespace(
        have_odom=True,
        mission_state='INIT',
        traj_active=True,
        last_goal_signature=goal_signature,
        terminal_hold_goal_signature=None,
        p_w=goal_position.copy(),
        traj_goal_pos=goal_position.copy(),
        q_wxyz=(1.0, 0.0, 0.0, 0.0),
        _update_mission=lambda: None,
        _update_fixed_hook_hold_phase=lambda: False,
        _goal_signature=lambda: goal_signature,
        _goal_position=lambda: goal_position.copy(),
        _goal_quaternion=lambda: goal_quaternion.copy(),
        _at_active_goal=lambda: True,
        _begin_fixed_hook_final_hold=lambda _signature: False,
        _complete_fixed_hook_retreat=lambda _signature: False,
        _reset_trajectory_from_current_pose=lambda: rebuilds.append(True),
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        get_logger=lambda: logger,
    )
    fake._trajectory_completion_reached = lambda: (
        MPCTrackTrajectoryAcados._trajectory_completion_reached(fake)
    )
    fake._enter_terminal_hold = lambda signature: (
        MPCTrackTrajectoryAcados._enter_terminal_hold(fake, signature)
    )

    MPCTrackTrajectoryAcados._maybe_refresh_trajectory(fake)
    assert fake.traj_active is True
    assert fake.terminal_hold_goal_signature is None

    fake.q_wxyz = tuple(goal_quaternion.tolist())
    MPCTrackTrajectoryAcados._maybe_refresh_trajectory(fake)
    assert fake.traj_active is False
    assert fake.terminal_hold_goal_signature == goal_signature
    assert rebuilds == []

    terminal_reference = MPCTrackTrajectoryAcados._trajectory_stage_param(
        fake, 0
    )
    np.testing.assert_allclose(terminal_reference[0:3], goal_position)
    np.testing.assert_allclose(terminal_reference[3:7], goal_quaternion)
    assert terminal_reference[7] == pytest.approx(1.0)


def test_fixed_hook_holds_five_seconds_then_retreats_to_pre_approach():
    logger = _Logger()
    now = [10.0]
    resets = []
    terminal_holds = []
    parameters = {
        'use_box_recovery_mission': False,
        'final_pose_hold_s': 5.0,
        'return_to_pre_approach_after_hold': True,
    }
    fake = SimpleNamespace(
        mission_state='FINAL_APPROACH',
        state_enter_time_sec=0.0,
        terminal_hold_goal_signature=None,
        last_goal_signature=None,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        get_logger=lambda: logger,
        _now_sec=lambda: now[0],
        _trajectory_completion_reached=lambda: True,
        _pre_approach_waypoint_enabled=lambda: True,
        _update_mission=lambda: None,
        _reset_trajectory_from_current_pose=lambda: resets.append(
            fake.mission_state
        ),
        _enter_terminal_hold=lambda signature: terminal_holds.append(
            signature
        ),
    )

    signature = (1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0)
    assert MPCTrackTrajectoryAcados._begin_fixed_hook_final_hold(
        fake, signature
    )
    assert fake.mission_state == 'FINAL_HOLD'
    assert fake.state_enter_time_sec == pytest.approx(10.0)
    assert terminal_holds == [signature]

    now[0] = 14.99
    assert MPCTrackTrajectoryAcados._update_fixed_hook_hold_phase(fake)
    assert fake.mission_state == 'FINAL_HOLD'
    assert resets == []

    now[0] = 15.0
    assert MPCTrackTrajectoryAcados._update_fixed_hook_hold_phase(fake)
    assert fake.mission_state == 'RETREAT'
    assert resets == ['RETREAT']

    assert MPCTrackTrajectoryAcados._complete_fixed_hook_retreat(
        fake, signature
    )
    assert fake.mission_state == 'COMPLETE'
    assert terminal_holds == [signature, signature]


def test_fixed_hook_hold_timer_resets_after_pose_drift():
    logger = _Logger()
    resets = []
    parameters = {'use_box_recovery_mission': False}
    fake = SimpleNamespace(
        mission_state='FINAL_HOLD',
        state_enter_time_sec=10.0,
        terminal_hold_goal_signature=(1.0,),
        last_goal_signature=(1.0,),
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        get_logger=lambda: logger,
        _now_sec=lambda: 12.0,
        _trajectory_completion_reached=lambda: False,
        _reset_trajectory_from_current_pose=lambda: resets.append(True),
    )

    assert MPCTrackTrajectoryAcados._update_fixed_hook_hold_phase(fake)
    assert fake.mission_state == 'FINAL_APPROACH'
    assert fake.terminal_hold_goal_signature is None
    assert fake.last_goal_signature is None
    assert resets == [True]


def test_fixed_hook_retreat_goal_is_exactly_the_pre_approach_pose():
    pre_approach = np.array([4.8448, 0.0777, 1.7281])
    final_goal = np.array([4.3596, -0.0429, 1.7281])
    parameters = {'use_box_recovery_mission': False}
    fake = SimpleNamespace(
        mission_state='RETREAT',
        active_goal_pos=np.zeros(3),
        active_goal_yaw=0.0,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        _pre_approach_waypoint_enabled=lambda: True,
        _pre_approach_position=lambda: pre_approach.copy(),
        _goal_position_static=lambda: final_goal.copy(),
        _goal_yaw_static=lambda: -2.9,
    )

    MPCTrackTrajectoryAcados._update_mission(fake)
    np.testing.assert_allclose(fake.active_goal_pos, pre_approach)
    assert fake.active_goal_yaw == pytest.approx(-2.9)

    fake.mission_state = 'FINAL_HOLD'
    MPCTrackTrajectoryAcados._update_mission(fake)
    np.testing.assert_allclose(fake.active_goal_pos, final_goal)


def _position_integral_fake(*, position_error_z_m=-0.32):
    fake = SimpleNamespace(
        thrust_sat_norm=0.12,
        force_axis_max_N=np.array([88.0, 88.0, 137.0]),
        position_integral_gain=3.0,
        position_integral_force_limit_fraction=0.05,
        position_integral_activation_error_m=0.50,
        position_integral_max_dt_s=0.10,
        position_integral_error_world=np.zeros(3, dtype=float),
        last_position_integral_update_sec=None,
        traj_active=True,
        traj_start_time_sec=0.0,
        traj_duration_sec=10.0,
        p_w=np.array([0.0, 0.0, -position_error_z_m]),
        q_wxyz=(1.0, 0.0, 0.0, 0.0),
        _goal_position=lambda: np.zeros(3, dtype=float),
    )
    fake._reset_position_integral = lambda: (
        MPCTrackTrajectoryAcados._reset_position_integral(fake)
    )
    fake._position_integral_reference_finished = lambda now_sec: (
        MPCTrackTrajectoryAcados._position_integral_reference_finished(
            fake, now_sec
        )
    )
    fake._fixed_hook_transit_active = lambda: False
    return fake


def test_fixed_hook_phase_speeds_keep_final_approach_deliberate():
    parameters = {
        'traj_speed_mps': 0.05,
        'forward_pass_speed_mps': 0.04,
        'backward_pass_speed_mps': 0.04,
    }
    fake = SimpleNamespace(
        mission_state='PRE_APPROACH',
        pre_approach_speed_mps=0.06,
        final_approach_speed_mps=0.04,
        get_parameter=lambda name: SimpleNamespace(
            value=parameters[name]
        ),
    )

    assert MPCTrackTrajectoryAcados._phase_traj_speed(fake) == pytest.approx(
        0.06
    )
    fake.mission_state = 'FINAL_APPROACH'
    assert MPCTrackTrajectoryAcados._phase_traj_speed(fake) == pytest.approx(
        0.04
    )
    fake.mission_state = 'RETREAT'
    assert MPCTrackTrajectoryAcados._phase_traj_speed(fake) == pytest.approx(
        0.04
    )
    fake.mission_state = 'COMPLETE'
    assert MPCTrackTrajectoryAcados._phase_traj_speed(fake) == pytest.approx(
        0.05
    )


def test_position_integral_removes_steady_error_after_reference_finishes():
    fake = _position_integral_fake(position_error_z_m=-0.32)
    base_force = np.array([0.0, 0.0, -3.7])

    # Normal lag behind the still-moving reference must not wind up.
    before_end = MPCTrackTrajectoryAcados._force_with_position_integral(
        fake, base_force, 9.9
    )
    np.testing.assert_allclose(before_end, base_force)
    np.testing.assert_allclose(fake.position_integral_error_world, 0.0)

    # Once the nominal reference is complete, the persistent 0.32 m depth
    # error builds a smooth force bias in the direction of the target.
    for step in range(80):
        force = MPCTrackTrajectoryAcados._force_with_position_integral(
            fake,
            base_force,
            10.0 + 0.1 * step,
        )

    assert force[2] < base_force[2]
    expected_bias_limit = 0.05 * 137.0
    assert force[2] == pytest.approx(
        base_force[2] - expected_bias_limit
    )


def test_fixed_hook_transit_freezes_only_learned_world_z_bias():
    fake = _position_integral_fake(position_error_z_m=-0.03)
    fake.position_integral_error_world[:] = [0.4, -0.2, -1.0]
    fake.last_position_integral_update_sec = 8.0
    fake._fixed_hook_transit_active = lambda: True
    base_force = np.array([1.0, 2.0, -2.0])

    force = MPCTrackTrajectoryAcados._force_with_position_integral(
        fake,
        base_force,
        9.0,
    )

    np.testing.assert_allclose(force, [1.0, 2.0, -5.0])
    np.testing.assert_allclose(
        fake.position_integral_error_world,
        [0.0, 0.0, -1.0],
    )
    assert fake.last_position_integral_update_sec is None


def test_position_integral_combined_force_never_exceeds_thrust_limit():
    fake = _position_integral_fake(position_error_z_m=-0.32)
    fake.traj_active = False
    # Preload more integral than is allowed to verify both the bias clamp and
    # the final combined MPC + integral clamp.
    fake.position_integral_error_world[:] = [0.0, 0.0, -100.0]
    fake.last_position_integral_update_sec = 20.0
    base_force = np.array([0.0, 0.0, -15.0])

    force = MPCTrackTrajectoryAcados._force_with_position_integral(
        fake, base_force, 20.0
    )

    total_limits = 0.12 * fake.force_axis_max_N
    assert np.all(np.abs(force) <= total_limits + 1e-12)
    assert force[2] == pytest.approx(-total_limits[2])
    # Back-calculation retains only the bias that was actually deliverable.
    delivered_bias = force[2] - base_force[2]
    assert fake.position_integral_error_world[2] == pytest.approx(
        delivered_bias / fake.position_integral_gain
    )


def test_position_integral_resets_on_trajectory_or_safety_reset():
    fake = _position_integral_fake(position_error_z_m=-0.32)
    fake.position_integral_error_world[:] = [1.0, 2.0, 3.0]
    fake.last_position_integral_update_sec = 5.0

    # A new moving trajectory must not inherit a previous waypoint's bias.
    force = MPCTrackTrajectoryAcados._force_with_position_integral(
        fake, np.array([1.0, 2.0, 3.0]), 5.1
    )
    np.testing.assert_allclose(force, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(fake.position_integral_error_world, 0.0)
    assert fake.last_position_integral_update_sec is None

    # The same reset is part of every fail-closed command invalidation.
    fake.position_integral_error_world[:] = [1.0, 2.0, 3.0]
    fake.last_position_integral_update_sec = 5.0
    fake.u_force_cmd_N = np.ones(3)
    fake.u_tau_cmd_Nm = np.ones(3)
    fake.command_valid = True
    fake.last_solution_sec = 5.0
    fake.traj_active = True
    fake.last_goal_signature = ('old',)
    fake.terminal_hold_goal_signature = ('old',)
    fake._trajectory_reset_pending = False
    fake._zero_command_cache = lambda: (
        MPCTrackTrajectoryAcados._zero_command_cache(fake)
    )
    fake.publish_zero = lambda: None

    MPCTrackTrajectoryAcados._invalidate_command(
        fake, reset_trajectory=True, publish_zero=True
    )
    np.testing.assert_allclose(fake.position_integral_error_world, 0.0)
    assert fake.last_position_integral_update_sec is None


def test_pre_approach_completion_rebuilds_final_then_enters_terminal_hold():
    logger = _Logger()
    pre_approach = np.array([4.85, 0.08, 1.73], dtype=float)
    final_goal = np.array([4.36, -0.04, 1.73], dtype=float)
    goal_quaternion = np.array(
        euler_to_quat_wxyz(0.0, 0.0, -0.4),
        dtype=float,
    )
    parameters = {
        'regenerate_on_goal_change': False,
        'goal_reached_tol_m': 0.05,
        'hold_attitude': True,
        'goal_reached_orientation_tol_rad': np.deg2rad(5.0),
        'use_box_recovery_mission': False,
    }
    rebuilds = []

    fake = SimpleNamespace(
        have_odom=True,
        mission_state='PRE_APPROACH',
        state_enter_time_sec=0.0,
        traj_active=True,
        last_goal_signature=None,
        terminal_hold_goal_signature=None,
        p_w=pre_approach.copy(),
        traj_goal_pos=pre_approach.copy(),
        q_wxyz=tuple(goal_quaternion.tolist()),
        active_goal_pos=pre_approach.copy(),
        active_goal_yaw=-0.4,
        _update_mission=lambda: None,
        _update_fixed_hook_hold_phase=lambda: False,
        _pre_approach_waypoint_enabled=lambda: True,
        _goal_position_static=lambda: final_goal.copy(),
        _goal_yaw_static=lambda: -0.4,
        _goal_quaternion=lambda: goal_quaternion.copy(),
        _at_active_goal=lambda: True,
        _begin_fixed_hook_final_hold=lambda _signature: False,
        _complete_fixed_hook_retreat=lambda _signature: False,
        _now_sec=lambda: 12.5,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        get_logger=lambda: logger,
    )

    def goal_signature():
        return tuple(
            np.concatenate(
                [fake.active_goal_pos, goal_quaternion]
            ).tolist()
        )

    def rebuild():
        rebuilds.append(fake.active_goal_pos.copy())
        fake.traj_goal_pos = fake.active_goal_pos.copy()
        fake.traj_active = True
        fake.last_goal_signature = goal_signature()
        fake.terminal_hold_goal_signature = None

    fake._goal_signature = goal_signature
    fake._reset_trajectory_from_current_pose = rebuild
    fake._trajectory_completion_reached = lambda: (
        MPCTrackTrajectoryAcados._trajectory_completion_reached(fake)
    )
    fake._advance_pre_approach_phase = lambda: (
        MPCTrackTrajectoryAcados._advance_pre_approach_phase(fake)
    )
    fake._enter_terminal_hold = lambda signature: (
        MPCTrackTrajectoryAcados._enter_terminal_hold(fake, signature)
    )

    MPCTrackTrajectoryAcados._maybe_refresh_trajectory(fake)

    assert fake.mission_state == 'FINAL_APPROACH'
    np.testing.assert_allclose(fake.active_goal_pos, final_goal)
    assert fake.state_enter_time_sec == pytest.approx(12.5)
    assert len(rebuilds) == 1
    np.testing.assert_allclose(rebuilds[0], final_goal)
    assert fake.terminal_hold_goal_signature is None

    fake.p_w = final_goal.copy()
    fake.traj_goal_pos = final_goal.copy()
    fake.q_wxyz = tuple(goal_quaternion.tolist())
    MPCTrackTrajectoryAcados._maybe_refresh_trajectory(fake)

    assert fake.mission_state == 'FINAL_APPROACH'
    assert fake.traj_active is False
    assert fake.terminal_hold_goal_signature == goal_signature()
    assert len(rebuilds) == 1


def test_fixed_hook_forward_and_retreat_share_level_pose_locked_path():
    pre_approach = np.array([4.8448, 0.0777, 1.7281], dtype=float)
    hook = np.array([4.3596, -0.0429, 1.7281], dtype=float)
    recorded_q = np.array(
        euler_to_quat_wxyz(0.03, -0.04, -2.90),
        dtype=float,
    )
    parameters = {
        'use_box_recovery_mission': False,
        'traj_angular_speed_rad_s': np.deg2rad(8.0),
        'min_traj_duration_s': 5.0,
        'Ts': 0.04,
        'hold_attitude': True,
    }
    fake = SimpleNamespace(
        mission_state='FINAL_APPROACH',
        p_w=pre_approach + np.array([0.012, -0.009, 0.04]),
        q_wxyz=(1.0, 0.0, 0.0, 0.0),
        now_sec=100.0,
        fixed_hook_projected_restart=False,
        position_integral_error_world=np.array([0.3, -0.2, -1.0]),
        last_position_integral_update_sec=90.0,
        terminal_hold_goal_signature='old-hold',
        last_goal_signature=None,
        pre_approach_speed_mps=0.06,
        final_approach_speed_mps=0.04,
        get_parameter=lambda name: SimpleNamespace(
            value=parameters[name]
        ),
        get_logger=lambda: _Logger(),
        _pre_approach_waypoint_enabled=lambda: True,
        _pre_approach_position=lambda: pre_approach.copy(),
        _goal_position_static=lambda: hook.copy(),
        _goal_quaternion=lambda: recorded_q.copy(),
        _goal_signature=lambda: ('fixed-hook-goal',),
        _phase_traj_speed=lambda: 0.04,
    )
    fake._now_sec = lambda: fake.now_sec
    fake._reset_position_integral = lambda: (
        MPCTrackTrajectoryAcados._reset_position_integral(fake)
    )
    fake._fixed_hook_transit_active = lambda: (
        MPCTrackTrajectoryAcados._fixed_hook_transit_active(fake)
    )
    fake._fixed_hook_line_segment = lambda: (
        MPCTrackTrajectoryAcados._fixed_hook_line_segment(fake)
    )

    def build_and_sample(goal):
        fake.now_sec = 100.0
        MPCTrackTrajectoryAcados._set_linear_trajectory(fake, goal)
        duration = fake.traj_duration_sec
        samples = []
        for alpha in np.linspace(0.0, 1.0, 5):
            fake.now_sec = 100.0 + alpha * duration
            samples.append(
                MPCTrackTrajectoryAcados._trajectory_stage_param(fake, 0)
            )
        return duration, samples

    forward_duration, forward = build_and_sample(hook)
    np.testing.assert_allclose(fake.traj_start_pos, pre_approach)
    np.testing.assert_allclose(fake.traj_goal_pos, hook)
    np.testing.assert_allclose(
        fake.position_integral_error_world,
        [0.0, 0.0, -1.0],
    )

    fake.mission_state = 'RETREAT'
    fake.p_w = hook + np.array([-0.01, 0.008, -0.04])
    fake.q_wxyz = tuple(
        euler_to_quat_wxyz(-0.2, 0.1, -2.7)
    )
    retreat_duration, retreat = build_and_sample(pre_approach)
    np.testing.assert_allclose(fake.traj_start_pos, hook)
    np.testing.assert_allclose(fake.traj_goal_pos, pre_approach)

    expected_duration = np.linalg.norm(hook - pre_approach) / 0.04
    assert forward_duration == pytest.approx(expected_duration)
    assert retreat_duration == pytest.approx(expected_duration)
    for index, alpha in enumerate(np.linspace(0.0, 1.0, 5)):
        expected_forward = pre_approach + alpha * (hook - pre_approach)
        np.testing.assert_allclose(forward[index][0:3], expected_forward)
        np.testing.assert_allclose(
            retreat[index][0:3],
            forward[-1 - index][0:3],
        )
        assert forward[index][2] == pytest.approx(hook[2])
        assert retreat[index][2] == pytest.approx(hook[2])
        assert quat_angular_distance_wxyz(
            forward[index][3:7],
            recorded_q,
        ) == pytest.approx(0.0, abs=1e-9)
        assert quat_angular_distance_wxyz(
            retreat[index][3:7],
            recorded_q,
        ) == pytest.approx(0.0, abs=1e-9)


def test_pre_approach_requires_tighter_depth_before_forward_motion():
    goal = np.array([4.8448, 0.0777, 1.7281], dtype=float)
    goal_q = np.array(
        euler_to_quat_wxyz(0.0, 0.0, -2.9),
        dtype=float,
    )
    parameters = {
        'goal_reached_tol_m': 0.05,
        'use_box_recovery_mission': False,
        'hold_attitude': True,
        'goal_reached_orientation_tol_rad': np.deg2rad(5.0),
    }
    fake = SimpleNamespace(
        mission_state='PRE_APPROACH',
        p_w=goal + np.array([0.0, 0.0, 0.04]),
        traj_goal_pos=goal.copy(),
        q_wxyz=tuple(goal_q.tolist()),
        fixed_hook_depth_tolerance_m=0.03,
        get_parameter=lambda name: SimpleNamespace(
            value=parameters[name]
        ),
        _goal_quaternion=lambda: goal_q.copy(),
        _pre_approach_waypoint_enabled=lambda: True,
        _at_active_goal=lambda: True,
    )

    assert not MPCTrackTrajectoryAcados._trajectory_completion_reached(fake)
    fake.p_w[2] = goal[2] + 0.02
    assert MPCTrackTrajectoryAcados._trajectory_completion_reached(fake)


def test_linear_trajectory_duration_respects_attitude_slew_rate():
    logger = _Logger()
    goal_position = np.array([1.0, 2.0, 3.0], dtype=float)
    goal_quaternion = np.array(
        euler_to_quat_wxyz(0.0, 0.0, np.pi / 2.0),
        dtype=float,
    )
    parameters = {
        'position_like_motion': False,
        'traj_angular_speed_rad_s': np.deg2rad(10.0),
        'min_traj_duration_s': 2.0,
        'Ts': 0.04,
    }
    fake = SimpleNamespace(
        p_w=goal_position.copy(),
        q_wxyz=(1.0, 0.0, 0.0, 0.0),
        terminal_hold_goal_signature='old-hold',
        last_goal_signature=None,
        _phase_traj_speed=lambda: 0.03,
        _goal_quaternion=lambda: goal_quaternion.copy(),
        _goal_signature=lambda: ('goal',),
        _now_sec=lambda: 100.0,
        _reset_position_integral=lambda: None,
        _fixed_hook_line_segment=lambda: None,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        get_logger=lambda: logger,
    )
    fake._trajectory_start_attitude_reference = lambda: (
        MPCTrackTrajectoryAcados._trajectory_start_attitude_reference(
            fake
        )
    )

    MPCTrackTrajectoryAcados._set_linear_trajectory(
        fake, goal_position
    )

    # A 90 degree rotation at 10 deg/s needs 9 s even with zero translation.
    assert fake.traj_duration_sec == pytest.approx(9.0)
    assert fake.traj_start_time_sec == pytest.approx(100.0)
    assert fake.traj_active is True
    assert fake.terminal_hold_goal_signature is None


def test_disabled_pre_approach_keeps_direct_goal_restart_behavior():
    final_goal = np.array([4.36, -0.04, 1.73], dtype=float)
    parameters = {
        'use_box_recovery_mission': False,
        'use_pre_approach_waypoint': False,
        'goal_x': final_goal[0],
        'goal_y': final_goal[1],
        'goal_z': final_goal[2],
        'goal_roll': 0.0,
        'goal_pitch': 0.0,
        'goal_yaw': -0.4,
        'require_mission_enable': True,
        # Zero retains the legacy meaning: disable both spherical start gates.
        'max_initial_goal_distance_m': 0.0,
        'max_initial_goal_orientation_error_rad': 0.0,
    }
    rebuilt_goals = []
    x0 = np.zeros(13, dtype=float)
    x0[3] = 1.0
    fake = SimpleNamespace(
        traj_active=True,
        last_goal_signature=('old',),
        terminal_hold_goal_signature=('old',),
        forward_pass_start_pos=np.ones(3),
        forward_pass_start_time_sec=5.0,
        mission_state='INIT',
        p_w=np.zeros(3),
        q_wxyz=(1.0, 0.0, 0.0, 0.0),
        q_goal=(1.0, 0.0, 0.0, 0.0),
        x_guess=np.ones((2, 13), dtype=float),
        u_guess=np.ones((1, 6), dtype=float),
        _trajectory_reset_pending=True,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
    )
    fake._pre_approach_waypoint_enabled = lambda: (
        MPCTrackTrajectoryAcados._pre_approach_waypoint_enabled(fake)
    )
    fake._goal_position_static = lambda: (
        MPCTrackTrajectoryAcados._goal_position_static(fake)
    )
    fake._goal_yaw_static = lambda: (
        MPCTrackTrajectoryAcados._goal_yaw_static(fake)
    )
    fake._update_mission = lambda: (
        MPCTrackTrajectoryAcados._update_mission(fake)
    )
    fake._goal_position = lambda: (
        MPCTrackTrajectoryAcados._goal_position(fake)
    )
    fake._goal_quaternion = lambda: (
        MPCTrackTrajectoryAcados._goal_quaternion(fake)
    )
    fake._reset_trajectory_from_current_pose = lambda: (
        rebuilt_goals.append(fake.active_goal_pos.copy())
    )
    fake._revoke_mission_enable = lambda reason: pytest.fail(reason)

    assert MPCTrackTrajectoryAcados._restart_trajectory_from_current_state(
        fake, x0
    ) is True

    assert fake.mission_state == 'INIT'
    np.testing.assert_allclose(fake.active_goal_pos, final_goal)
    assert len(rebuilt_goals) == 1
    np.testing.assert_allclose(rebuilt_goals[0], final_goal)
    np.testing.assert_allclose(fake.x_guess, np.tile(x0, (2, 1)))
    np.testing.assert_allclose(fake.u_guess, np.zeros((1, 6)))
    assert fake._trajectory_reset_pending is False


def test_preflight_health_blip_before_operator_request_does_not_latch():
    logger = _Logger()
    fake = SimpleNamespace(
        first_heartbeat_time=10.0,
        require_request_enable=True,
        request_enabled=False,
        offboard_active=False,
        controller_rearm_required=False,
        get_logger=lambda: logger,
    )

    OffboardEnableNode._withdraw_heartbeat_for_health_failure(
        fake, 'preflight feedback blip'
    )

    assert fake.request_enabled is False
    assert fake.controller_rearm_required is False
    assert fake.first_heartbeat_time is None


def test_health_failure_in_actual_offboard_latches_without_request_true():
    logger = _Logger()
    fake = SimpleNamespace(
        first_heartbeat_time=10.0,
        require_request_enable=True,
        request_enabled=False,
        offboard_active=True,
        controller_rearm_required=False,
        offboard_ack_accepted=True,
        arm_accepted=True,
        manual_arm_notice_logged=True,
        get_logger=lambda: logger,
    )

    OffboardEnableNode._withdraw_heartbeat_for_health_failure(
        fake, 'offboard feedback failure'
    )

    assert fake.request_enabled is False
    assert fake.controller_rearm_required is True
    assert fake.offboard_ack_accepted is False
    assert fake.arm_accepted is False
    assert fake.first_heartbeat_time is None


def test_hard_publisher_violation_latches_before_operator_request():
    logger = _Logger()
    fake = SimpleNamespace(
        first_heartbeat_time=None,
        require_request_enable=True,
        request_enabled=False,
        offboard_active=False,
        controller_rearm_required=False,
        offboard_ack_accepted=True,
        arm_accepted=True,
        manual_arm_notice_logged=True,
        get_logger=lambda: logger,
    )

    OffboardEnableNode._withdraw_heartbeat_for_health_failure(
        fake, 'unexpected control publisher', force_rearm=True
    )

    assert fake.controller_rearm_required is True
    assert fake.offboard_ack_accepted is False
    assert fake.arm_accepted is False


def _endpoint(node_name, node_namespace='/'):
    return SimpleNamespace(
        node_name=node_name,
        node_namespace=node_namespace,
    )


def _publisher_guard_fake(endpoint_map):
    return SimpleNamespace(
        offboard_mode_topic='/glub/fmu/in/offboard_control_mode',
        vehicle_cmd_topic='/glub/fmu/in/vehicle_command',
        exclusive_thrust_sp_topic=(
            '/glub/fmu/in/vehicle_thrust_setpoint'
        ),
        exclusive_torque_sp_topic=(
            '/glub/fmu/in/vehicle_torque_setpoint'
        ),
        expected_controller_node_fqn='/mpc_fixed_hook_pose_validation',
        get_fully_qualified_name=lambda: '/offboard_enable_fixed_hook',
        get_publishers_info_by_topic=lambda topic: endpoint_map.get(topic, []),
        _endpoint_node_fqn=OffboardEnableNode._endpoint_node_fqn,
    )


def test_control_publisher_exclusivity_accepts_exact_expected_graph():
    fake = _publisher_guard_fake({
        '/glub/fmu/in/offboard_control_mode': [
            _endpoint('offboard_enable_fixed_hook')
        ],
        '/glub/fmu/in/vehicle_command': [
            _endpoint('offboard_enable_fixed_hook')
        ],
        '/glub/fmu/in/vehicle_thrust_setpoint': [
            _endpoint('mpc_fixed_hook_pose_validation')
        ],
        '/glub/fmu/in/vehicle_torque_setpoint': [
            _endpoint('mpc_fixed_hook_pose_validation')
        ],
    })

    assert OffboardEnableNode._control_publishers_are_exclusive(fake) == (
        True, False, ''
    )


def test_control_publisher_exclusivity_distinguishes_missing_from_wrong():
    missing_fake = _publisher_guard_fake({})
    valid, hard_violation, _detail = (
        OffboardEnableNode._control_publishers_are_exclusive(missing_fake)
    )
    assert valid is False
    assert hard_violation is False

    wrong_fake = _publisher_guard_fake({
        '/glub/fmu/in/offboard_control_mode': [
            _endpoint('offboard_enable_fixed_hook'),
            _endpoint('legacy_heartbeat'),
        ],
        '/glub/fmu/in/vehicle_command': [
            _endpoint('offboard_enable_fixed_hook')
        ],
        '/glub/fmu/in/vehicle_thrust_setpoint': [
            _endpoint('wrong_controller')
        ],
        '/glub/fmu/in/vehicle_torque_setpoint': [
            _endpoint('mpc_fixed_hook_pose_validation')
        ],
    })
    valid, hard_violation, detail = (
        OffboardEnableNode._control_publishers_are_exclusive(wrong_fake)
    )
    assert valid is False
    assert hard_violation is True
    assert 'legacy_heartbeat' in detail
    assert 'wrong_controller' in detail


def test_vehicle_command_ack_topic_follows_compiled_message_version():
    class NoVersion:
        pass

    class VersionZero:
        MESSAGE_VERSION = 0

    class VersionThree:
        MESSAGE_VERSION = 3

    base = '/glub/fmu/out/vehicle_command_ack'
    assert versioned_px4_topic(base, NoVersion) == base
    assert versioned_px4_topic(base, VersionZero) == base
    assert versioned_px4_topic(base, VersionThree) == base + '_v3'
    compiled_version = int(VehicleCommandAck.MESSAGE_VERSION)
    compiled_topic = (
        base + f'_v{compiled_version}' if compiled_version > 0 else base
    )
    assert versioned_px4_topic(base, VehicleCommandAck) == compiled_topic


def test_real_launch_rejects_px4_msgs_not_matching_current_firmware(
    monkeypatch,
):
    module = _load_launch_module()
    prefix, ack_version, status_version = (
        module._validate_firmware_matched_px4_msgs()
    )
    assert prefix
    assert ack_version == 0
    assert status_version == 1

    class WrongAck:
        MESSAGE_VERSION = 1

    monkeypatch.setattr(module, 'VehicleCommandAck', WrongAck)
    with pytest.raises(RuntimeError, match='does not match'):
        module._validate_firmware_matched_px4_msgs()


def test_ack_result_names_allow_older_message_without_optional_constants():
    class OlderAck:
        VEHICLE_CMD_RESULT_ACCEPTED = 0
        VEHICLE_CMD_RESULT_TEMPORARILY_REJECTED = 1
        VEHICLE_CMD_RESULT_DENIED = 2
        VEHICLE_CMD_RESULT_UNSUPPORTED = 3
        VEHICLE_CMD_RESULT_FAILED = 4
        VEHICLE_CMD_RESULT_IN_PROGRESS = 5
        VEHICLE_CMD_RESULT_CANCELLED = 6

    names = _build_ack_result_names(OlderAck)
    assert names[0] == 'ACCEPTED'
    assert names[6] == 'CANCELLED'
    assert set(names) == set(range(7))
    assert DEFAULT_CONTROL_MODE_TIMEOUT_SEC == 1.25


def test_fixed_hook_mode_feedback_timeouts_allow_slow_vehicle_status():
    path = (
        Path(__file__).resolve().parents[1]
        / 'launch'
        / 'fixed_hook_pose_validation.launch.py'
    )
    tree = ast.parse(path.read_text(encoding='utf-8'))
    literal_parameters = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and isinstance(value, ast.Constant):
                literal_parameters.append((key.value, value.value))

    assert ('control_mode_timeout_s', 1.25) in literal_parameters
    assert ('control_mode_timeout_sec', 1.25) in literal_parameters
    assert ('controller_heartbeat_timeout_sec', 0.30) in literal_parameters
    assert ('enforce_control_publisher_exclusivity', True) in (
        literal_parameters
    )
    assert ('operating_bounds_enable', True) in literal_parameters


def test_real_offboard_launch_defaults_are_safe_and_versioned(monkeypatch):
    path = (
        Path(__file__).resolve().parents[1]
        / 'launch'
        / 'offboard_enable_real.launch.py'
    )
    spec = importlib.util.spec_from_file_location(
        'offboard_enable_real_launch', path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    context = LaunchContext()
    for entity in module.generate_launch_description().entities:
        if getattr(entity, 'name', None) is not None:
            entity.execute(context)

    assert context.launch_configurations['control_mode_timeout_sec'] == '1.25'
    assert (
        context.launch_configurations[
            'enforce_control_publisher_exclusivity'
        ]
        == 'false'
    )

    monkeypatch.setattr(module, 'Node', lambda **kwargs: kwargs)
    node = module._launch_setup(context)[0]
    parameters = node['parameters'][0]
    assert parameters['vehicle_cmd_ack_topic'] == versioned_px4_topic(
        '/glub/fmu/out/vehicle_command_ack', VehicleCommandAck
    )
    assert parameters['control_mode_timeout_sec'] == 1.25
    assert parameters['enforce_control_publisher_exclusivity'] is False
