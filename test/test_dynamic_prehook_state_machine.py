"""Focused unit tests for the dynamic fixed-hook pre-hook state machine."""

from concurrent.futures import Future
import math
import time
from types import SimpleNamespace

import casadi as ca
import numpy as np
import pytest
from std_srvs.srv import Trigger

from bluerov2_control.mpc_track_trajectory_acados import (
    euler_to_quat_wxyz,
    MPCTrackTrajectoryAcados,
    quat_angular_distance_wxyz,
    quat_to_rpy_wxyz,
    quat_to_rotation_matrix_wxyz,
    quat_to_rotation_matrix_sym_wxyz,
)


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message, **_kwargs):
        self.messages.append(('info', message))

    def warning(self, message, **_kwargs):
        self.messages.append(('warning', message))

    def error(self, message, **_kwargs):
        self.messages.append(('error', message))


class _DeferredExecutor:
    def __init__(self):
        self.calls = []
        self.futures = []

    def submit(self, function, **kwargs):
        future = Future()
        self.calls.append((function, kwargs))
        self.futures.append(future)
        return future


def _planning_fake():
    logger = _Logger()
    planned_trajectories = []
    revoked_reasons = []
    executor = _DeferredExecutor()
    prehook = np.array([4.8, 0.1, 1.7], dtype=float)
    parameters = {
        'astar_diagonal_motion': True,
        'prehook_path_smoothing_iterations': 1,
        'prehook_path_smoothing_corner_fraction': 0.2,
        'prehook_path_smoothing_samples_per_corner': 4,
        'prehook_reached_hold_s': 1.0,
        'solve_rate_hz': 25.0,
        'prehook_attitude_reference_mode': 'recorded_hook',
    }
    fake = SimpleNamespace(
        mission_state='INIT',
        state_enter_time_sec=0.0,
        p_w=np.array([1.0, -0.2, 1.1], dtype=float),
        q_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        active_goal_pos=np.zeros(3, dtype=float),
        active_goal_yaw=0.0,
        _prehook_plan_generation=0,
        _prehook_replan_future=None,
        _prehook_replan_context=None,
        _prehook_replan_executor=executor,
        _prehook_mandatory_replan_pending=False,
        _prehook_planner_hold_active=False,
        _prehook_planner_hold_pos=np.zeros(3, dtype=float),
        _prehook_planner_hold_q_wxyz=np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=float
        ),
        _now_sec=lambda: 10.0,
        _pre_approach_position=lambda: prehook.copy(),
        _goal_yaw_static=lambda: -0.4,
        _goal_signature=lambda: ('prehook-goal',),
        _real_prehook_planner_grid=lambda: SimpleNamespace(
            bounds=(0.0, 9.0, -2.5, 2.5),
            obstacles=[],
            resolution=0.1,
        ),
        _candidate_path_from_current=lambda path: list(path),
        _set_path_trajectory=lambda path, goal_z, start_z=None: (
            planned_trajectories.append(
                (list(path), float(goal_z), float(start_z))
            )
        ),
        _set_linear_trajectory=lambda _goal: pytest.fail(
            'dynamic pre-hook planning must not fall back to a line'
        ),
        _revoke_mission_enable=lambda reason: revoked_reasons.append(reason),
        get_parameter=lambda name: SimpleNamespace(
            value=parameters[name]
        ),
        get_logger=lambda: logger,
    )
    fake._reset_position_integral = lambda: None
    fake._retire_prehook_plan_request = lambda: (
        MPCTrackTrajectoryAcados._retire_prehook_plan_request(fake)
    )
    fake._reset_dynamic_prehook_runtime = lambda: (
        MPCTrackTrajectoryAcados._reset_dynamic_prehook_runtime(fake)
    )
    fake._activate_prehook_planner_hold = lambda **kwargs: (
        MPCTrackTrajectoryAcados._activate_prehook_planner_hold(
            fake,
            **kwargs,
        )
    )
    fake._submit_prehook_plan_request = lambda **kwargs: (
        MPCTrackTrajectoryAcados._submit_prehook_plan_request(
            fake, **kwargs
        )
    )
    fake._poll_prehook_replan = lambda **kwargs: (
        MPCTrackTrajectoryAcados._poll_prehook_replan(fake, **kwargs)
    )
    fake._plan_to_prehook_from_current = lambda reason: (
        MPCTrackTrajectoryAcados._plan_to_prehook_from_current(
            fake, reason
        )
    )
    fake._capture_prehook_trim_if_needed = lambda: (
        MPCTrackTrajectoryAcados._capture_prehook_trim_if_needed(fake)
    )
    fake._prehook_attitude_reference_mode = lambda: (
        MPCTrackTrajectoryAcados._prehook_attitude_reference_mode(fake)
    )
    fake._dynamic_prehook_planner_enabled = lambda: True
    fake._mission_allowed = lambda: True
    fake._prehook_alignment_status = lambda: {
        'translation_ready': False,
        'attitude_ok': True,
    }
    fake._update_prehook_attitude_wait = lambda _status: False
    fake._prehook_attitude_wait_active = lambda status: (
        status['translation_ready'] and not status['attitude_ok']
    )
    return (
        fake,
        executor,
        planned_trajectories,
        revoked_reasons,
        prehook,
    )


def test_plan_to_prehook_runs_in_background_and_submits_only_once():
    path = [(1.0, -0.2), (2.0, 0.0), (4.8, 0.1)]
    fake, executor, planned, revoked, prehook = _planning_fake()

    result = MPCTrackTrajectoryAcados._plan_to_prehook_from_current(
        fake,
        reason='unit-test initial plan',
    )

    assert result is True
    assert fake.mission_state == 'PLAN_TO_PREHOOK'
    np.testing.assert_allclose(fake.active_goal_pos, prehook)
    assert fake.active_goal_yaw == pytest.approx(-0.4)
    assert planned == []
    assert revoked == []
    assert len(executor.calls) == 1
    assert executor.calls[0][0].__name__ == 'plan_xy_path'
    assert fake._prehook_planner_hold_active is True
    np.testing.assert_allclose(fake._prehook_planner_hold_pos, fake.p_w)

    for _unused in range(5):
        assert MPCTrackTrajectoryAcados._update_dynamic_prehook_phase(fake)
    assert len(executor.calls) == 1
    assert fake.mission_state == 'PLAN_TO_PREHOOK'

    executor.futures[0].set_result((list(path), None))
    assert MPCTrackTrajectoryAcados._update_dynamic_prehook_phase(fake)
    assert fake.mission_state == 'TRACK_TO_PREHOOK'
    assert planned == [(path, prehook[2], fake.p_w[2])]
    assert fake._prehook_planner_hold_active is False
    assert fake._prehook_last_check_monotonic is not None
    assert fake._prehook_last_switch_monotonic == pytest.approx(
        fake._prehook_last_check_monotonic
    )


def test_dynamic_planner_rejects_non_latching_mission_mode():
    parameters = {
        'planner_mode': 'astar',
        'use_box_recovery_mission': False,
        'require_mission_enable': False,
    }
    fake = SimpleNamespace(
        operating_bounds_enabled=True,
        _dynamic_prehook_planner_enabled=lambda: True,
        _pre_approach_waypoint_enabled=lambda: True,
        get_parameter=lambda name: SimpleNamespace(
            value=parameters[name]
        ),
    )

    with pytest.raises(ValueError, match='require_mission_enable=true'):
        MPCTrackTrajectoryAcados._validate_dynamic_prehook_planner_parameters(
            fake
        )


def test_plan_to_prehook_failure_revokes_without_linear_fallback():
    fake, executor, planned, revoked, _prehook = _planning_fake()

    result = MPCTrackTrajectoryAcados._plan_to_prehook_from_current(
        fake,
        reason='unit-test blocked plan',
    )

    assert result is True
    assert fake.mission_state == 'PLAN_TO_PREHOOK'
    assert planned == []
    assert revoked == []

    executor.futures[0].set_exception(
        RuntimeError('no collision-free path')
    )
    MPCTrackTrajectoryAcados._poll_prehook_replan(fake)
    assert len(revoked) == 1
    assert 'failed closed; no linear fallback' in revoked[0]
    assert 'no collision-free path' in revoked[0]


def test_plan_state_all_nmpc_stages_use_captured_hold_pose():
    hold_pos = np.array([1.2, -0.4, 1.1], dtype=float)
    hold_q = np.array([0.98, 0.0, 0.0, 0.2], dtype=float)
    fake = SimpleNamespace(
        mission_state='PLAN_TO_PREHOOK',
        _prehook_planner_hold_active=True,
        _prehook_planner_hold_pos=hold_pos.copy(),
        _prehook_planner_hold_q_wxyz=hold_q.copy(),
        traj_active=False,
        _goal_quaternion=lambda: np.array(
            [0.0, 0.0, 0.0, 1.0], dtype=float
        ),
        _goal_position=lambda: np.array([4.8, 0.1, 1.7], dtype=float),
        get_parameter=lambda name: SimpleNamespace(
            value={'hold_attitude': True}[name]
        ),
    )

    for stage in range(21):
        reference = MPCTrackTrajectoryAcados._trajectory_stage_param(
            fake, stage
        )
        assert reference.shape == (12,)
        np.testing.assert_allclose(reference[0:3], hold_pos)
        np.testing.assert_allclose(reference[3:7], hold_q)
        assert reference[7] == pytest.approx(1.0)
        np.testing.assert_allclose(reference[8:11], np.zeros(3))
        assert reference[11] == pytest.approx(1.0)


def test_reset_retains_uncancellable_worker_until_it_can_be_polled():
    running_future = Future()
    assert running_future.set_running_or_notify_cancel()
    context = {
        'generation': 2,
        'goal_signature': ('old-goal',),
        'request_level': 'optional',
    }
    fake = SimpleNamespace(
        _prehook_plan_generation=2,
        _prehook_replan_future=running_future,
        _prehook_replan_context=context,
        _prehook_attitude_fault_latched=False,
    )
    fake._retire_prehook_plan_request = lambda: (
        MPCTrackTrajectoryAcados._retire_prehook_plan_request(fake)
    )

    MPCTrackTrajectoryAcados._reset_dynamic_prehook_runtime(fake)

    assert fake._prehook_plan_generation == 3
    assert fake._prehook_replan_future is running_future
    assert fake._prehook_replan_context is context
    assert fake._prehook_mandatory_replan_pending is False
    assert fake._prehook_attitude_fault_latched is False


def test_attitude_fault_survives_generic_command_and_trajectory_reset():
    logger = _Logger()
    hold_pos = np.array([4.81, 0.10, 1.705], dtype=float)
    hold_q = np.asarray(
        euler_to_quat_wxyz(0.1, -0.2, 0.3),
        dtype=float,
    )
    fake = SimpleNamespace(
        mission_state='TRACK_TO_PREHOOK',
        _prehook_attitude_fault_latched=True,
        _prehook_trim_q_wxyz=np.array([0.9, 0.1, 0.1, 0.3]),
        _operator_hook_confirmation_pending=True,
        _wait_hook_enter_monotonic=1.0,
        _prehook_planner_hold_active=True,
        _prehook_planner_hold_pos=hold_pos.copy(),
        _prehook_planner_hold_q_wxyz=hold_q.copy(),
        _prehook_plan_generation=1,
        _prehook_replan_future=None,
        _prehook_replan_context=None,
        u_force_cmd_N=np.ones(3, dtype=float),
        u_tau_cmd_Nm=np.ones(3, dtype=float),
        command_valid=True,
        last_solution_sec=10.0,
        traj_active=True,
        last_goal_signature=('prehook',),
        terminal_hold_goal_signature=('prehook',),
        _trajectory_reset_pending=False,
        fixed_hook_projected_restart=False,
        position_integral_error_world=np.ones(3, dtype=float),
        last_position_integral_update_sec=10.0,
        forward_pass_start_pos=None,
        forward_pass_start_time_sec=0.0,
        x_guess=np.ones((4, 13), dtype=float),
        u_guess=np.ones((3, 6), dtype=float),
        active_goal_pos=np.zeros(3, dtype=float),
        active_goal_yaw=0.0,
        _dynamic_prehook_planner_enabled=lambda: True,
        _pre_approach_position=lambda: np.array(
            [4.8, 0.1, 1.7], dtype=float
        ),
        _goal_yaw_static=lambda: -0.4,
        get_logger=lambda: logger,
        publish_zero=lambda: None,
    )
    fake._zero_command_cache = lambda: (
        MPCTrackTrajectoryAcados._zero_command_cache(fake)
    )
    fake._reset_position_integral = lambda: (
        MPCTrackTrajectoryAcados._reset_position_integral(fake)
    )
    fake._clear_position_integral_xy_preserve_z = lambda: (
        MPCTrackTrajectoryAcados._clear_position_integral_xy_preserve_z(
            fake
        )
    )
    fake._retire_prehook_plan_request = lambda: (
        MPCTrackTrajectoryAcados._retire_prehook_plan_request(fake)
    )
    fake._reset_dynamic_prehook_runtime = lambda: (
        MPCTrackTrajectoryAcados._reset_dynamic_prehook_runtime(fake)
    )

    MPCTrackTrajectoryAcados._invalidate_command(
        fake,
        reset_trajectory=True,
        publish_zero=True,
    )

    assert fake._prehook_attitude_fault_latched is True
    assert fake._operator_hook_confirmation_pending is False
    assert fake._wait_hook_enter_monotonic is None
    assert fake._prehook_planner_hold_active is True
    np.testing.assert_allclose(fake._prehook_planner_hold_pos, hold_pos)
    np.testing.assert_allclose(fake._prehook_planner_hold_q_wxyz, hold_q)
    assert fake._trajectory_reset_pending is True
    np.testing.assert_allclose(
        fake.position_integral_error_world,
        [0.0, 0.0, 1.0],
    )
    assert fake.last_position_integral_update_sec is None

    x0 = np.zeros(13, dtype=float)
    x0[3] = 1.0
    assert MPCTrackTrajectoryAcados._restart_trajectory_from_current_state(
        fake,
        x0,
    )
    assert fake.mission_state == 'TRACK_TO_PREHOOK'
    assert fake._prehook_attitude_fault_latched is True
    assert fake._prehook_planner_hold_active is True
    assert fake._trajectory_reset_pending is False
    np.testing.assert_allclose(fake.x_guess, np.tile(x0, (4, 1)))
    np.testing.assert_allclose(fake.u_guess, 0.0)
    assert any(
        'fault hold preserved' in message
        for level, message in logger.messages
        if level == 'warning'
    )


def test_explicit_mission_false_clears_attitude_fault_latch():
    invalidations = []
    logger = _Logger()
    fake = SimpleNamespace(
        mission_enable=True,
        mission_rearm_required=False,
        _prehook_attitude_fault_latched=True,
        _operator_hook_confirmation_pending=True,
        _wait_hook_enter_monotonic=1.0,
        get_parameter=lambda name: SimpleNamespace(
            value={'require_mission_enable': True}[name]
        ),
        get_logger=lambda: logger,
        _invalidate_command=lambda **kwargs: invalidations.append(kwargs),
    )
    fake._mission_allowed = lambda: (
        MPCTrackTrajectoryAcados._mission_allowed(fake)
    )

    MPCTrackTrajectoryAcados.on_mission_enable(
        fake,
        SimpleNamespace(data=False),
    )

    assert fake.mission_enable is False
    assert fake._prehook_attitude_fault_latched is False
    assert fake._prehook_trim_q_wxyz is None
    assert fake._operator_hook_confirmation_pending is False
    assert fake._wait_hook_enter_monotonic is None
    assert invalidations == [
        {'reset_trajectory': True, 'publish_zero': True}
    ]
    assert any(
        'fault hold cleared' in message
        for level, message in logger.messages
        if level == 'info'
    )


def test_capture_start_trim_is_captured_once_and_keeps_recorded_yaw():
    logger = _Logger()
    recorded_yaw = -2.9
    parameters = {
        'prehook_attitude_reference_mode': 'capture_start_trim',
        'goal_yaw': recorded_yaw,
    }
    fake = SimpleNamespace(
        q_wxyz=np.asarray(
            euler_to_quat_wxyz(0.18, -0.11, 0.7), dtype=float
        ),
        _prehook_trim_q_wxyz=None,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        get_logger=lambda: logger,
        _goal_yaw_static=lambda: recorded_yaw,
    )
    fake._prehook_attitude_reference_mode = lambda: (
        MPCTrackTrajectoryAcados._prehook_attitude_reference_mode(fake)
    )

    MPCTrackTrajectoryAcados._capture_prehook_trim_if_needed(fake)
    captured = fake._prehook_trim_q_wxyz.copy()
    roll, pitch, yaw = quat_to_rpy_wxyz(captured)
    assert roll == pytest.approx(0.18)
    assert pitch == pytest.approx(-0.11)
    assert yaw == pytest.approx(recorded_yaw)

    fake.q_wxyz = np.asarray(
        euler_to_quat_wxyz(-0.4, 0.3, 1.1), dtype=float
    )
    MPCTrackTrajectoryAcados._capture_prehook_trim_if_needed(fake)
    np.testing.assert_allclose(fake._prehook_trim_q_wxyz, captured)
    assert sum(
        'Captured pre-hook attitude trim' in message
        for _level, message in logger.messages
    ) == 1


def test_prehook_path_reference_stays_at_captured_trim_across_replan():
    trim = np.asarray(
        euler_to_quat_wxyz(0.15, -0.08, -2.9), dtype=float
    )
    parameters = {
        'min_traj_duration_s': 0.1,
        'Ts': 0.04,
        'traj_angular_speed_rad_s': 0.5,
        'hold_attitude': True,
    }
    now = [20.0]
    fake = SimpleNamespace(
        mission_state='TRACK_TO_PREHOOK',
        p_w=np.array([1.0, 0.0, 1.2]),
        q_wxyz=np.asarray(euler_to_quat_wxyz(-0.3, 0.2, 0.4)),
        terminal_hold_goal_signature=None,
        last_goal_signature=None,
        _now_sec=lambda: now[0],
        _reset_position_integral=lambda: None,
        _phase_traj_speed=lambda: 0.2,
        _dynamic_prehook_planner_enabled=lambda: True,
        _prehook_attitude_reference_mode=lambda: 'capture_start_trim',
        _prehook_trim_q_wxyz=trim.copy(),
        _prehook_reference_quaternion=lambda: trim.copy(),
        _goal_quaternion=lambda: trim.copy(),
        _goal_signature=lambda: ('trim-prehook',),
        _goal_position=lambda: np.array([2.0, 0.0, 1.2]),
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        get_logger=lambda: _Logger(),
    )
    fake._sample_path_pref = lambda alpha: (
        MPCTrackTrajectoryAcados._sample_path_pref(fake, alpha)
    )

    for start_x in (1.0, 1.3):
        fake.p_w[0] = start_x
        MPCTrackTrajectoryAcados._set_path_trajectory(
            fake,
            [(start_x, 0.0), (2.0, 0.0)],
            1.2,
            start_z=1.2,
        )
        np.testing.assert_allclose(fake.traj_start_q_wxyz, trim)
        for alpha in (0.0, 0.5, 1.0):
            now[0] = fake.traj_start_time_sec + (
                alpha * fake.traj_duration_sec
            )
            reference = MPCTrackTrajectoryAcados._trajectory_stage_param(
                fake, 0
            )
            assert quat_angular_distance_wxyz(
                reference[3:7], trim
            ) == pytest.approx(0.0, abs=1e-9)
        now[0] += 1.0


def test_go_forward_slerps_from_trim_to_recorded_hook_attitude():
    prehook = np.array([4.9, 0.1, 1.7], dtype=float)
    hook = np.array([4.4, 0.1, 1.7], dtype=float)
    trim = np.asarray(euler_to_quat_wxyz(0.16, -0.09, math.pi))
    recorded = np.asarray(euler_to_quat_wxyz(-0.04, 0.03, math.pi))
    parameters = {
        'traj_angular_speed_rad_s': 1.0,
        'min_traj_duration_s': 0.1,
        'Ts': 0.04,
        'hold_attitude': True,
        'fixed_hook_line_cross_track_tol_m': 0.03,
        'fixed_hook_line_yaw_tol_rad': math.radians(3.0),
        'fixed_hook_line_max_reference_lead_m': 0.02,
        'fixed_hook_line_interlock_release_ratio': 0.8,
        'fixed_hook_line_velocity_weight_multiplier': 20.0,
        'fixed_hook_line_position_mode': False,
        'goal_reached_tol_m': 0.05,
        'goal_reached_orientation_tol_rad': math.radians(5.0),
        'use_box_recovery_mission': False,
    }
    now = [30.0]
    fake = SimpleNamespace(
        mission_state='GO_FORWARD',
        p_w=prehook.copy(),
        q_wxyz=trim.copy(),
        fixed_hook_projected_restart=False,
        position_integral_error_world=np.zeros(3),
        last_position_integral_update_sec=None,
        terminal_hold_goal_signature=None,
        last_goal_signature=None,
        _prehook_trim_q_wxyz=trim.copy(),
        _fixed_hook_line_segment=lambda: (prehook.copy(), hook.copy()),
        _prehook_attitude_reference_mode=lambda: 'capture_start_trim',
        _goal_quaternion=lambda: recorded.copy(),
        _goal_position=lambda: hook.copy(),
        _goal_signature=lambda: ('hook',),
        _goal_yaw_static=lambda: math.pi,
        _phase_traj_speed=lambda: 0.1,
        _now_sec=lambda: now[0],
        N_horizon=25,
        fixed_hook_depth_tolerance_m=0.03,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        get_logger=lambda: _Logger(),
    )
    fake._fixed_hook_line_governor_active = lambda: (
        MPCTrackTrajectoryAcados._fixed_hook_line_governor_active(fake)
    )
    fake._fixed_hook_line_governor_status = lambda: (
        MPCTrackTrajectoryAcados._fixed_hook_line_governor_status(fake)
    )
    fake._fixed_hook_line_stage_reference = lambda k, q_goal: (
        MPCTrackTrajectoryAcados._fixed_hook_line_stage_reference(
            fake, k, q_goal
        )
    )

    MPCTrackTrajectoryAcados._set_linear_trajectory(fake, hook)
    np.testing.assert_allclose(fake.traj_start_q_wxyz, trim)
    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)
    start = MPCTrackTrajectoryAcados._trajectory_stage_param(fake, 0)
    fake.p_w = 0.5 * (prehook + hook)
    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)
    midpoint = MPCTrackTrajectoryAcados._trajectory_stage_param(fake, 0)
    fake.p_w = hook.copy()
    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)
    finish = MPCTrackTrajectoryAcados._trajectory_stage_param(fake, 0)
    assert quat_angular_distance_wxyz(start[3:7], trim) == pytest.approx(
        0.0, abs=1e-9
    )
    assert quat_angular_distance_wxyz(
        finish[3:7], recorded
    ) == pytest.approx(0.0, abs=1e-9)
    full_distance = quat_angular_distance_wxyz(trim, recorded)
    assert quat_angular_distance_wxyz(
        trim, midpoint[3:7]
    ) == pytest.approx(0.5 * full_distance)


def _fixed_hook_governor_fake(
    *,
    mission_state='GO_FORWARD',
    position_mode=False,
):
    logger = _Logger()
    if mission_state == 'GO_FORWARD':
        line_start = np.array([0.0, 0.0, 1.7], dtype=float)
        line_goal = np.array([0.5, 0.0, 1.7], dtype=float)
    else:
        line_start = np.array([0.5, 0.0, 1.7], dtype=float)
        line_goal = np.array([0.0, 0.0, 1.7], dtype=float)
    goal_q = np.asarray(euler_to_quat_wxyz(0.0, 0.0, 0.0))
    parameters = {
        'traj_angular_speed_rad_s': math.radians(8.0),
        'min_traj_duration_s': 5.0,
        'Ts': 0.04,
        'hold_attitude': True,
        'fixed_hook_line_cross_track_tol_m': 0.03,
        'fixed_hook_line_yaw_tol_rad': math.radians(3.0),
        'fixed_hook_line_max_reference_lead_m': 0.02,
        'fixed_hook_line_interlock_release_ratio': 0.8,
        'fixed_hook_line_velocity_weight_multiplier': 20.0,
        'fixed_hook_line_position_mode': position_mode,
        'goal_reached_tol_m': 0.05,
        'goal_reached_orientation_tol_rad': math.radians(5.0),
        'use_box_recovery_mission': False,
    }
    fake = SimpleNamespace(
        mission_state=mission_state,
        p_w=line_start.copy(),
        q_wxyz=goal_q.copy(),
        fixed_hook_depth_tolerance_m=0.03,
        fixed_hook_projected_restart=False,
        position_integral_error_world=np.zeros(3),
        last_position_integral_update_sec=None,
        terminal_hold_goal_signature=None,
        last_goal_signature=None,
        _prehook_trim_q_wxyz=goal_q.copy(),
        _fixed_hook_line_segment=lambda: (
            line_start.copy(), line_goal.copy()
        ),
        _prehook_attitude_reference_mode=lambda: 'capture_start_trim',
        _goal_quaternion=lambda: goal_q.copy(),
        _goal_position=lambda: line_goal.copy(),
        _goal_signature=lambda: (mission_state,),
        _goal_yaw_static=lambda: 0.0,
        _phase_traj_speed=lambda: 0.09,
        _now_sec=lambda: 100.0,
        N_horizon=25,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        get_logger=lambda: logger,
    )
    fake._fixed_hook_line_governor_active = lambda: (
        MPCTrackTrajectoryAcados._fixed_hook_line_governor_active(fake)
    )
    fake._fixed_hook_line_position_mode_active = lambda: (
        MPCTrackTrajectoryAcados._fixed_hook_line_position_mode_active(fake)
    )
    fake._fixed_hook_line_governor_status = lambda: (
        MPCTrackTrajectoryAcados._fixed_hook_line_governor_status(fake)
    )
    fake._fixed_hook_line_stage_reference = lambda k, q_goal: (
        MPCTrackTrajectoryAcados._fixed_hook_line_stage_reference(
            fake, k, q_goal
        )
    )
    fake._fixed_hook_line_position_stage_reference = (
        lambda k, q_goal, sample_time_sec=None: (
            MPCTrackTrajectoryAcados
            ._fixed_hook_line_position_stage_reference(
                fake,
                k,
                q_goal,
                sample_time_sec=sample_time_sec,
            )
        )
    )
    MPCTrackTrajectoryAcados._set_linear_trajectory(fake, line_goal)
    return fake, line_start, line_goal, logger


def test_wait_hook_near_hook_restart_preserves_current_attitude():
    fake, line_start, line_goal, _logger = _fixed_hook_governor_fake()
    trim = np.asarray(
        euler_to_quat_wxyz(0.14, -0.08, 0.0),
        dtype=float,
    )
    current = np.asarray(
        euler_to_quat_wxyz(0.03, -0.02, 0.0),
        dtype=float,
    )
    recorded = np.asarray(
        euler_to_quat_wxyz(0.0, 0.0, 0.0),
        dtype=float,
    )
    fake._prehook_trim_q_wxyz = trim.copy()
    fake.q_wxyz = current.copy()
    fake._goal_quaternion = lambda: recorded.copy()
    fake.p_w = line_start + 0.90 * (line_goal - line_start)
    fake.fixed_hook_projected_restart = True

    MPCTrackTrajectoryAcados._set_linear_trajectory(fake, line_goal)

    np.testing.assert_allclose(
        fake.traj_start_pos,
        line_start + 0.90 * (line_goal - line_start),
    )
    assert quat_angular_distance_wxyz(
        fake.traj_start_q_wxyz,
        current,
    ) == pytest.approx(0.0, abs=1e-9)
    assert quat_angular_distance_wxyz(
        fake.traj_start_q_wxyz,
        trim,
    ) > 0.05

    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)
    first_reference = MPCTrackTrajectoryAcados._trajectory_stage_param(
        fake,
        0,
    )
    assert quat_angular_distance_wxyz(
        first_reference[3:7],
        current,
    ) == pytest.approx(0.0, abs=1e-9)


def test_fixed_hook_line_reference_is_measured_progress_not_wall_time():
    fake, line_start, _line_goal, _logger = _fixed_hook_governor_fake()
    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)

    # Even long after the nominal 5.56 s duration, a stationary vehicle gets
    # only the bounded 2 cm prediction lead, never an endpoint reference.
    references = [
        MPCTrackTrajectoryAcados._trajectory_stage_param(fake, k)
        for k in range(21)
    ]
    assert references[0][0] == pytest.approx(line_start[0])
    assert max(reference[0] for reference in references) == pytest.approx(
        0.02
    )
    assert fake._fixed_hook_line_progress_m == pytest.approx(0.0)

    fake.p_w[0] = 0.12
    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)
    progress_before_sampling = fake._fixed_hook_line_progress_m
    references = [
        MPCTrackTrajectoryAcados._trajectory_stage_param(fake, k)
        for k in range(21)
    ]
    assert references[0][0] == pytest.approx(0.12)
    assert max(reference[0] for reference in references) == pytest.approx(
        0.14
    )
    assert fake._fixed_hook_line_progress_m == pytest.approx(
        progress_before_sampling
    )
    assert all(reference[2] == pytest.approx(1.7) for reference in references)


@pytest.mark.parametrize(
    ('mission_state', 'expected_velocity_x_mps'),
    [
        ('GO_FORWARD', 0.09),
        ('GO_BACK', -0.09),
    ],
)
def test_fixed_hook_compatibility_line_includes_ned_velocity_feedforward(
    mission_state,
    expected_velocity_x_mps,
):
    fake, _line_start, _line_goal, _logger = _fixed_hook_governor_fake(
        mission_state=mission_state
    )
    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)

    first_stage = MPCTrackTrajectoryAcados._trajectory_stage_param(fake, 0)
    terminal_stage = MPCTrackTrajectoryAcados._trajectory_stage_param(
        fake,
        fake.N_horizon,
    )

    assert first_stage.shape == (12,)
    assert first_stage[7] == pytest.approx(1.0)
    np.testing.assert_allclose(
        first_stage[8:11],
        [expected_velocity_x_mps, 0.0, 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(terminal_stage[8:11], np.zeros(3))
    assert first_stage[11] == pytest.approx(math.sqrt(20.0))
    assert terminal_stage[11] == pytest.approx(math.sqrt(20.0))


def test_fixed_hook_line_velocity_feedforward_stays_in_ned_frame():
    fake, _line_start, _line_goal, _logger = _fixed_hook_governor_fake()
    yaw_quaternion = np.asarray(
        euler_to_quat_wxyz(0.0, 0.0, math.pi / 2.0)
    )
    fake.p_w = np.array([0.0, 0.0, 1.7])
    fake.q_wxyz = yaw_quaternion.copy()
    fake.traj_start_pos = fake.p_w.copy()
    fake.traj_goal_pos = np.array([0.0, 0.5, 1.7])
    fake.traj_start_q_wxyz = yaw_quaternion.copy()
    fake._goal_quaternion = lambda: yaw_quaternion.copy()
    fake._goal_yaw_static = lambda: math.pi / 2.0
    fake._fixed_hook_line_progress_m = 0.0
    fake._fixed_hook_line_actual_progress_m = 0.0
    fake._fixed_hook_line_raw_progress_m = 0.0
    fake._fixed_hook_line_interlock_active = False
    fake._fixed_hook_line_interlock_reason = ''

    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)
    reference = MPCTrackTrajectoryAcados._trajectory_stage_param(fake, 0)

    # Parameters 8:11 are an NED/world reference.  A +90-degree yaw must not
    # rotate world +Y into body +X before the residual reaches the solver.
    np.testing.assert_allclose(reference[8:11], [0.0, 0.09, 0.0], atol=1e-12)
    assert reference[11] == pytest.approx(math.sqrt(20.0))


@pytest.mark.parametrize(
    ('roll', 'pitch', 'yaw'),
    [
        (0.0, 0.0, 0.0),
        (-0.12, 0.17, 3.1155),
        (0.31, -0.22, -1.4),
    ],
)
def test_symbolic_world_velocity_rotation_matches_numeric(
    roll,
    pitch,
    yaw,
):
    q_symbol = ca.SX.sym('q_test', 4)
    rotation_function = ca.Function(
        'rotation_body_to_world_test',
        [q_symbol],
        [quat_to_rotation_matrix_sym_wxyz(q_symbol)],
    )
    quaternion = np.asarray(
        euler_to_quat_wxyz(roll, pitch, yaw),
        dtype=float,
    )
    symbolic_rotation = np.asarray(
        rotation_function(quaternion),
        dtype=float,
    )
    np.testing.assert_allclose(
        symbolic_rotation,
        quat_to_rotation_matrix_wxyz(quaternion),
        atol=2e-12,
    )


@pytest.mark.parametrize('mission_state', ['GO_FORWARD', 'GO_BACK'])
def test_position_mode_uses_time_reference_despite_corridor_error(
    mission_state,
):
    fake, line_start, line_goal, _logger = _fixed_hook_governor_fake(
        mission_state=mission_state,
        position_mode=True,
    )
    clock = [100.0]
    fake._now_sec = lambda: clock[0]
    line_delta = line_goal - line_start
    line_length_m = float(np.linalg.norm(line_delta))
    line_unit = line_delta / line_length_m
    line_normal = np.array([-line_unit[1], line_unit[0], 0.0])

    # Deliberately violate every compatibility corridor threshold and leave a
    # stale interlock flag latched.  Position mode must not call the measured-
    # progress governor, freeze, or key its reference to the measured pose.
    fake.p_w = (
        line_start
        + 0.12 * line_unit
        + 0.20 * line_normal
        + np.array([0.0, 0.0, 0.20])
    )
    fake.q_wxyz = np.asarray(
        euler_to_quat_wxyz(0.0, 0.0, math.radians(30.0))
    )
    fake._fixed_hook_line_interlock_active = True
    fake._fixed_hook_line_interlock_reason = 'stale compatibility state'
    assert MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake) is None

    sampled_progress = []
    for sample_time_sec in (100.0, 102.0, 106.0):
        clock[0] = sample_time_sec
        reference = MPCTrackTrajectoryAcados._trajectory_stage_param(fake, 0)
        sampled_progress.append(float(np.dot(
            reference[0:3] - line_start,
            line_unit,
        )))
        assert reference[2] == pytest.approx(line_start[2])

    assert sampled_progress == pytest.approx([0.0, 0.18, line_length_m])
    assert sampled_progress == sorted(sampled_progress)
    np.testing.assert_allclose(reference[0:3], line_goal)
    np.testing.assert_allclose(reference[8:11], np.zeros(3))
    assert fake._fixed_hook_line_progress_m == pytest.approx(0.0)


@pytest.mark.parametrize('mission_state', ['GO_FORWARD', 'GO_BACK'])
def test_position_mode_has_horizontal_ned_velocity_and_constant_weight(
    mission_state,
):
    fake, line_start, line_goal, _logger = _fixed_hook_governor_fake(
        mission_state=mission_state,
        position_mode=True,
    )
    clock = [102.0]
    fake._now_sec = lambda: clock[0]
    fake._fixed_hook_line_interlock_active = True
    line_unit = (line_goal - line_start) / np.linalg.norm(
        line_goal - line_start
    )

    references = [
        MPCTrackTrajectoryAcados._trajectory_stage_param(fake, k)
        for k in range(fake.N_horizon + 1)
    ]
    progress = [
        float(np.dot(reference[0:3] - line_start, line_unit))
        for reference in references
    ]
    assert progress == sorted(progress)
    assert progress[0] == pytest.approx(0.18)
    assert progress[-1] == pytest.approx(0.27)
    for reference in references:
        np.testing.assert_allclose(
            reference[8:11],
            0.09 * line_unit,
            atol=1e-12,
        )
        assert reference[8 + 2] == pytest.approx(0.0, abs=1e-12)
    assert all(
        reference[11] == pytest.approx(math.sqrt(20.0))
        for reference in references
    )


@pytest.mark.parametrize(
    ('position_mode', 'expected_vertical_integral'),
    [(True, 0.0), (False, -0.4)],
)
def test_position_mode_discards_upward_integral_on_line_entry(
    position_mode,
    expected_vertical_integral,
):
    fake, _line_start, line_goal, _logger = _fixed_hook_governor_fake(
        position_mode=position_mode,
    )
    fake.position_integral_error_world = np.array([0.2, -0.1, -0.4])
    fake.last_position_integral_update_sec = 42.0

    MPCTrackTrajectoryAcados._set_linear_trajectory(fake, line_goal)

    # The line points along NED +X, so only its normal Y compensation remains.
    # NED +Z is down: a negative Z integral is an upward command and is
    # cleared only by the real Position-like entry policy.
    np.testing.assert_allclose(
        fake.position_integral_error_world,
        [0.0, -0.1, expected_vertical_integral],
    )
    assert fake.last_position_integral_update_sec is None


@pytest.mark.parametrize('mission_state', ['GO_FORWARD', 'GO_BACK'])
def test_fixed_hook_line_interlock_freezes_with_default_velocity_weight(
    mission_state,
):
    fake, line_start, _line_goal, _logger = _fixed_hook_governor_fake(
        mission_state=mission_state
    )
    fake.p_w = line_start.copy()
    fake.p_w[1] += 0.031
    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)

    assert fake._fixed_hook_line_interlock_active is True
    progress_before_sampling = fake._fixed_hook_line_progress_m
    for k in range(fake.N_horizon + 1):
        reference = MPCTrackTrajectoryAcados._trajectory_stage_param(fake, k)
        assert reference.shape == (12,)
        np.testing.assert_allclose(reference[0:3], line_start)
        np.testing.assert_allclose(reference[8:11], np.zeros(3))
        assert reference[11] == pytest.approx(1.0)
    assert fake._fixed_hook_line_progress_m == pytest.approx(
        progress_before_sampling
    )


@pytest.mark.parametrize(
    ('offset', 'yaw_deg', 'depth_offset', 'expected_reason'),
    [
        (0.031, 0.0, 0.0, 'cross-track'),
        (0.0, 3.1, 0.0, 'yaw'),
        (0.0, 0.0, 0.031, 'depth'),
    ],
)
def test_fixed_hook_line_interlock_freezes_until_corridor_recovers(
    offset,
    yaw_deg,
    depth_offset,
    expected_reason,
):
    fake, _line_start, _line_goal, logger = _fixed_hook_governor_fake()
    fake.p_w = np.array([0.10, 0.0, 1.7])
    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)
    assert fake._fixed_hook_line_progress_m == pytest.approx(0.10)

    fake.p_w = np.array([0.16, offset, 1.7 + depth_offset])
    fake.q_wxyz = np.asarray(
        euler_to_quat_wxyz(0.0, 0.0, math.radians(yaw_deg))
    )
    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)
    assert fake._fixed_hook_line_interlock_active is True
    assert expected_reason in fake._fixed_hook_line_interlock_reason
    assert fake._fixed_hook_line_progress_m == pytest.approx(0.16)
    frozen = [
        MPCTrackTrajectoryAcados._trajectory_stage_param(fake, k)[0:3]
        for k in range(21)
    ]
    assert all(reference[0] == pytest.approx(0.16) for reference in frozen)
    assert all(reference[1] == pytest.approx(0.0) for reference in frozen)
    assert all(reference[2] == pytest.approx(1.7) for reference in frozen)

    fake.p_w = np.array([0.16, 0.0, 1.7])
    fake.q_wxyz = np.asarray(euler_to_quat_wxyz(0.0, 0.0, 0.0))
    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)
    assert fake._fixed_hook_line_interlock_active is False
    assert fake._fixed_hook_line_progress_m == pytest.approx(0.16)
    assert any(
        level == 'warning' and expected_reason in message
        for level, message in logger.messages
    )
    assert any(
        level == 'info' and 'interlock cleared' in message
        for level, message in logger.messages
    )


def test_fixed_hook_cross_track_uses_infinite_line_behind_start():
    fake, line_start, line_goal, _logger = _fixed_hook_governor_fake()
    line_unit = line_goal - line_start
    line_unit /= np.linalg.norm(line_unit)
    line_normal = np.array([-line_unit[1], line_unit[0], 0.0])

    fake.p_w = line_start - 0.04 * line_unit
    status = MPCTrackTrajectoryAcados._fixed_hook_line_governor_status(fake)
    assert status['raw_progress_m'] == pytest.approx(0.0)
    assert status['cross_track_m'] == pytest.approx(0.0, abs=1e-12)

    fake.p_w += 0.02 * line_normal
    status = MPCTrackTrajectoryAcados._fixed_hook_line_governor_status(fake)
    assert status['raw_progress_m'] == pytest.approx(0.0)
    assert status['cross_track_m'] == pytest.approx(0.02)


@pytest.mark.parametrize(
    ('enter_offset', 'near_offset', 'release_offset'),
    [
        ((0.031, 0.0, 0.0), (0.025, 0.0, 0.0), (0.023, 0.0, 0.0)),
        ((0.0, 3.1, 0.0), (0.0, 2.5, 0.0), (0.0, 2.3, 0.0)),
        ((0.0, 0.0, 0.031), (0.0, 0.0, 0.025), (0.0, 0.0, 0.023)),
    ],
)
def test_fixed_hook_line_interlock_uses_point_eight_release_hysteresis(
    enter_offset,
    near_offset,
    release_offset,
):
    fake, _line_start, _line_goal, _logger = _fixed_hook_governor_fake()

    def set_corridor_error(progress_m, offsets):
        cross_track_m, yaw_deg, depth_offset_m = offsets
        fake.p_w = np.array([
            progress_m,
            cross_track_m,
            1.7 + depth_offset_m,
        ])
        fake.q_wxyz = np.asarray(
            euler_to_quat_wxyz(0.0, 0.0, math.radians(yaw_deg))
        )

    set_corridor_error(0.10, (0.0, 0.0, 0.0))
    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)
    assert fake._fixed_hook_line_progress_m == pytest.approx(0.10)

    set_corridor_error(0.16, enter_offset)
    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)
    assert fake._fixed_hook_line_interlock_active is True
    assert fake._fixed_hook_line_progress_m == pytest.approx(0.16)

    # Once latched, falling below the 3 cm / 3 degree entry limit is not
    # enough.  The error must cross the configured 0.8 release boundary.
    set_corridor_error(0.17, near_offset)
    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)
    assert fake._fixed_hook_line_interlock_active is True
    assert fake._fixed_hook_line_progress_m == pytest.approx(0.17)

    set_corridor_error(0.18, release_offset)
    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)
    assert fake._fixed_hook_line_interlock_active is False
    assert fake._fixed_hook_line_progress_m == pytest.approx(0.18)


@pytest.mark.parametrize('mission_state', ['GO_FORWARD', 'GO_BACK'])
def test_fixed_hook_reference_never_retreats_after_backslide(mission_state):
    fake, line_start, line_goal, _logger = _fixed_hook_governor_fake(
        mission_state=mission_state
    )
    line_unit = line_goal - line_start
    line_unit /= np.linalg.norm(line_unit)
    fake.p_w = line_start + 0.30 * line_unit
    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)
    assert fake._fixed_hook_line_progress_m == pytest.approx(0.30)

    fake.p_w = line_start + 0.15 * line_unit
    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)
    # A backslide may create a larger recovery position error, but it must not
    # drag any reference back toward this phase's start.
    assert fake._fixed_hook_line_raw_progress_m == pytest.approx(0.15)
    assert fake._fixed_hook_line_progress_m == pytest.approx(0.30)
    references = [
        MPCTrackTrajectoryAcados._trajectory_stage_param(fake, k)[0:3]
        for k in range(21)
    ]
    progress_references = [
        float(np.dot(reference - line_start, line_unit))
        for reference in references
    ]
    assert min(progress_references) == pytest.approx(0.30)
    assert max(progress_references) == pytest.approx(0.30)


@pytest.mark.parametrize('mission_state', ['GO_FORWARD', 'GO_BACK'])
def test_fixed_hook_interlock_tracks_furthest_station_without_reversal(
    mission_state,
):
    fake, line_start, line_goal, _logger = _fixed_hook_governor_fake(
        mission_state=mission_state
    )
    line_unit = line_goal - line_start
    line_unit /= np.linalg.norm(line_unit)
    line_normal = np.array([-line_unit[1], line_unit[0], 0.0])

    fake.p_w = line_start + 0.30 * line_unit
    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)

    # Enter the interlock slightly farther along the mission direction. The
    # current longitudinal station, not the previous safe sample, is held.
    fake.p_w = line_start + 0.32 * line_unit + 0.031 * line_normal
    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)
    assert fake._fixed_hook_line_interlock_active is True
    assert fake._fixed_hook_line_progress_m == pytest.approx(0.32)

    # Inertia may carry the vehicle farther while the corridor is invalid.
    # Capture that station, then prove a later backslide cannot rewind it.
    fake.p_w = line_start + 0.36 * line_unit + 0.031 * line_normal
    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)
    fake.p_w = line_start + 0.15 * line_unit + 0.031 * line_normal
    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)
    assert fake._fixed_hook_line_progress_m == pytest.approx(0.36)

    for k in range(fake.N_horizon + 1):
        reference = MPCTrackTrajectoryAcados._trajectory_stage_param(fake, k)
        progress_reference = float(np.dot(
            reference[0:3] - line_start,
            line_unit,
        ))
        assert progress_reference == pytest.approx(0.36)
        np.testing.assert_allclose(reference[8:11], np.zeros(3))
        assert reference[11] == pytest.approx(1.0)

    # Once cross-track recovers, the controller still holds 0.36 m until the
    # vehicle catches up, then resumes only toward this phase's goal.
    fake.p_w = line_start + 0.15 * line_unit
    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)
    assert fake._fixed_hook_line_interlock_active is False
    recovered = [
        MPCTrackTrajectoryAcados._trajectory_stage_param(fake, k)[0:3]
        for k in range(fake.N_horizon + 1)
    ]
    recovered_progress = [
        float(np.dot(reference - line_start, line_unit))
        for reference in recovered
    ]
    assert min(recovered_progress) == pytest.approx(0.36)
    assert max(recovered_progress) == pytest.approx(0.36)

    fake.p_w = line_start + 0.36 * line_unit
    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)
    resumed = [
        MPCTrackTrajectoryAcados._trajectory_stage_param(fake, k)[0:3]
        for k in range(fake.N_horizon + 1)
    ]
    resumed_progress = [
        float(np.dot(reference - line_start, line_unit))
        for reference in resumed
    ]
    assert min(resumed_progress) == pytest.approx(0.36)
    assert max(resumed_progress) == pytest.approx(0.38)


def test_go_back_uses_same_governor_and_constant_depth():
    fake, line_start, _line_goal, _logger = _fixed_hook_governor_fake(
        mission_state='GO_BACK'
    )
    fake.p_w[0] = 0.35
    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)
    assert fake._fixed_hook_line_progress_m == pytest.approx(0.15)
    references = [
        MPCTrackTrajectoryAcados._trajectory_stage_param(fake, k)
        for k in range(21)
    ]
    assert references[0][0] == pytest.approx(line_start[0] - 0.15)
    assert min(reference[0] for reference in references) == pytest.approx(
        line_start[0] - 0.17
    )
    assert all(reference[1] == pytest.approx(0.0) for reference in references)
    assert all(reference[2] == pytest.approx(1.7) for reference in references)


def test_fixed_hook_completion_requires_actual_progress_within_lead():
    fake, _line_start, _line_goal, _logger = _fixed_hook_governor_fake()
    fake.p_w[0] = 0.479
    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)
    assert not MPCTrackTrajectoryAcados._trajectory_completion_reached(fake)

    fake.p_w[0] = 0.481
    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)
    assert MPCTrackTrajectoryAcados._trajectory_completion_reached(fake)


@pytest.mark.parametrize('mission_state', ['GO_FORWARD', 'GO_BACK'])
def test_no_retreat_anchor_cannot_complete_without_current_goal_progress(
    mission_state,
):
    fake, line_start, line_goal, _logger = _fixed_hook_governor_fake(
        mission_state=mission_state
    )
    line_unit = line_goal - line_start
    line_unit /= np.linalg.norm(line_unit)

    fake.p_w = line_start + 0.49 * line_unit
    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)
    assert fake._fixed_hook_line_progress_m == pytest.approx(0.49)

    # The current pose is just inside the broad 5 cm goal sphere, but it has
    # backslid too far for raw progress + lead to reach the 0.5 m endpoint.
    # Monotonic references must not create a false WAIT_HOOK/COMPLETE event.
    fake.p_w = line_start + 0.451 * line_unit
    assert np.linalg.norm(fake.p_w - line_goal) < 0.05
    assert not MPCTrackTrajectoryAcados._trajectory_completion_reached(fake)


@pytest.mark.parametrize(
    ('cross_track_m', 'yaw_deg', 'depth_offset_m'),
    [
        (0.031, 0.0, 0.0),
        (0.0, 3.1, 0.0),
        (0.0, 0.0, 0.031),
    ],
)
def test_fixed_hook_completion_cannot_bypass_fresh_corridor_gate(
    cross_track_m,
    yaw_deg,
    depth_offset_m,
):
    fake, _line_start, _line_goal, _logger = _fixed_hook_governor_fake()
    fake.p_w = np.array([0.49, 0.0, 1.7])
    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)
    assert fake._fixed_hook_line_interlock_active is False

    # Change the newest odometry after the previous governor tick. The fresh
    # completion check must reject it before _maybe_refresh can enter WAIT.
    fake.p_w = np.array([
        0.49,
        cross_track_m,
        1.7 + depth_offset_m,
    ])
    fake.q_wxyz = np.asarray(
        euler_to_quat_wxyz(0.0, 0.0, math.radians(yaw_deg))
    )
    assert not MPCTrackTrajectoryAcados._trajectory_completion_reached(fake)


def test_fixed_hook_completion_accepts_inside_all_strict_boundaries():
    fake, _line_start, _line_goal, _logger = _fixed_hook_governor_fake()
    fake.p_w = np.array([0.481, 0.029, 1.729])
    fake.q_wxyz = np.asarray(
        euler_to_quat_wxyz(0.0, 0.0, math.radians(2.9))
    )
    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)
    assert MPCTrackTrajectoryAcados._trajectory_completion_reached(fake)


def test_fixed_hook_governor_retains_only_line_normal_xy_integral():
    fake, _line_start, line_goal, _logger = _fixed_hook_governor_fake()
    fake.thrust_sat_norm = 0.12
    fake.force_axis_max_N = np.array([80.0, 80.0, 100.0])
    fake.position_integral_gain = 3.0
    fake.position_integral_force_limit_fraction = 0.07
    fake.position_integral_activation_error_m = 0.5
    fake.position_integral_max_dt_s = 0.1
    fake.position_integral_error_world = np.array([0.2, -0.1, 0.3])
    fake.last_position_integral_update_sec = 1.0
    fake._fixed_hook_transit_active = lambda: True
    fake._goal_position = lambda: line_goal.copy()
    fake._reset_position_integral = lambda: (
        MPCTrackTrajectoryAcados._reset_position_integral(fake)
    )
    fake._position_integral_reference_finished = lambda now_sec: (
        MPCTrackTrajectoryAcados._position_integral_reference_finished(
            fake, now_sec
        )
    )

    assert not MPCTrackTrajectoryAcados._position_integral_reference_finished(
        fake, now_sec=10000.0
    )
    MPCTrackTrajectoryAcados._force_with_position_integral(
        fake,
        np.zeros(3),
        now_sec=10000.0,
    )
    # The +X moving reference may not create an along-line X integral.  The
    # learned Y cross-current and Z buoyancy support must survive the phase
    # transition and the measured-progress transit.
    np.testing.assert_allclose(
        fake.position_integral_error_world,
        [0.0, -0.1, 0.3],
    )
    assert fake.last_position_integral_update_sec == pytest.approx(10000.0)


@pytest.mark.parametrize('mission_state', ['GO_FORWARD', 'GO_BACK'])
def test_fixed_hook_governor_integrates_cross_track_and_depth_not_along_track(
    mission_state,
):
    fake, line_start, line_goal, _logger = _fixed_hook_governor_fake(
        mission_state=mission_state
    )
    fake.thrust_sat_norm = 0.12
    fake.force_axis_max_N = np.array([80.0, 80.0, 100.0])
    fake.position_integral_gain = 3.0
    fake.position_integral_force_limit_fraction = 0.07
    fake.position_integral_activation_error_m = 0.5
    fake.position_integral_max_dt_s = 0.1
    fake.position_integral_error_world = np.zeros(3)
    fake.last_position_integral_update_sec = 10.0
    fake._fixed_hook_transit_active = lambda: True
    fake._goal_position = lambda: line_goal.copy()
    fake._reset_position_integral = lambda: (
        MPCTrackTrajectoryAcados._reset_position_integral(fake)
    )
    fake._position_integral_reference_finished = lambda now_sec: (
        MPCTrackTrajectoryAcados._position_integral_reference_finished(
            fake, now_sec
        )
    )

    line_xy = line_goal[0:2] - line_start[0:2]
    line_unit_xy = line_xy / np.linalg.norm(line_xy)
    line_normal_xy = np.array([-line_unit_xy[1], line_unit_xy[0]])
    fake.p_w = line_start.copy()
    fake.p_w[0:2] += 0.04 * line_normal_xy
    fake.p_w[2] += 0.05

    force = MPCTrackTrajectoryAcados._force_with_position_integral(
        fake,
        np.zeros(3),
        now_sec=10.1,
    )

    integral_xy = fake.position_integral_error_world[0:2]
    assert np.dot(integral_xy, line_unit_xy) == pytest.approx(0.0, abs=1e-12)
    assert np.dot(integral_xy, line_normal_xy) == pytest.approx(-0.004)
    assert fake.position_integral_error_world[2] == pytest.approx(-0.005)
    assert np.dot(force[0:2], line_normal_xy) < 0.0
    assert force[2] < 0.0


def test_fixed_hook_interlock_reports_a_changed_blocking_reason():
    fake, _line_start, _line_goal, logger = _fixed_hook_governor_fake()
    fake.p_w = np.array([0.1, 0.04, 1.7])
    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)
    assert 'cross-track' in fake._fixed_hook_line_interlock_reason

    fake.p_w = np.array([0.1, 0.0, 1.7])
    fake.q_wxyz = np.asarray(
        euler_to_quat_wxyz(0.0, 0.0, math.radians(4.0))
    )
    MPCTrackTrajectoryAcados._update_fixed_hook_line_governor(fake)

    assert 'yaw' in fake._fixed_hook_line_interlock_reason
    warnings = [
        message for level, message in logger.messages
        if level == 'warning'
    ]
    assert any('cross-track' in message for message in warnings)
    assert any('yaw' in message for message in warnings)


def test_plan_holds_while_stale_uncancellable_worker_finishes():
    fake, executor, planned, revoked, _prehook = _planning_fake()
    stale_future = Future()
    assert stale_future.set_running_or_notify_cancel()
    fake._prehook_replan_future = stale_future
    fake._prehook_replan_context = {
        'generation': fake._prehook_plan_generation,
        'goal_signature': ('old-goal',),
        'request_level': 'optional',
    }

    result = MPCTrackTrajectoryAcados._plan_to_prehook_from_current(
        fake, reason='restart while old worker runs'
    )

    assert result is True
    assert fake.mission_state == 'PLAN_TO_PREHOOK'
    assert fake._prehook_planner_hold_active is True
    assert fake._prehook_replan_future is stale_future
    assert executor.calls == []
    assert planned == []
    assert revoked == []

    stale_future.set_result(([(0.0, 0.0), (1.0, 1.0)], None))
    MPCTrackTrajectoryAcados._update_dynamic_prehook_phase(fake)

    assert fake.mission_state == 'PLAN_TO_PREHOOK'
    assert len(executor.calls) == 1
    assert fake._prehook_replan_future is executor.futures[0]
    assert fake._prehook_replan_context['request_level'] == 'initial'


def test_running_track_worker_is_retained_through_dwell_drift_to_plan():
    fake, executor, _planned, revoked, _prehook = _planning_fake()
    running_future = Future()
    assert running_future.set_running_or_notify_cancel()
    fake.mission_state = 'TRACK_TO_PREHOOK'
    fake._prehook_replan_future = running_future
    fake._prehook_replan_context = {
        'generation': fake._prehook_plan_generation,
        'goal_signature': ('prehook-goal',),
        'request_level': 'optional',
        'reason': 'candidate_path_improvement_check',
    }
    at_goal = [True]
    fake._trajectory_completion_reached = lambda: at_goal[0]
    fake._prehook_reached_since_sec = None
    fake.terminal_hold_goal_signature = None
    fake.last_goal_signature = None
    fake._enter_terminal_hold = lambda signature: (
        setattr(fake, 'terminal_hold_goal_signature', signature)
    )

    MPCTrackTrajectoryAcados._update_dynamic_prehook_phase(fake)

    assert fake.mission_state == 'PREHOOK_REACHED'
    assert fake._prehook_replan_future is running_future
    retained_generation = fake._prehook_plan_generation

    at_goal[0] = False
    MPCTrackTrajectoryAcados._update_dynamic_prehook_phase(fake)

    assert fake.mission_state == 'PLAN_TO_PREHOOK'
    assert fake._prehook_planner_hold_active is True
    assert fake._prehook_replan_future is running_future
    assert fake._prehook_plan_generation > retained_generation
    assert executor.calls == []
    assert revoked == []

    running_future.set_result(([(1.0, -0.2), (4.8, 0.1)], None))
    MPCTrackTrajectoryAcados._update_dynamic_prehook_phase(fake)

    assert fake.mission_state == 'PLAN_TO_PREHOOK'
    assert len(executor.calls) == 1
    assert fake._prehook_replan_future is executor.futures[0]
    assert fake._prehook_replan_context['request_level'] == 'initial'


def test_planner_hold_disables_position_integral_toward_prehook():
    fake = SimpleNamespace(
        mission_state='PLAN_TO_PREHOOK',
        _prehook_planner_hold_active=True,
        thrust_sat_norm=0.2,
        force_axis_max_N=np.array([40.0, 40.0, 50.0], dtype=float),
        position_integral_error_world=np.array(
            [0.8, -0.4, 2.0], dtype=float
        ),
        last_position_integral_update_sec=5.0,
    )
    fake._reset_position_integral = lambda: (
        MPCTrackTrajectoryAcados._reset_position_integral(fake)
    )
    base_force = np.array([1.0, -2.0, 3.0], dtype=float)

    result = MPCTrackTrajectoryAcados._force_with_position_integral(
        fake, base_force, now_sec=6.0
    )

    np.testing.assert_allclose(result, base_force)
    np.testing.assert_allclose(fake.position_integral_error_world, 0.0)
    assert fake.last_position_integral_update_sec is None


def test_fault_hold_preserves_vertical_support_against_positive_buoyancy():
    """The attitude-timeout transition must not step the heave force down."""
    fake = SimpleNamespace(
        mission_state='TRACK_TO_PREHOOK',
        _prehook_planner_hold_active=True,
        _prehook_attitude_fault_latched=True,
        _prehook_planner_hold_pos=np.array([4.8, 0.1, 1.75]),
        p_w=np.array([4.8, 0.1, 1.75]),
        q_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        thrust_sat_norm=0.12,
        force_axis_max_N=np.array([88.0, 88.0, 137.0]),
        position_integral_gain=3.0,
        position_integral_force_limit_fraction=0.07,
        position_integral_activation_error_m=0.50,
        position_integral_max_dt_s=0.10,
        # 1.85 m*s * 3 N/(m*s) = 5.55 N, matching the vertical support
        # removed by the 2026-08-26 real fault-hold transition.
        position_integral_error_world=np.array([0.0, 0.0, 1.85]),
        last_position_integral_update_sec=None,
    )
    fake._reset_position_integral = lambda: (
        MPCTrackTrajectoryAcados._reset_position_integral(fake)
    )
    fake._clear_position_integral_xy_preserve_z = lambda: (
        MPCTrackTrajectoryAcados._clear_position_integral_xy_preserve_z(
            fake
        )
    )
    base_force = np.array([0.0, 0.0, 4.82])

    force = MPCTrackTrajectoryAcados._force_with_position_integral(
        fake,
        base_force,
        now_sec=100.0,
    )

    np.testing.assert_allclose(force, [0.0, 0.0, 10.37])
    np.testing.assert_allclose(
        fake.position_integral_error_world,
        [0.0, 0.0, 1.85],
    )
    assert fake.last_position_integral_update_sec == pytest.approx(100.0)


def test_fault_hold_preserves_world_vertical_support_when_vehicle_is_tilted():
    """Preservation is in NED world Z, not a body-thrust component."""
    quaternion = np.asarray(
        euler_to_quat_wxyz(
            math.radians(12.0),
            math.radians(-7.0),
            math.radians(80.0),
        )
    )
    rotation_body_to_world = quat_to_rotation_matrix_wxyz(quaternion)
    base_world = np.array([0.0, 0.0, 4.82])
    fake = SimpleNamespace(
        mission_state='TRACK_TO_PREHOOK',
        _prehook_planner_hold_active=True,
        _prehook_attitude_fault_latched=True,
        _prehook_planner_hold_pos=np.array([4.8, 0.1, 1.75]),
        p_w=np.array([4.8, 0.1, 1.75]),
        q_wxyz=quaternion,
        thrust_sat_norm=0.12,
        force_axis_max_N=np.array([88.0, 88.0, 137.0]),
        position_integral_gain=3.0,
        position_integral_force_limit_fraction=0.07,
        position_integral_activation_error_m=0.50,
        position_integral_max_dt_s=0.10,
        position_integral_error_world=np.array([0.0, 0.0, 1.85]),
        last_position_integral_update_sec=None,
    )
    fake._reset_position_integral = lambda: (
        MPCTrackTrajectoryAcados._reset_position_integral(fake)
    )
    fake._clear_position_integral_xy_preserve_z = lambda: (
        MPCTrackTrajectoryAcados._clear_position_integral_xy_preserve_z(
            fake
        )
    )

    force_body = MPCTrackTrajectoryAcados._force_with_position_integral(
        fake,
        rotation_body_to_world.T @ base_world,
        now_sec=100.0,
    )

    np.testing.assert_allclose(
        rotation_body_to_world @ force_body,
        [0.0, 0.0, 10.37],
        atol=1e-12,
    )


def test_fault_hold_large_displacement_freezes_vertical_support():
    """A drift beyond the activation gate must not drop buoyancy support."""
    fake = SimpleNamespace(
        mission_state='TRACK_TO_PREHOOK',
        _prehook_planner_hold_active=True,
        _prehook_attitude_fault_latched=True,
        _prehook_planner_hold_pos=np.array([4.8, 0.1, 1.75]),
        p_w=np.array([5.4, 0.1, 1.75]),
        q_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        thrust_sat_norm=0.12,
        force_axis_max_N=np.array([88.0, 88.0, 137.0]),
        position_integral_gain=3.0,
        position_integral_force_limit_fraction=0.07,
        position_integral_activation_error_m=0.50,
        position_integral_max_dt_s=0.10,
        position_integral_error_world=np.array([0.4, -0.2, 1.85]),
        last_position_integral_update_sec=99.0,
    )
    fake._reset_position_integral = lambda: (
        MPCTrackTrajectoryAcados._reset_position_integral(fake)
    )
    fake._clear_position_integral_xy_preserve_z = lambda: (
        MPCTrackTrajectoryAcados._clear_position_integral_xy_preserve_z(
            fake
        )
    )

    force = MPCTrackTrajectoryAcados._force_with_position_integral(
        fake,
        np.array([0.0, 0.0, 4.82]),
        now_sec=100.0,
    )

    np.testing.assert_allclose(force, [0.0, 0.0, 10.37])
    np.testing.assert_allclose(
        fake.position_integral_error_world,
        [0.0, 0.0, 1.85],
    )
    assert fake.last_position_integral_update_sec is None


def test_running_optional_plan_does_not_suppress_safety_check_or_upgrade(
    monkeypatch,
):
    now = [5.0]
    monkeypatch.setattr(
        'bluerov2_control.mpc_track_trajectory_acados.time.monotonic',
        lambda: now[0],
    )
    optional_future = Future()
    assert optional_future.set_running_or_notify_cancel()
    status_calls = []
    mandatory_requests = []
    logger = _Logger()
    parameters = {
        'prehook_planner_check_rate_hz': 2.0,
        'prehook_replan_deviation_m': 0.3,
    }
    fake = SimpleNamespace(
        mission_state='TRACK_TO_PREHOOK',
        p_w=np.array([2.0, 0.3, 1.2], dtype=float),
        q_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        active_goal_pos=np.array([4.8, 0.1, 1.7], dtype=float),
        _prehook_plan_generation=4,
        _prehook_replan_future=optional_future,
        _prehook_replan_context={
            'generation': 4,
            'goal_signature': ('prehook',),
            'reason': 'candidate_path_improvement_check',
            'request_level': 'optional',
        },
        _prehook_mandatory_replan_pending=False,
        _prehook_last_check_monotonic=4.0,
        _prehook_deviation_since_monotonic=None,
        _prehook_planner_hold_active=False,
        _prehook_planner_hold_pos=np.zeros(3, dtype=float),
        _prehook_planner_hold_q_wxyz=np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=float
        ),
        position_integral_error_world=np.ones(3, dtype=float),
        last_position_integral_update_sec=4.0,
        _goal_signature=lambda: ('prehook',),
        _prehook_path_status=lambda: (
            status_calls.append(now[0])
            or {
                'safe': False,
                'cross_track_m': 0.1,
                'remaining_cost_m': 2.0,
            }
        ),
        _set_path_trajectory=lambda *_args, **_kwargs: pytest.fail(
            'an optional result must not be installed as a safety repair'
        ),
        _candidate_path_from_current=lambda path: list(path),
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        get_logger=lambda: logger,
        _revoke_mission_enable=lambda reason: pytest.fail(reason),
    )
    fake._reset_position_integral = lambda: (
        MPCTrackTrajectoryAcados._reset_position_integral(fake)
    )
    fake._activate_prehook_planner_hold = lambda: (
        MPCTrackTrajectoryAcados._activate_prehook_planner_hold(fake)
    )

    def submit_request(**kwargs):
        mandatory_requests.append(kwargs)
        fake._prehook_replan_future = Future()
        return True

    fake._submit_prehook_plan_request = submit_request

    MPCTrackTrajectoryAcados._submit_prehook_replan_if_needed(fake)

    assert status_calls == [5.0]
    assert fake._prehook_planner_hold_active is True
    np.testing.assert_allclose(fake._prehook_planner_hold_pos, fake.p_w)
    assert fake._prehook_mandatory_replan_pending is True
    assert fake._prehook_replan_future is optional_future
    assert mandatory_requests == []

    optional_future.set_result(([(2.0, 0.3), (4.8, 0.1)], None))
    MPCTrackTrajectoryAcados._poll_prehook_replan(fake)

    assert len(mandatory_requests) == 1
    assert mandatory_requests[0]['request_level'] == 'mandatory'
    assert fake._prehook_mandatory_replan_pending is False
    assert fake._prehook_replan_future is not optional_future


def _attitude_wait_fake():
    logger = _Logger()
    revoked = []
    ros_time = [100.0]
    goal_quaternion = np.asarray(
        euler_to_quat_wxyz(0.0, 0.0, math.pi / 2.0),
        dtype=float,
    )
    parameters = {
        'goal_reached_tol_m': 0.05,
        'goal_reached_orientation_tol_rad': math.radians(5.0),
        'prehook_reached_orientation_tol_rad': 0.0,
        'prehook_reached_yaw_tol_rad': math.radians(3.0),
        'prehook_reached_forward_axis_tol_rad': 0.0,
        'prehook_attitude_alignment_timeout_s': 30.0,
        'prehook_attitude_wait_exit_hysteresis_ratio': 1.5,
    }
    fake = SimpleNamespace(
        mission_state='TRACK_TO_PREHOOK',
        p_w=np.array([4.81, 0.10, 1.705], dtype=float),
        traj_goal_pos=np.array([4.80, 0.10, 1.70], dtype=float),
        q_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        traj_active=True,
        traj_start_time_sec=0.0,
        traj_duration_sec=10.0,
        fixed_hook_depth_tolerance_m=0.03,
        _prehook_attitude_wait_since_monotonic=None,
        _prehook_attitude_fault_latched=False,
        _prehook_planner_hold_active=False,
        _prehook_planner_hold_pos=np.zeros(3, dtype=float),
        _prehook_planner_hold_q_wxyz=np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=float
        ),
        position_integral_error_world=np.array(
            [0.4, -0.2, 1.85], dtype=float
        ),
        last_position_integral_update_sec=9.0,
        _prehook_plan_generation=0,
        _prehook_replan_future=None,
        _prehook_mandatory_replan_pending=False,
        _prehook_replan_context=None,
        _now_sec=lambda: ros_time[0],
        _goal_quaternion=lambda: goal_quaternion.copy(),
        get_parameter=lambda name: SimpleNamespace(
            value=parameters[name]
        ),
        get_logger=lambda: logger,
        _revoke_mission_enable=lambda reason: revoked.append(reason),
    )
    fake._position_integral_reference_finished = lambda now_sec: (
        MPCTrackTrajectoryAcados._position_integral_reference_finished(
            fake,
            now_sec,
        )
    )
    fake._prehook_orientation_tolerance_rad = lambda: (
        MPCTrackTrajectoryAcados._prehook_orientation_tolerance_rad(fake)
    )
    fake._prehook_yaw_tolerance_rad = lambda: (
        MPCTrackTrajectoryAcados._prehook_yaw_tolerance_rad(fake)
    )
    fake._prehook_forward_axis_tolerance_rad = lambda: (
        MPCTrackTrajectoryAcados._prehook_forward_axis_tolerance_rad(fake)
    )
    fake._prehook_attitude_wait_active = lambda status: (
        MPCTrackTrajectoryAcados._prehook_attitude_wait_active(
            fake, status
        )
    )
    fake._reset_position_integral = lambda: (
        MPCTrackTrajectoryAcados._reset_position_integral(fake)
    )
    fake._clear_position_integral_xy_preserve_z = lambda: (
        MPCTrackTrajectoryAcados._clear_position_integral_xy_preserve_z(
            fake
        )
    )
    fake._retire_prehook_plan_request = lambda: (
        MPCTrackTrajectoryAcados._retire_prehook_plan_request(fake)
    )
    fake._activate_prehook_planner_hold = lambda **kwargs: (
        MPCTrackTrajectoryAcados._activate_prehook_planner_hold(
            fake,
            **kwargs,
        )
    )
    return fake, ros_time, logger, revoked


def test_prehook_alignment_status_identifies_attitude_only_stall():
    fake, _ros_time, _logger, _revoked = _attitude_wait_fake()

    status = MPCTrackTrajectoryAcados._prehook_alignment_status(fake)

    assert status['position_error_m'] == pytest.approx(
        math.hypot(0.01, 0.005)
    )
    assert status['position_ok'] is True
    assert status['depth_error_m'] == pytest.approx(0.005)
    assert status['depth_ok'] is True
    assert status['translation_ready'] is True
    assert status['attitude_error_rad'] == pytest.approx(math.pi / 2.0)
    assert status['yaw_error_rad'] == pytest.approx(math.pi / 2.0)
    assert status['yaw_tolerance_rad'] == pytest.approx(
        math.radians(3.0)
    )
    assert status['yaw_ok'] is False
    assert status['attitude_ok'] is False
    assert status['complete'] is False
    assert status['reference_finished'] is True


def test_prehook_attitude_wait_latches_current_pose_hold_at_timeout():
    fake, _ros_time, logger, revoked = _attitude_wait_fake()
    status = MPCTrackTrajectoryAcados._prehook_alignment_status(fake)

    assert not MPCTrackTrajectoryAcados._update_prehook_attitude_wait(
        fake,
        status,
        now_monotonic=10.0,
    )
    assert fake._prehook_attitude_wait_since_monotonic == pytest.approx(10.0)
    assert any(
        level == 'warning'
        and 'blocked by attitude only' in message
        and '90.00deg' in message
        for level, message in logger.messages
    )
    assert not MPCTrackTrajectoryAcados._update_prehook_attitude_wait(
        fake,
        status,
        now_monotonic=39.999,
    )
    assert revoked == []
    assert MPCTrackTrajectoryAcados._update_prehook_attitude_wait(
        fake,
        status,
        now_monotonic=40.0,
    )
    assert revoked == []
    assert fake._prehook_attitude_fault_latched is True
    assert fake._prehook_planner_hold_active is True
    np.testing.assert_allclose(fake._prehook_planner_hold_pos, fake.p_w)
    np.testing.assert_allclose(
        fake._prehook_planner_hold_q_wxyz,
        fake.q_wxyz,
    )
    np.testing.assert_allclose(
        fake.position_integral_error_world,
        [0.0, 0.0, 1.85],
    )
    assert fake.last_position_integral_update_sec is None
    assert any(
        level == 'error'
        and 'current-pose fault hold is latched' in message
        and 'heartbeat remains active' in message
        for level, message in logger.messages
    )

    fake.p_w[0] += 0.20
    lost_translation = MPCTrackTrajectoryAcados._prehook_alignment_status(
        fake
    )
    assert not MPCTrackTrajectoryAcados._update_prehook_attitude_wait(
        fake,
        lost_translation,
        now_monotonic=41.0,
    )
    assert fake._prehook_attitude_wait_since_monotonic is None


def test_zero_attitude_timeout_keeps_aligning_indefinitely():
    fake, _ros_time, logger, revoked = _attitude_wait_fake()
    parameters = {
        'goal_reached_tol_m': 0.05,
        'goal_reached_orientation_tol_rad': math.radians(5.0),
        'prehook_reached_orientation_tol_rad': 0.0,
        'prehook_reached_yaw_tol_rad': math.radians(3.0),
        'prehook_reached_forward_axis_tol_rad': 0.0,
        'prehook_attitude_alignment_timeout_s': 0.0,
        'prehook_attitude_wait_exit_hysteresis_ratio': 1.5,
    }
    fake.get_parameter = lambda name: SimpleNamespace(
        value=parameters[name]
    )
    status = MPCTrackTrajectoryAcados._prehook_alignment_status(fake)

    assert not MPCTrackTrajectoryAcados._update_prehook_attitude_wait(
        fake,
        status,
        now_monotonic=10.0,
    )
    assert not MPCTrackTrajectoryAcados._update_prehook_attitude_wait(
        fake,
        status,
        now_monotonic=10000.0,
    )

    assert fake._prehook_attitude_wait_since_monotonic == pytest.approx(10.0)
    assert fake._prehook_attitude_fault_latched is False
    assert fake._prehook_planner_hold_active is False
    assert fake._prehook_plan_generation == 0
    np.testing.assert_allclose(
        fake.position_integral_error_world,
        [0.4, -0.2, 1.85],
    )
    assert fake.last_position_integral_update_sec == pytest.approx(9.0)
    assert revoked == []
    assert any(
        level == 'warning'
        and 'elapsed-time timeout disabled' in message
        for level, message in logger.messages
    )
    assert not any(level == 'error' for level, _message in logger.messages)


def test_attitude_timeout_waits_for_reference_and_mandatory_hold():
    fake, ros_time, _logger, revoked = _attitude_wait_fake()
    ros_time[0] = 5.0
    unfinished = MPCTrackTrajectoryAcados._prehook_alignment_status(fake)
    assert unfinished['translation_ready'] is True
    assert unfinished['reference_finished'] is False
    assert not MPCTrackTrajectoryAcados._update_prehook_attitude_wait(
        fake,
        unfinished,
        now_monotonic=10.0,
    )
    assert fake._prehook_attitude_wait_since_monotonic is None

    ros_time[0] = 100.0
    finished = MPCTrackTrajectoryAcados._prehook_alignment_status(fake)
    fake._prehook_planner_hold_active = True
    assert not MPCTrackTrajectoryAcados._update_prehook_attitude_wait(
        fake,
        finished,
        now_monotonic=50.0,
    )
    assert fake._prehook_attitude_wait_since_monotonic is None
    assert revoked == []


def test_attitude_wait_timer_survives_small_position_boundary_jitter():
    fake, _ros_time, _logger, _revoked = _attitude_wait_fake()
    status = MPCTrackTrajectoryAcados._prehook_alignment_status(fake)
    assert not MPCTrackTrajectoryAcados._update_prehook_attitude_wait(
        fake, status, now_monotonic=10.0
    )
    assert fake._prehook_attitude_wait_since_monotonic == pytest.approx(10.0)

    # Outside the 5 cm entry gate, but inside its 1.5x exit gate.
    fake.p_w = np.array([4.86, 0.10, 1.705], dtype=float)
    jittered = MPCTrackTrajectoryAcados._prehook_alignment_status(fake)
    assert jittered['translation_ready'] is False
    assert not MPCTrackTrajectoryAcados._update_prehook_attitude_wait(
        fake, jittered, now_monotonic=20.0
    )
    assert fake._prehook_attitude_wait_since_monotonic == pytest.approx(10.0)

    # Truly leaving the 7.5 cm exit gate clears the continuous timer.
    fake.p_w = np.array([4.88, 0.10, 1.705], dtype=float)
    outside = MPCTrackTrajectoryAcados._prehook_alignment_status(fake)
    assert not MPCTrackTrajectoryAcados._update_prehook_attitude_wait(
        fake, outside, now_monotonic=21.0
    )
    assert fake._prehook_attitude_wait_since_monotonic is None


def test_attitude_wait_timer_clears_outside_depth_hysteresis_gate():
    """A nonzero depth error must be compared, never treated as truthy."""
    fake, _ros_time, _logger, _revoked = _attitude_wait_fake()
    status = MPCTrackTrajectoryAcados._prehook_alignment_status(fake)
    assert not MPCTrackTrajectoryAcados._update_prehook_attitude_wait(
        fake, status, now_monotonic=10.0
    )
    assert fake._prehook_attitude_wait_since_monotonic == pytest.approx(10.0)

    # Position remains inside its 7.5 cm exit gate, but 4.6 cm depth error is
    # outside the independent 3 cm * 1.5 depth exit gate.
    fake.p_w = np.array([4.80, 0.10, 1.746], dtype=float)
    outside_depth = MPCTrackTrajectoryAcados._prehook_alignment_status(fake)
    assert outside_depth['position_error_m'] < 0.075
    assert outside_depth['depth_error_m'] > 0.045
    assert not MPCTrackTrajectoryAcados._update_prehook_attitude_wait(
        fake, outside_depth, now_monotonic=11.0
    )
    assert fake._prehook_attitude_wait_since_monotonic is None


def test_hysteresis_band_keeps_optional_replans_suppressed():
    status = {
        'position_error_m': 0.060,
        'position_tolerance_m': 0.050,
        'depth_error_m': 0.040,
        'depth_tolerance_m': 0.030,
        'translation_ready': False,
        'attitude_ok': False,
    }
    calls = []
    fake = SimpleNamespace(
        mission_state='TRACK_TO_PREHOOK',
        _prehook_attitude_fault_latched=False,
        _prehook_attitude_wait_since_monotonic=10.0,
        _dynamic_prehook_planner_enabled=lambda: True,
        _prehook_alignment_status=lambda: dict(status),
        _poll_prehook_replan=lambda **kwargs: calls.append(
            ('poll', kwargs['suppress_optional_improvement'])
        ),
        _mission_allowed=lambda: True,
        _trajectory_completion_reached=lambda: False,
        _update_prehook_attitude_wait=lambda _status: False,
        _submit_prehook_replan_if_needed=lambda **kwargs: calls.append(
            ('submit', kwargs['suppress_optional_improvement'])
        ),
        get_parameter=lambda name: SimpleNamespace(value={
            'prehook_attitude_wait_exit_hysteresis_ratio': 1.5,
        }[name]),
    )
    fake._prehook_attitude_wait_active = lambda current_status: (
        MPCTrackTrajectoryAcados._prehook_attitude_wait_active(
            fake, current_status
        )
    )

    assert MPCTrackTrajectoryAcados._update_dynamic_prehook_phase(fake)
    assert calls == [('poll', True), ('submit', True)]


def test_attitude_wait_timer_clears_when_attitude_becomes_ready():
    fake, _ros_time, _logger, _revoked = _attitude_wait_fake()
    status = MPCTrackTrajectoryAcados._prehook_alignment_status(fake)
    MPCTrackTrajectoryAcados._update_prehook_attitude_wait(
        fake, status, now_monotonic=10.0
    )
    fake.q_wxyz = status['goal_quaternion'].copy()
    ready = MPCTrackTrajectoryAcados._prehook_alignment_status(fake)
    assert ready['complete'] is True
    assert not MPCTrackTrajectoryAcados._update_prehook_attitude_wait(
        fake, ready, now_monotonic=11.0
    )
    assert fake._prehook_attitude_wait_since_monotonic is None


def test_independent_prehook_orientation_tolerance_overrides_global_gate():
    fake, _ros_time, _logger, _revoked = _attitude_wait_fake()
    fake.get_parameter = lambda name: SimpleNamespace(value={
        'goal_reached_tol_m': 0.05,
        'goal_reached_orientation_tol_rad': math.radians(5.0),
        'prehook_reached_orientation_tol_rad': math.radians(8.0),
        'prehook_reached_yaw_tol_rad': math.radians(3.0),
        'prehook_reached_forward_axis_tol_rad': 0.0,
        'prehook_attitude_alignment_timeout_s': 30.0,
        'prehook_attitude_wait_exit_hysteresis_ratio': 1.5,
    }[name])
    goal = fake._goal_quaternion()
    goal_roll, goal_pitch, goal_yaw = quat_to_rpy_wxyz(goal)
    fake.q_wxyz = np.asarray(euler_to_quat_wxyz(
        goal_roll,
        goal_pitch,
        goal_yaw + math.radians(7.0),
    ))

    status = MPCTrackTrajectoryAcados._prehook_alignment_status(fake)
    assert status['attitude_tolerance_rad'] == pytest.approx(
        math.radians(8.0)
    )
    assert status['attitude_error_rad'] == pytest.approx(
        math.radians(7.0)
    )
    assert status['orientation_ok'] is True
    assert status['yaw_ok'] is False
    assert status['attitude_ok'] is False
    assert not MPCTrackTrajectoryAcados._update_prehook_attitude_wait(
        fake,
        status,
        now_monotonic=10.0,
    )
    assert fake._prehook_attitude_wait_since_monotonic == pytest.approx(10.0)


def test_forward_axis_gate_accepts_roll_within_full_attitude_envelope():
    fake, _ros_time, _logger, _revoked = _attitude_wait_fake()
    fake.get_parameter = lambda name: SimpleNamespace(value={
        'goal_reached_tol_m': 0.05,
        'goal_reached_orientation_tol_rad': math.radians(5.0),
        'prehook_reached_orientation_tol_rad': math.radians(8.0),
        'prehook_reached_yaw_tol_rad': math.radians(3.0),
        'prehook_reached_forward_axis_tol_rad': math.radians(3.0),
        'prehook_attitude_alignment_timeout_s': 0.0,
        'prehook_attitude_wait_exit_hysteresis_ratio': 1.5,
    }[name])
    goal_roll, goal_pitch, goal_yaw = quat_to_rpy_wxyz(
        fake._goal_quaternion()
    )
    fake.q_wxyz = np.asarray(euler_to_quat_wxyz(
        goal_roll + math.radians(7.0),
        goal_pitch,
        goal_yaw,
    ))

    status = MPCTrackTrajectoryAcados._prehook_alignment_status(fake)

    assert status['attitude_error_rad'] == pytest.approx(
        math.radians(7.0)
    )
    assert status['orientation_ok'] is True
    assert status['forward_axis_error_rad'] == pytest.approx(
        0.0,
        abs=1e-7,
    )
    assert status['forward_axis_tolerance_rad'] == pytest.approx(
        math.radians(3.0)
    )
    assert status['forward_axis_ok'] is True
    assert status['yaw_ok'] is True
    assert status['attitude_ok'] is True
    assert status['complete'] is True


def test_dynamic_completion_rejects_forward_axis_outside_gate():
    goal = np.asarray(euler_to_quat_wxyz(0.0, 0.0, math.pi / 2.0))
    parameters = {
        'goal_reached_tol_m': 0.05,
        'goal_reached_orientation_tol_rad': math.radians(5.0),
        'prehook_reached_orientation_tol_rad': math.radians(8.0),
        'prehook_reached_yaw_tol_rad': math.radians(3.0),
        'prehook_reached_forward_axis_tol_rad': math.radians(3.0),
        'hold_attitude': True,
        'use_box_recovery_mission': False,
    }
    fake = SimpleNamespace(
        mission_state='TRACK_TO_PREHOOK',
        p_w=np.array([4.9, 0.1, 1.7]),
        traj_goal_pos=np.array([4.9, 0.1, 1.7]),
        q_wxyz=np.asarray(euler_to_quat_wxyz(
            math.radians(7.0),
            0.0,
            math.pi / 2.0,
        )),
        fixed_hook_depth_tolerance_m=0.03,
        _dynamic_prehook_planner_enabled=lambda: True,
        _pre_approach_waypoint_enabled=lambda: True,
        _goal_quaternion=lambda: goal.copy(),
        _now_sec=lambda: 100.0,
        _position_integral_reference_finished=lambda _now_sec: True,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
    )
    fake._prehook_orientation_tolerance_rad = lambda: (
        MPCTrackTrajectoryAcados._prehook_orientation_tolerance_rad(fake)
    )
    fake._prehook_yaw_tolerance_rad = lambda: (
        MPCTrackTrajectoryAcados._prehook_yaw_tolerance_rad(fake)
    )
    fake._prehook_forward_axis_tolerance_rad = lambda: (
        MPCTrackTrajectoryAcados._prehook_forward_axis_tolerance_rad(fake)
    )

    # Pure roll leaves body-X unchanged, so 7 degrees is accepted by the
    # independent 8 degree full-attitude safety envelope.
    assert MPCTrackTrajectoryAcados._trajectory_completion_reached(fake)

    # A 4 degree pitch error remains inside that 8 degree envelope and keeps
    # yaw exact, but points body-X outside the independent 3 degree gate.
    fake.q_wxyz = np.asarray(euler_to_quat_wxyz(
        0.0,
        math.radians(4.0),
        math.pi / 2.0,
    ))
    status = MPCTrackTrajectoryAcados._prehook_alignment_status(fake)
    assert status['orientation_ok'] is True
    assert status['yaw_ok'] is True
    assert status['forward_axis_error_rad'] == pytest.approx(
        math.radians(4.0)
    )
    assert status['forward_axis_ok'] is False
    assert status['attitude_ok'] is False
    assert status['complete'] is False
    assert not MPCTrackTrajectoryAcados._trajectory_completion_reached(fake)


def test_independent_prehook_tolerance_does_not_weaken_hook_or_wait_gate():
    goal = np.asarray(euler_to_quat_wxyz(0.0, 0.0, -2.9))
    current = np.asarray(
        euler_to_quat_wxyz(0.0, 0.0, -2.9 + math.radians(7.0))
    )
    parameters = {
        'goal_reached_tol_m': 0.05,
        'goal_reached_orientation_tol_rad': math.radians(5.0),
        'prehook_reached_orientation_tol_rad': math.radians(8.0),
        'prehook_reached_yaw_tol_rad': math.radians(3.0),
        'prehook_reached_forward_axis_tol_rad': 0.0,
        'hold_attitude': True,
        'use_box_recovery_mission': False,
    }
    fake = SimpleNamespace(
        mission_state='TRACK_TO_PREHOOK',
        p_w=np.array([4.9, 0.1, 1.7]),
        traj_goal_pos=np.array([4.9, 0.1, 1.7]),
        q_wxyz=current,
        fixed_hook_depth_tolerance_m=0.03,
        _dynamic_prehook_planner_enabled=lambda: True,
        _pre_approach_waypoint_enabled=lambda: True,
        _goal_quaternion=lambda: goal.copy(),
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
    )
    fake._prehook_orientation_tolerance_rad = lambda: (
        MPCTrackTrajectoryAcados._prehook_orientation_tolerance_rad(fake)
    )
    fake._prehook_yaw_tolerance_rad = lambda: (
        MPCTrackTrajectoryAcados._prehook_yaw_tolerance_rad(fake)
    )
    fake._prehook_forward_axis_tolerance_rad = lambda: (
        MPCTrackTrajectoryAcados._prehook_forward_axis_tolerance_rad(fake)
    )

    assert not MPCTrackTrajectoryAcados._trajectory_completion_reached(fake)

    # A custom wider full-attitude gate still cannot bypass the independent
    # 3 degree yaw gate during pre-hook.
    fake.q_wxyz = np.asarray(
        euler_to_quat_wxyz(0.0, 0.0, -2.9 + math.radians(2.0))
    )
    assert MPCTrackTrajectoryAcados._trajectory_completion_reached(fake)

    fake.q_wxyz = current
    for hook_state in ('GO_FORWARD', 'WAIT_HOOK'):
        fake.mission_state = hook_state
        assert not MPCTrackTrajectoryAcados._trajectory_completion_reached(
            fake
        )

    fake.q_wxyz = np.asarray(
        euler_to_quat_wxyz(0.0, 0.0, -2.9 + math.radians(4.0))
    )
    assert MPCTrackTrajectoryAcados._trajectory_completion_reached(fake)


def _optional_suppression_fake(path_safe):
    status_calls = []
    requests = []
    holds = []
    parameters = {
        'prehook_planner_check_rate_hz': 2.0,
        'prehook_replan_deviation_m': 0.30,
    }
    fake = SimpleNamespace(
        mission_state='TRACK_TO_PREHOOK',
        _prehook_last_check_monotonic=None,
        _prehook_deviation_since_monotonic=None,
        _prehook_mandatory_replan_pending=False,
        _prehook_replan_future=None,
        _prehook_replan_context=None,
        _prehook_path_status=lambda: (
            status_calls.append(True)
            or {
                'safe': path_safe,
                'cross_track_m': 0.0,
                'remaining_cost_m': 0.0,
            }
        ),
        _activate_prehook_planner_hold=lambda: holds.append(True),
        _submit_prehook_plan_request=lambda **kwargs: (
            requests.append(kwargs) or True
        ),
        _revoke_mission_enable=lambda reason: pytest.fail(reason),
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        get_logger=lambda: _Logger(),
    )
    return fake, status_calls, requests, holds


def test_alignment_wait_suppresses_optional_but_checks_path_safety():
    fake, status_calls, requests, holds = _optional_suppression_fake(True)

    MPCTrackTrajectoryAcados._submit_prehook_replan_if_needed(
        fake,
        suppress_optional_improvement=True,
    )

    assert status_calls == [True]
    assert requests == []
    assert holds == []


def test_alignment_wait_preserves_mandatory_unsafe_path_repair():
    fake, status_calls, requests, holds = _optional_suppression_fake(False)

    MPCTrackTrajectoryAcados._submit_prehook_replan_if_needed(
        fake,
        suppress_optional_improvement=True,
    )

    assert status_calls == [True]
    assert holds == [True]
    assert len(requests) == 1
    assert requests[0]['reason'] == 'unsafe_remaining_path'
    assert requests[0]['request_level'] == 'mandatory'


def test_completed_optional_candidate_is_discarded_during_alignment_wait():
    future = Future()
    future.set_result(([(4.7, 0.1), (4.8, 0.1)], None))
    installed = []
    logger = _Logger()
    fake = SimpleNamespace(
        mission_state='TRACK_TO_PREHOOK',
        _prehook_plan_generation=3,
        _prehook_replan_future=future,
        _prehook_replan_context={
            'generation': 3,
            'goal_signature': ('prehook',),
            'reason': 'candidate_path_improvement_check',
            'request_level': 'optional',
        },
        _prehook_mandatory_replan_pending=False,
        _prehook_planner_hold_active=False,
        _goal_signature=lambda: ('prehook',),
        _set_path_trajectory=lambda *args, **kwargs: installed.append(
            (args, kwargs)
        ),
        get_logger=lambda: logger,
    )

    MPCTrackTrajectoryAcados._poll_prehook_replan(
        fake,
        suppress_optional_improvement=True,
    )

    assert installed == []
    assert fake._prehook_replan_future is None
    assert fake._prehook_replan_context is None
    assert any(
        'discarded' in message
        for level, message in logger.messages
        if level == 'info'
    )


def test_latched_attitude_fault_holds_without_replanning_or_transition():
    fake = SimpleNamespace(
        mission_state='TRACK_TO_PREHOOK',
        _prehook_attitude_fault_latched=True,
        _dynamic_prehook_planner_enabled=lambda: True,
        _prehook_alignment_status=lambda: pytest.fail(
            'a latched fault must keep the captured reference'
        ),
        _poll_prehook_replan=lambda **_kwargs: pytest.fail(
            'a latched fault must not install another path'
        ),
    )

    assert MPCTrackTrajectoryAcados._update_dynamic_prehook_phase(fake)
    assert fake.mission_state == 'TRACK_TO_PREHOOK'


def test_latched_attitude_fault_keeps_heartbeat_through_cascaded_odom_latch():
    fake = SimpleNamespace(
        ocp_solver=object(),
        mission_rearm_required=True,
        mission_enable=False,
        enabled=True,
        _prehook_attitude_fault_latched=True,
        _control_mode_feedback_fresh=lambda: True,
        _odom_fresh=lambda: pytest.fail(
            'latched containment must not require bad odometry'
        ),
        _command_fresh=lambda: pytest.fail(
            'latched containment must not require an MPC command'
        ),
    )

    assert MPCTrackTrajectoryAcados._controller_heartbeat_ready(fake)


@pytest.mark.parametrize(
    ('solver_present', 'feedback_fresh', 'enabled'),
    [
        (False, True, True),
        (True, False, True),
        (True, True, False),
    ],
)
def test_latched_attitude_fault_heartbeat_still_requires_live_control_chain(
    solver_present,
    feedback_fresh,
    enabled,
):
    fake = SimpleNamespace(
        ocp_solver=object() if solver_present else None,
        mission_rearm_required=True,
        mission_enable=False,
        enabled=enabled,
        _prehook_attitude_fault_latched=True,
        _control_mode_feedback_fresh=lambda: feedback_fresh,
    )

    assert not MPCTrackTrajectoryAcados._controller_heartbeat_ready(fake)


def test_regular_odom_latch_still_withdraws_controller_heartbeat():
    fake = SimpleNamespace(
        ocp_solver=object(),
        mission_rearm_required=True,
        mission_enable=False,
        enabled=True,
        _prehook_attitude_fault_latched=False,
        _control_mode_feedback_fresh=lambda: True,
        _odom_fresh=lambda: pytest.fail(
            'generic rearm latch must fail before odometry is consulted'
        ),
    )

    assert not MPCTrackTrajectoryAcados._controller_heartbeat_ready(fake)


def test_latched_attitude_fault_cascade_publishes_heartbeat_and_zero_wrench():
    heartbeat_messages = []
    wrench_messages = []
    fake = SimpleNamespace(
        ocp_solver=object(),
        mission_rearm_required=True,
        mission_enable=False,
        enabled=True,
        _prehook_attitude_fault_latched=True,
        _control_mode_feedback_fresh=lambda: True,
        _control_gate_active=lambda: True,
        _mission_allowed=lambda: False,
        pub_controller_heartbeat=SimpleNamespace(
            publish=lambda message: heartbeat_messages.append(message)
        ),
        u_force_cmd_N=np.array([3.0, -2.0, 1.0], dtype=float),
        u_tau_cmd_Nm=np.array([0.3, -0.2, 0.1], dtype=float),
        command_valid=True,
        last_solution_sec=100.0,
        traj_active=True,
        last_goal_signature=('unsafe',),
        terminal_hold_goal_signature=('unsafe',),
        _trajectory_reset_pending=False,
        fixed_hook_projected_restart=True,
        position_integral_error_world=np.ones(3, dtype=float),
        last_position_integral_update_sec=100.0,
        publish_zero=lambda: wrench_messages.append(
            (
                fake.u_force_cmd_N.copy(),
                fake.u_tau_cmd_Nm.copy(),
            )
        ),
    )
    fake._controller_heartbeat_ready = lambda: (
        MPCTrackTrajectoryAcados._controller_heartbeat_ready(fake)
    )
    fake._zero_command_cache = lambda: (
        MPCTrackTrajectoryAcados._zero_command_cache(fake)
    )
    fake._reset_position_integral = lambda: (
        MPCTrackTrajectoryAcados._reset_position_integral(fake)
    )
    fake._clear_position_integral_xy_preserve_z = lambda: (
        MPCTrackTrajectoryAcados._clear_position_integral_xy_preserve_z(
            fake
        )
    )
    fake._invalidate_command = lambda **kwargs: (
        MPCTrackTrajectoryAcados._invalidate_command(fake, **kwargs)
    )

    MPCTrackTrajectoryAcados.publish_tick(fake)

    assert len(heartbeat_messages) == 1
    assert len(wrench_messages) == 1
    np.testing.assert_allclose(wrench_messages[0][0], np.zeros(3))
    np.testing.assert_allclose(wrench_messages[0][1], np.zeros(3))
    np.testing.assert_allclose(fake.u_force_cmd_N, np.zeros(3))
    np.testing.assert_allclose(fake.u_tau_cmd_Nm, np.zeros(3))
    assert fake.command_valid is False
    assert fake.last_solution_sec is None


def test_prehook_reached_requires_full_dwell_before_go_forward():
    now = [20.0]
    rebuilt_states = []
    hook = np.array([4.3, -0.05, 1.7], dtype=float)
    fake = SimpleNamespace(
        mission_state='PREHOOK_REACHED',
        state_enter_time_sec=20.0,
        active_goal_pos=np.array([4.8, 0.1, 1.7], dtype=float),
        active_goal_yaw=-0.4,
        terminal_hold_goal_signature=('prehook',),
        last_goal_signature=('prehook',),
        _prehook_reached_since_sec=20.0,
        _prehook_planner_hold_active=True,
        _dynamic_prehook_planner_enabled=lambda: True,
        _trajectory_completion_reached=lambda: True,
        _now_sec=lambda: now[0],
        _goal_position_static=lambda: hook.copy(),
        _goal_yaw_static=lambda: -2.9,
        _reset_trajectory_from_current_pose=lambda: (
            rebuilt_states.append(fake.mission_state)
        ),
        get_parameter=lambda name: SimpleNamespace(
            value={'prehook_reached_hold_s': 1.0}[name]
        ),
        get_logger=lambda: _Logger(),
    )

    now[0] = 20.99
    assert MPCTrackTrajectoryAcados._update_dynamic_prehook_phase(fake)
    assert fake.mission_state == 'PREHOOK_REACHED'
    assert rebuilt_states == []

    now[0] = 21.0
    assert MPCTrackTrajectoryAcados._update_dynamic_prehook_phase(fake)
    assert fake.mission_state == 'GO_FORWARD'
    np.testing.assert_allclose(fake.active_goal_pos, hook)
    assert fake.active_goal_yaw == pytest.approx(-2.9)
    assert fake.terminal_hold_goal_signature is None
    assert fake.last_goal_signature is None
    assert fake._prehook_planner_hold_active is False
    assert rebuilt_states == ['GO_FORWARD']


@pytest.mark.parametrize('mission_state', [
    'GO_FORWARD',
    'WAIT_HOOK',
    'GO_BACK',
])
def test_hook_sequence_states_never_enable_astar(mission_state):
    fake = SimpleNamespace(
        mission_state=mission_state,
        _dynamic_prehook_planner_enabled=lambda: True,
        get_parameter=lambda name: SimpleNamespace(
            value={
                'planner_mode': 'astar',
                'planner_use_for_align': True,
                'planner_use_for_return': True,
            }[name]
        ),
    )

    assert not MPCTrackTrajectoryAcados._planner_enabled_for_current_phase(
        fake
    )


def _manual_hook_confirmation_fake(*, mission_state='WAIT_HOOK'):
    """Build the smallest real fixed-hook operator-confirmation harness."""
    logger = _Logger()
    now = [20.0]
    flags = {
        'mission_allowed': True,
        'control_gate': True,
        'control_fresh': True,
        'odom_fresh': True,
        'state_valid': True,
        'command_fresh': True,
        'at_hook_pose': True,
    }
    resets = []
    mission_updates = []
    parameters = {
        'require_operator_hook_confirmation': True,
        'hook_confirmation_service': (
            '/bluerov2/fixed_hook/confirm_hook'
        ),
        'use_box_recovery_mission': False,
        'return_to_pre_approach_after_hold': True,
        # This deliberately remains five seconds. In manual mode it must
        # never advance WAIT_HOOK, regardless of elapsed time.
        'final_pose_hold_s': 5.0,
        'require_mission_enable': True,
    }
    fake = SimpleNamespace(
        mission_state=mission_state,
        state_enter_time_sec=10.0,
        terminal_hold_goal_signature=('hook',),
        last_goal_signature=('hook',),
        require_operator_hook_confirmation=True,
        _operator_hook_confirmation_pending=False,
        _wait_hook_enter_monotonic=time.monotonic() - 1.0,
        hook_confirmation_min_wait_s=0.25,
        mission_enable=True,
        mission_rearm_required=False,
        enabled=True,
        ocp_solver=object(),
        get_parameter=lambda name: SimpleNamespace(
            value=parameters[name]
        ),
        get_logger=lambda: logger,
        _now_sec=lambda: now[0],
        _mission_allowed=lambda: flags['mission_allowed'],
        _control_gate_active=lambda: (
            flags['control_gate'] and flags['control_fresh']
        ),
        _control_mode_feedback_fresh=lambda: flags['control_fresh'],
        _odom_fresh=lambda: flags['odom_fresh'],
        _state_valid=lambda _state: flags['state_valid'],
        _x_meas=lambda: np.zeros(13, dtype=float),
        _command_fresh=lambda: flags['command_fresh'],
        _trajectory_completion_reached=lambda: flags['at_hook_pose'],
        _pre_approach_waypoint_enabled=lambda: True,
        _update_mission=lambda: mission_updates.append(
            fake.mission_state
        ),
        _reset_trajectory_from_current_pose=lambda: resets.append(
            fake.mission_state
        ),
        _enter_terminal_hold=lambda _signature: None,
    )
    return fake, now, flags, resets, mission_updates, logger


def _request_hook_confirmation(fake):
    response = Trigger.Response()
    returned = MPCTrackTrajectoryAcados.on_hook_confirmation(
        fake,
        Trigger.Request(),
        response,
    )
    assert returned is response
    return response


def test_manual_wait_hook_never_times_out_without_operator_h():
    fake, now, _flags, resets, _updates, _logger = (
        _manual_hook_confirmation_fake()
    )

    for timestamp in (20.0, 25.0, 120.0, 10020.0):
        now[0] = timestamp
        assert MPCTrackTrajectoryAcados._update_fixed_hook_hold_phase(
            fake
        )
        assert fake.mission_state == 'WAIT_HOOK'
        assert fake._operator_hook_confirmation_pending is False

    assert resets == []


def test_enter_wait_hook_starts_a_fresh_monotonic_confirmation_epoch(
    monkeypatch,
):
    fake, _now, _flags, resets, _updates, _logger = (
        _manual_hook_confirmation_fake(mission_state='GO_FORWARD')
    )
    fake._operator_hook_confirmation_pending = True
    fake._wait_hook_enter_monotonic = 1.0
    monkeypatch.setattr(
        'bluerov2_control.mpc_track_trajectory_acados.time.monotonic',
        lambda: 50.0,
    )

    assert MPCTrackTrajectoryAcados._begin_fixed_hook_final_hold(
        fake,
        ('hook',),
    )
    assert fake.mission_state == 'WAIT_HOOK'
    assert fake._operator_hook_confirmation_pending is False
    assert fake._wait_hook_enter_monotonic == pytest.approx(50.0)
    assert resets == []


def test_valid_h_is_consumed_on_next_tick_and_starts_go_back_once():
    fake, _now, _flags, resets, mission_updates, _logger = (
        _manual_hook_confirmation_fake()
    )

    response = _request_hook_confirmation(fake)
    assert response.success is True
    assert fake.mission_state == 'WAIT_HOOK'
    assert fake._operator_hook_confirmation_pending is True
    assert resets == []

    assert MPCTrackTrajectoryAcados._update_fixed_hook_hold_phase(fake)
    assert fake.mission_state == 'GO_BACK'
    assert fake._operator_hook_confirmation_pending is False
    assert fake._wait_hook_enter_monotonic is None
    assert fake.terminal_hold_goal_signature is None
    assert fake.last_goal_signature is None
    assert mission_updates == ['GO_BACK']
    assert resets == ['GO_BACK']

    duplicate = _request_hook_confirmation(fake)
    assert duplicate.success is False
    assert fake._operator_hook_confirmation_pending is False
    assert not MPCTrackTrajectoryAcados._update_fixed_hook_hold_phase(fake)
    assert resets == ['GO_BACK']


@pytest.mark.parametrize(
    'mission_state',
    [
        'INIT',
        'PLAN_TO_PREHOOK',
        'TRACK_TO_PREHOOK',
        'PREHOOK_REACHED',
        'GO_FORWARD',
        'GO_BACK',
        'COMPLETE',
    ],
)
def test_early_h_is_rejected_and_never_cached(mission_state):
    fake, _now, _flags, resets, _updates, _logger = (
        _manual_hook_confirmation_fake(mission_state=mission_state)
    )

    response = _request_hook_confirmation(fake)

    assert response.success is False
    assert fake.mission_state == mission_state
    assert fake._operator_hook_confirmation_pending is False
    assert resets == []


def test_duplicate_h_while_pending_is_idempotently_rejected():
    fake, _now, _flags, resets, _updates, _logger = (
        _manual_hook_confirmation_fake()
    )

    first = _request_hook_confirmation(fake)
    second = _request_hook_confirmation(fake)

    assert first.success is True
    assert second.success is False
    assert fake._operator_hook_confirmation_pending is True
    assert fake.mission_state == 'WAIT_HOOK'
    assert resets == []


def test_h_queued_at_wait_entry_is_rejected_until_debounce(monkeypatch):
    fake, _now, _flags, resets, _updates, _logger = (
        _manual_hook_confirmation_fake()
    )
    monotonic_now = [100.0]
    fake._wait_hook_enter_monotonic = monotonic_now[0]
    monkeypatch.setattr(
        'bluerov2_control.mpc_track_trajectory_acados.time.monotonic',
        lambda: monotonic_now[0],
    )

    same_tick = _request_hook_confirmation(fake)
    assert same_tick.success is False
    assert fake._operator_hook_confirmation_pending is False

    monotonic_now[0] += 0.249
    still_early = _request_hook_confirmation(fake)
    assert still_early.success is False
    assert fake._operator_hook_confirmation_pending is False

    monotonic_now[0] += 0.001
    fresh_press = _request_hook_confirmation(fake)
    assert fresh_press.success is True
    assert fake._operator_hook_confirmation_pending is True
    assert resets == []


@pytest.mark.parametrize(
    'blocked_gate',
    [
        'mission_allowed',
        'control_gate',
        'control_fresh',
        'odom_fresh',
        'solver_ready',
        'state_valid',
        'command_fresh',
    ],
)
def test_h_requires_all_mission_and_feedback_freshness_gates(
    blocked_gate,
):
    fake, _now, flags, resets, _updates, _logger = (
        _manual_hook_confirmation_fake()
    )
    if blocked_gate == 'solver_ready':
        fake.ocp_solver = None
    else:
        flags[blocked_gate] = False
    if blocked_gate in ('control_gate', 'control_fresh'):
        # Cover implementations that use either the combined control gate or
        # its raw armed/Offboard feedback component.
        fake.enabled = False

    response = _request_hook_confirmation(fake)

    assert response.success is False
    assert fake.mission_state == 'WAIT_HOOK'
    assert fake._operator_hook_confirmation_pending is False
    assert resets == []


def test_h_accepts_pose_drift_after_wait_and_starts_go_back():
    fake, _now, flags, resets, mission_updates, _logger = (
        _manual_hook_confirmation_fake()
    )
    # Entry into WAIT_HOOK proves that the strict Hook completion gate was
    # already satisfied.  Subsequent contact motion or tether disturbance may
    # move the measured pose outside that gate; the operator's visual H is now
    # authoritative for engagement and must not force another GO_FORWARD.
    flags['at_hook_pose'] = False

    response = _request_hook_confirmation(fake)
    assert response.success is True
    assert fake.mission_state == 'WAIT_HOOK'
    assert fake._operator_hook_confirmation_pending is True

    assert MPCTrackTrajectoryAcados._update_fixed_hook_hold_phase(fake)

    assert fake.mission_state == 'GO_BACK'
    assert fake._operator_hook_confirmation_pending is False
    assert fake._wait_hook_enter_monotonic is None
    assert mission_updates == ['GO_BACK']
    assert resets == ['GO_BACK']


def test_dynamic_prehook_reset_discards_pending_hook_confirmation():
    fake = SimpleNamespace(
        _operator_hook_confirmation_pending=True,
        _wait_hook_enter_monotonic=1.0,
        _prehook_attitude_fault_latched=False,
        _prehook_plan_generation=0,
        _prehook_replan_future=None,
        _prehook_replan_context=None,
    )
    fake._retire_prehook_plan_request = lambda: (
        MPCTrackTrajectoryAcados._retire_prehook_plan_request(fake)
    )

    MPCTrackTrajectoryAcados._reset_dynamic_prehook_runtime(fake)

    assert fake._operator_hook_confirmation_pending is False
    assert fake._wait_hook_enter_monotonic is None


def test_path_trajectory_has_continuous_depth_and_attitude_duration():
    logger = _Logger()
    start = np.array([0.0, 0.0, 0.4], dtype=float)
    goal_z = 1.4
    goal_quaternion = np.asarray(
        euler_to_quat_wxyz(0.0, 0.0, math.pi / 2.0),
        dtype=float,
    )
    angular_speed = math.pi / 4.0
    parameters = {
        'min_traj_duration_s': 0.10,
        'Ts': 0.04,
        'traj_angular_speed_rad_s': angular_speed,
    }
    fake = SimpleNamespace(
        p_w=start.copy(),
        q_wxyz=(1.0, 0.0, 0.0, 0.0),
        terminal_hold_goal_signature=('old',),
        last_goal_signature=None,
        _now_sec=lambda: 100.0,
        _reset_position_integral=lambda: None,
        _phase_traj_speed=lambda: 10.0,
        _goal_quaternion=lambda: goal_quaternion.copy(),
        _goal_signature=lambda: ('prehook-goal',),
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        get_logger=lambda: logger,
    )

    MPCTrackTrajectoryAcados._set_path_trajectory(
        fake,
        [(1.0, 0.0), (2.0, 0.0)],
        goal_z,
        start_z=start[2],
    )

    np.testing.assert_allclose(fake.path_points[0], start)
    np.testing.assert_allclose(fake.path_points[1], [1.0, 0.0, 0.9])
    np.testing.assert_allclose(fake.path_points[-1], [2.0, 0.0, goal_z])
    sampled_midpoint = MPCTrackTrajectoryAcados._sample_path_pref(fake, 0.5)
    np.testing.assert_allclose(sampled_midpoint, [1.0, 0.0, 0.9])

    expected_angular_duration = quat_angular_distance_wxyz(
        fake.q_wxyz,
        goal_quaternion,
    ) / angular_speed
    assert expected_angular_duration == pytest.approx(2.0)
    assert fake.path_total_length / 10.0 < expected_angular_duration
    assert fake.traj_duration_sec == pytest.approx(
        expected_angular_duration
    )
    assert fake.traj_kind == 'path'
    assert fake.traj_active is True
