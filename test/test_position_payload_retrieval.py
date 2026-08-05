"""Unit tests for the PX4-attitude-backed position mission."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from px4_msgs.msg import VehicleOdometry

from bluerov2_control.position_payload_retrieval import (
    PositionPayloadRetrieval,
    level_yaw_quat_wxyz,
    move_towards_vector,
    move_towards_yaw,
    position_pd_thrust_body,
    quat_to_rotation_wxyz,
    retrieval_targets,
    velocity_in_navigation_frame,
    wrap_pi,
)


def _mission_targets(raise_m):
    return retrieval_targets(
        box_xyz_sdf=(-1.5, -1.5, -96.5),
        hook_mount_xyz_sdf=(0.42, 0.04, -0.08),
        hook_tip_extra_x=0.10,
        handle_offset_ned=(0.1, -0.10, -0.025),
        approach_yaw=-1.57,
        approach_clearance=0.1,
        pass_overshoot=0.1,
        backward_extra=0.08,
        alignment_raise=raise_m,
    )


def test_hook_remains_fixed_and_vehicle_targets_are_raised_in_ned():
    baseline = _mission_targets(0.0)
    raised = _mission_targets(0.04)
    for baseline_target, raised_target in zip(baseline, raised):
        np.testing.assert_allclose(
            raised_target[:2],
            baseline_target[:2],
        )
        assert raised_target[2] == pytest.approx(
            baseline_target[2] - 0.04
        )


def test_mission_aligns_horizontally_before_descending():
    align, forward, backward = _mission_targets(0.04)
    parameters = {
        'approach_yaw': -1.57,
        'align_hold_lower_m': 0.01,
        'return_to_start': True,
    }
    fake = SimpleNamespace(
        mission_state='ALIGN_XY',
        align_target=align,
        forward_target=forward,
        backward_target=backward,
        home_position=np.array([-0.182, 0.157, 94.471]),
        home_yaw=1.461,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
    )
    horizontal, _ = PositionPayloadRetrieval._stage_target(fake)
    assert horizontal[2] == pytest.approx(fake.home_position[2])
    np.testing.assert_allclose(horizontal[:2], align[:2])

    fake.mission_state = 'DESCEND'
    descent, _ = PositionPayloadRetrieval._stage_target(fake)
    np.testing.assert_allclose(descent, align)
    baseline_z = _mission_targets(0.0)[0][2]
    assert descent[2] < baseline_z
    assert descent[2] == pytest.approx(baseline_z - 0.04)

    fake.mission_state = 'ALIGN_HOLD'
    hold, _ = PositionPayloadRetrieval._stage_target(fake)
    np.testing.assert_allclose(hold[:2], align[:2])
    assert hold[2] == pytest.approx(align[2] + 0.01)

    fake.mission_state = 'FORWARD_PASS'
    forward_pass, _ = PositionPayloadRetrieval._stage_target(fake)
    assert forward_pass[2] == pytest.approx(forward[2] + 0.01)


def test_yaw_reference_uses_shortest_bounded_step():
    start = math.radians(179.0)
    goal = math.radians(-179.0)
    updated = move_towards_yaw(start, goal, max_rate=0.1, dt=0.2)
    assert abs(wrap_pi(updated - start)) == pytest.approx(0.02)
    assert abs(wrap_pi(goal - updated)) < abs(wrap_pi(goal - start))


def test_position_reference_moves_immediately_while_yaw_is_slewing():
    current = np.array([-0.182, 0.157, 94.471])
    target = np.array([-1.44, -0.98, 94.471])
    updated = move_towards_vector(current, target, 0.06, 0.02)
    assert np.linalg.norm(updated - current) == pytest.approx(0.0012)


def test_level_yaw_attitude_and_translation_are_decoupled():
    yaw = math.pi / 2.0
    quaternion = level_yaw_quat_wxyz(yaw)
    rotation = quat_to_rotation_wxyz(quaternion)
    np.testing.assert_allclose(rotation[:, 2], [0.0, 0.0, 1.0], atol=1e-12)

    thrust_body = position_pd_thrust_body(
        position=np.zeros(3),
        velocity_nav=np.zeros(3),
        quaternion=quaternion,
        reference_position=np.array([0.05, 0.0, 0.0]),
        kp=np.ones(3),
        kd=np.full(3, 0.2),
        axis_limits=np.full(3, 0.08),
    )
    # A northward world command at +90 deg yaw becomes body-left thrust.  It
    # remains nonzero while PX4 independently tracks the yaw quaternion.
    np.testing.assert_allclose(thrust_body, [0.0, -0.05, 0.0], atol=1e-12)


def test_body_velocity_is_rotated_to_navigation_frame():
    message = SimpleNamespace(
        velocity=[1.0, 0.0, 0.0],
        velocity_frame=VehicleOdometry.VELOCITY_FRAME_BODY_FRD,
        pose_frame=VehicleOdometry.POSE_FRAME_NED,
    )
    result = velocity_in_navigation_frame(
        message,
        level_yaw_quat_wxyz(math.pi / 2.0),
    )
    np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-12)
