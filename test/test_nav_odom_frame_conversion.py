"""Unit tests for explicit MoCap-to-FRD frame conversion helpers."""

from types import SimpleNamespace

import numpy as np

from bluerov2_control.nav_odom_to_vehicle_odometry import (
    NavOdomToVehicleOdometry,
    _axis_transform,
    _matrix_to_quat_wxyz,
    _quat_xyzw_to_matrix,
    _validated_body_frd_angular_velocity,
)


def test_nwu_flu_level_pose_becomes_frd_level_pose():
    world = _axis_transform("nwu")
    body = _axis_transform("flu", body=True)
    rotation_out = world @ np.eye(3) @ body.T

    np.testing.assert_allclose(rotation_out, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(
        _matrix_to_quat_wxyz(rotation_out),
        [1.0, 0.0, 0.0, 0.0],
        atol=1e-12,
    )


def test_enu_position_maps_to_ned_axes():
    world = _axis_transform("enu")
    position_enu = np.array([2.0, 3.0, 1.5])

    np.testing.assert_allclose(
        world @ position_enu,
        [3.0, 2.0, -1.5],
        atol=1e-12,
    )


def test_flu_velocity_maps_to_frd_axes():
    body = _axis_transform("flu", body=True)

    np.testing.assert_allclose(
        body @ np.array([1.0, 2.0, 3.0]),
        [1.0, -2.0, -3.0],
        atol=1e-12,
    )


def test_quaternion_matrix_round_trip():
    quaternion_xyzw = np.array([0.1, -0.2, 0.3, 0.9], dtype=float)
    quaternion_xyzw /= np.linalg.norm(quaternion_xyzw)
    rotation = _quat_xyzw_to_matrix(quaternion_xyzw)
    quaternion_wxyz = _matrix_to_quat_wxyz(rotation)

    expected_wxyz = quaternion_xyzw[[3, 0, 1, 2]]
    if np.dot(quaternion_wxyz, expected_wxyz) < 0.0:
        expected_wxyz = -expected_wxyz
    np.testing.assert_allclose(quaternion_wxyz, expected_wxyz, atol=1e-12)


def test_invalid_quaternion_is_not_silently_replaced_with_identity():
    assert _quat_xyzw_to_matrix([0.0, 0.0, 0.0, 0.0]) is None
    assert _quat_xyzw_to_matrix([np.nan, 0.0, 0.0, 1.0]) is None


def test_external_body_frd_rate_validation_is_fail_closed():
    np.testing.assert_allclose(
        _validated_body_frd_angular_velocity([0.1, -0.2, 0.3], 5.0),
        [0.1, -0.2, 0.3],
    )
    assert _validated_body_frd_angular_velocity(
        [np.nan, 0.0, 0.0], 5.0
    ) is None
    assert _validated_body_frd_angular_velocity(
        [4.0, 4.0, 0.0], 5.0
    ) is None


def test_required_external_body_rate_expires_fail_closed():
    now = [10.05]
    fake = SimpleNamespace(
        _angular_velocity_override_topic='/glub/fmu/out/vehicle_odometry',
        _angular_velocity_override=np.array([0.1, -0.2, 0.3]),
        _angular_velocity_override_rx_sec=10.0,
        _angular_velocity_override_timeout_sec=0.10,
        _now_sec=lambda: now[0],
    )

    np.testing.assert_allclose(
        NavOdomToVehicleOdometry._fresh_angular_velocity_override(fake),
        [0.1, -0.2, 0.3],
    )
    now[0] = 10.101
    assert (
        NavOdomToVehicleOdometry._fresh_angular_velocity_override(fake)
        is None
    )
