"""Fail-closed publication and tracking-recovery tests for MoCap EKF."""

from types import MethodType, SimpleNamespace

import numpy as np

from bluerov2_control.mocap_ekf_odom import (
    MocapEkfOdom,
    QuaternionCvEkf,
    quat_exp,
)


class _Logger:
    def __init__(self):
        self.warnings = []

    def warn(self, message, **_kwargs):
        self.warnings.append(message)


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _TfBroadcaster:
    def __init__(self):
        self.transforms = []

    def sendTransform(self, transform):
        self.transforms.append(transform)


class _Now:
    def to_msg(self):
        return 'stamp'


def _tick_fixture(predicted_orientation):
    ekf = QuaternionCvEkf(max_base_link_z_axis_angle_rad=0.4)
    ekf.initialize(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
    publisher = _Publisher()
    tf_broadcaster = _TfBroadcaster()
    logger = _Logger()
    node = SimpleNamespace(
        _filter=ekf,
        _last_filter_sec=9.0,
        _last_pose_rx_sec=9.0,
        _last_pose_accepted_sec=9.0,
        _consecutive_full_rejections=0,
        _imu_gyro_lpf=np.zeros(3),
        _last_trusted_filter_snapshot=None,
        _tracking_loss_latched=False,
        _tracking_loss_started_sec=None,
        _max_coast_sec=2.0,
        _max_rejected_samples=0,
        _odom_pub=publisher,
        _tf_broadcaster=tf_broadcaster,
        _now_sec=lambda: 10.0,
        get_clock=lambda: SimpleNamespace(now=lambda: _Now()),
        get_logger=lambda: logger,
        _odom_message=lambda stamp: ('odom', stamp),
        _transform_message=lambda stamp: ('tf', stamp),
    )

    def predict_to(_self, now):
        assert now == 10.0
        _self._filter.orientation = predicted_orientation.copy()
        _self._last_filter_sec = now

    node._predict_to = MethodType(predict_to, node)
    node._reset_filter = MethodType(MocapEkfOdom._reset_filter, node)
    node._tracking_loss_due = MethodType(
        MocapEkfOdom._tracking_loss_due,
        node,
    )
    node._enter_tracking_loss_latch = MethodType(
        MocapEkfOdom._enter_tracking_loss_latch,
        node,
    )
    node._last_trusted_filter_snapshot = ekf._snapshot()
    return node, ekf, publisher, tf_broadcaster, logger


def _tracking_loss_fixture():
    anchor_position = np.array([4.64, -0.07, 1.46], dtype=float)
    anchor_orientation = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    ekf = QuaternionCvEkf(
        max_position_innovation_m=0.20,
        max_orientation_innovation_rad=0.55,
        max_base_link_z_axis_angle_rad=0.55,
    )
    ekf.initialize(anchor_position, anchor_orientation)
    publisher = _Publisher()
    logger = _Logger()
    clock_sec = [10.0]
    node = SimpleNamespace(
        _filter=ekf,
        _last_filter_sec=9.0,
        _last_pose_rx_sec=9.0,
        _last_pose_accepted_sec=7.9,
        _consecutive_full_rejections=0,
        _imu_gyro_lpf=np.zeros(3),
        _last_trusted_filter_snapshot=ekf._snapshot(),
        _tracking_loss_latched=False,
        _tracking_loss_started_sec=None,
        _max_coast_sec=2.0,
        _max_rejected_samples=1,
        _odom_pub=publisher,
        _tf_broadcaster=None,
        _now_sec=lambda: clock_sec[0],
        get_clock=lambda: SimpleNamespace(now=lambda: _Now()),
        get_logger=lambda: logger,
        _odom_message=lambda stamp: ('odom', stamp),
        _transform_message=lambda stamp: ('tf', stamp),
    )
    for method in (
        '_predict_to',
        '_reset_filter',
        '_record_trusted_filter_anchor',
        '_tracking_loss_due',
        '_enter_tracking_loss_latch',
        '_update_filter_from_pose',
        '_handle_update_result',
    ):
        setattr(
            node,
            method,
            MethodType(getattr(MocapEkfOdom, method), node),
        )
    return (
        node,
        ekf,
        publisher,
        logger,
        clock_sec,
        anchor_position,
        anchor_orientation,
    )


def test_tick_latches_and_does_not_publish_unsafe_predicted_orientation():
    node, old_filter, publisher, tf_broadcaster, logger = _tick_fixture(
        quat_exp(np.array([0.5, 0.0, 0.0]))
    )

    MocapEkfOdom._tick(node)

    assert node._filter is old_filter
    assert node._filter.initialized is True
    assert node._tracking_loss_latched is True
    assert node._last_filter_sec is None
    assert node._last_pose_rx_sec == 9.0
    assert node._last_pose_accepted_sec == 9.0
    np.testing.assert_allclose(
        node._filter.orientation,
        np.array([0.0, 0.0, 0.0, 1.0]),
    )
    assert publisher.messages == []
    assert tf_broadcaster.transforms == []
    assert len(logger.warnings) == 1
    assert 'predicted EKF orientation is unsafe' in logger.warnings[0]
    assert '0.500rad > 0.400rad' in logger.warnings[0]
    assert 'latched off' in logger.warnings[0]


def test_tick_publishes_safe_predicted_orientation_without_reset():
    safe_orientation = quat_exp(np.array([0.2, 0.0, 0.0]))
    node, old_filter, publisher, tf_broadcaster, logger = _tick_fixture(
        safe_orientation
    )

    MocapEkfOdom._tick(node)

    assert node._filter is old_filter
    assert node._filter.initialized is True
    assert publisher.messages == [('odom', 'stamp')]
    assert tf_broadcaster.transforms == [('tf', 'stamp')]
    assert logger.warnings == []


def test_tick_keeps_last_safe_orientation_when_real_predict_is_unsafe():
    node, old_filter, publisher, tf_broadcaster, logger = _tick_fixture(
        np.array([0.0, 0.0, 0.0, 1.0])
    )
    node._predict_to = MethodType(MocapEkfOdom._predict_to, node)
    old_filter.angular_velocity = np.array([0.5, 0.0, 0.0])

    MocapEkfOdom._tick(node)

    # QuaternionCvEkf.predict() is the first safety layer: an unsafe
    # propagation is rolled back atomically.  _tick() then sees the restored
    # safe state and may publish it.  The two tests above exercise _tick()'s
    # independent fail-closed guard for an unsafe state that reaches it.
    assert node._filter is old_filter
    assert node._filter.initialized is True
    np.testing.assert_allclose(
        node._filter.orientation,
        np.array([0.0, 0.0, 0.0, 1.0]),
    )
    assert publisher.messages == [('odom', 'stamp')]
    assert tf_broadcaster.transforms == [('tf', 'stamp')]
    assert logger.warnings == []


def test_coast_timeout_latches_and_retains_anchor_without_publish():
    (
        node,
        old_filter,
        publisher,
        logger,
        _clock_sec,
        anchor_position,
        anchor_orientation,
    ) = _tracking_loss_fixture()
    # Simulate a coast prediction moving away from the last pose-corrected
    # state.  Entering the latch must restore the trusted anchor.
    old_filter.position = anchor_position + np.array([0.12, 0.03, -0.04])
    old_filter.orientation = quat_exp(np.array([0.0, 0.0, 0.25]))

    MocapEkfOdom._tick(node)

    assert node._filter is old_filter
    assert node._filter.initialized is True
    assert node._tracking_loss_latched is True
    assert node._last_filter_sec is None
    np.testing.assert_allclose(node._filter.position, anchor_position)
    np.testing.assert_allclose(node._filter.orientation, anchor_orientation)
    assert publisher.messages == []
    assert len(logger.warnings) == 1
    assert 'MoCap tracking lost' in logger.warnings[0]
    assert 'latched off' in logger.warnings[0]

    # The loss warning and state transition are one-shot while latched.
    MocapEkfOdom._tick(node)
    assert len(logger.warnings) == 1
    assert publisher.messages == []


def test_far_pose_after_coast_is_rejected_without_reinitialize_or_publish():
    (
        node,
        old_filter,
        publisher,
        _logger,
        _clock_sec,
        anchor_position,
        anchor_orientation,
    ) = _tracking_loss_fixture()
    MocapEkfOdom._tick(node)
    far_position = anchor_position + np.array([0.52, 0.0, 0.0])
    far_orientation = quat_exp(np.array([0.0, 0.0, 2.58]))

    accepted = node._update_filter_from_pose(
        far_position,
        far_orientation,
        10.1,
    )
    node._handle_update_result(
        accepted,
        far_position,
        far_orientation,
    )

    assert accepted is False
    assert node._filter is old_filter
    assert node._filter.initialized is True
    assert node._tracking_loss_latched is True
    assert node._last_filter_sec is None
    np.testing.assert_allclose(node._filter.position, anchor_position)
    np.testing.assert_allclose(node._filter.orientation, anchor_orientation)
    assert 'position_innovation=0.520m > 0.200m' in (
        node._filter.last_pose_rejection_reason
    )
    assert 'orientation_innovation=2.580rad > 0.550rad' in (
        node._filter.last_pose_rejection_reason
    )
    # max_rejected_samples=1 deliberately proves the generic reinitializer
    # cannot bypass a tracking-loss latch.
    assert node._consecutive_full_rejections == 1
    MocapEkfOdom._tick(node)
    assert publisher.messages == []


def test_near_anchor_pose_recovers_without_long_dt_prediction():
    (
        node,
        old_filter,
        publisher,
        logger,
        clock_sec,
        anchor_position,
        _anchor_orientation,
    ) = _tracking_loss_fixture()
    MocapEkfOdom._tick(node)
    near_position = anchor_position + np.array([0.02, -0.01, 0.01])
    near_orientation = quat_exp(np.array([0.0, 0.0, 0.10]))

    accepted = node._update_filter_from_pose(
        near_position,
        near_orientation,
        10.2,
    )

    assert accepted is True
    assert node._filter is old_filter
    assert node._tracking_loss_latched is False
    assert node._tracking_loss_started_sec is None
    assert node._last_filter_sec == 10.2
    np.testing.assert_allclose(node._filter.position, near_position)
    np.testing.assert_allclose(node._filter.orientation, near_orientation)
    np.testing.assert_allclose(node._filter.linear_velocity, np.zeros(3))
    np.testing.assert_allclose(node._filter.angular_velocity, np.zeros(3))
    assert any('MoCap tracking recovered' in msg for msg in logger.warnings)

    # Publication resumes from a short post-recovery dt, never from the full
    # unobserved interval.
    clock_sec[0] = 10.21
    node._last_pose_accepted_sec = 10.2
    MocapEkfOdom._tick(node)
    assert publisher.messages == [('odom', 'stamp')]


def test_normal_first_pose_still_initializes_without_recovery_latch():
    ekf = QuaternionCvEkf(
        max_position_innovation_m=0.20,
        max_orientation_innovation_rad=0.55,
        max_base_link_z_axis_angle_rad=0.55,
    )
    logger = _Logger()
    node = SimpleNamespace(
        _filter=ekf,
        _last_filter_sec=None,
        _tracking_loss_latched=False,
        _tracking_loss_started_sec=None,
        get_logger=lambda: logger,
    )
    node._predict_to = MethodType(MocapEkfOdom._predict_to, node)
    node._update_filter_from_pose = MethodType(
        MocapEkfOdom._update_filter_from_pose,
        node,
    )
    first_position = np.array([4.0, 0.5, 1.7], dtype=float)
    first_orientation = quat_exp(np.array([0.0, 0.0, 0.2]))

    accepted = node._update_filter_from_pose(
        first_position,
        first_orientation,
        1.0,
    )

    assert accepted is True
    assert node._filter.initialized is True
    assert node._tracking_loss_latched is False
    assert node._last_filter_sec == 1.0
    np.testing.assert_allclose(node._filter.position, first_position)
    np.testing.assert_allclose(node._filter.orientation, first_orientation)
    assert logger.warnings == []
