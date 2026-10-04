"""Focused safety tests for fixed-hook target pose recording."""

import math

import pytest
import rclpy
from nav_msgs.msg import Odometry

from bluerov2_control.record_mocap_target_pose import (
    TargetPoseRecorder,
    main,
    normalize_quat,
    orientation_statistics,
    parse_args,
    pose_to_sample,
    validation_failures,
)


def _valid_odometry(stamp):
    message = Odometry()
    message.header.stamp = stamp
    message.header.frame_id = 'qualisys_world'
    message.pose.pose.orientation.w = 1.0
    return message


def test_invalid_quaternions_are_rejected():
    """Zero-norm and non-finite quaternions must never become identity."""
    with pytest.raises(ValueError, match='norm'):
        normalize_quat((0.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match='non-finite'):
        normalize_quat((0.0, 0.0, math.nan, 1.0))


def test_non_finite_position_is_rejected():
    """A non-finite position must make the complete pose invalid."""
    message = Odometry()
    message.pose.pose.position.x = math.inf
    message.pose.pose.orientation.w = 1.0
    with pytest.raises(ValueError, match='position'):
        pose_to_sample(message)


def test_orientation_statistics_handle_quaternion_sign():
    """Equivalent positive and negative quaternions have zero spread."""
    metrics = orientation_statistics(
        [(0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, -1.0)]
    )
    assert metrics['angular_std_deg'] == pytest.approx(0.0)
    assert metrics['pairwise_spread_deg'] == pytest.approx(0.0)


def test_duplicate_and_older_header_stamps_are_not_collected():
    """Only strictly increasing, non-zero header stamps are accepted."""
    args, _ = parse_args(
        [
            '--samples',
            '3',
            '--max-message-age-sec',
            '2.0',
            '--min-sampling-span-sec',
            '0',
        ]
    )
    rclpy.init(args=None)
    node = TargetPoseRecorder(args)
    try:
        first_stamp = node.get_clock().now().to_msg()
        first = _valid_odometry(first_stamp)
        node._on_msg(first)
        node._on_msg(first)

        older_stamp = type(first_stamp)()
        older_stamp.sec = first_stamp.sec
        older_stamp.nanosec = max(0, first_stamp.nanosec - 1)
        node._on_msg(_valid_odometry(older_stamp))

        assert len(node.positions) == 1
        assert node.rejection_counts['non_increasing_header_stamp'] == 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_existing_output_requires_explicit_overwrite(tmp_path):
    """Recorder must fail before ROS starts when its output already exists."""
    output = tmp_path / 'target.json'
    output.write_text('{}\n', encoding='utf-8')
    with pytest.raises(FileExistsError, match='--overwrite'):
        main(['--output-file', str(output)])


def test_safety_defaults_are_enabled():
    """Default recording uses finite, non-zero safety thresholds."""
    args, _ = parse_args([])
    assert args.topic == '/mocap/glub/pose'
    assert args.max_message_age_sec == pytest.approx(0.2)
    assert args.min_sampling_span_sec == pytest.approx(0.75)
    assert args.max_position_std_m == pytest.approx(0.015)
    assert args.max_orientation_std_deg == pytest.approx(1.5)
    assert args.max_orientation_spread_deg == pytest.approx(5.0)
    assert args.overwrite is False


def test_stability_thresholds_block_an_unsafe_recording():
    """Every configured window-level threshold can prevent saving."""
    args, _ = parse_args([])
    metrics = {
        'sampling_span_sec': 0.1,
        'max_axis_position_standard_deviation_m': 0.02,
        'orientation': {
            'angular_std_deg': 2.0,
            'pairwise_spread_deg': 6.0,
        },
    }
    failures = validation_failures(metrics, args)
    assert len(failures) == 4
