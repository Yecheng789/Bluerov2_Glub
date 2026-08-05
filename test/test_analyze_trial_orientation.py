"""Pure-function tests for fixed-hook target-attitude analysis."""

import math

import pytest

from bluerov2_control.research.analyze_trial import (
    normalize_quaternion_wxyz,
    orientation_error_series,
    parse_args,
    quaternion_angle_error_deg,
    summarize_orientation_error,
)


def test_target_quaternion_is_strictly_validated_and_normalized():
    assert normalize_quaternion_wxyz((2.0, 0.0, 0.0, 0.0)) == (
        1.0,
        0.0,
        0.0,
        0.0,
    )
    with pytest.raises(ValueError, match="four"):
        normalize_quaternion_wxyz((1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        normalize_quaternion_wxyz((math.nan, 0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="non-zero"):
        normalize_quaternion_wxyz((0.0, 0.0, 0.0, 0.0))


def test_cli_normalizes_target_and_rejects_invalid_hold_window():
    args = parse_args(
        [
            "trial",
            "--target-quaternion-wxyz=2,0,0,0",
            "--hold-window-s=3.5",
        ]
    )
    assert args.target_quaternion_wxyz == (1.0, 0.0, 0.0, 0.0)
    assert args.hold_window_s == pytest.approx(3.5)

    with pytest.raises(SystemExit):
        parse_args(["trial", "--target-quaternion-wxyz=0,0,0,0"])
    with pytest.raises(SystemExit):
        parse_args(["trial", "--hold-window-s=nan"])


def test_attitude_error_uses_shortest_full_quaternion_angle():
    assert quaternion_angle_error_deg(
        (1.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0),
    ) == pytest.approx(0.0)
    assert quaternion_angle_error_deg(
        (0.0, 1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
    ) == pytest.approx(180.0)


def test_orientation_series_uses_only_valid_selected_pose_rows():
    rows = [
        {
            "time_ros_s": "10.0",
            "odom_valid": "1",
            "odom_x": "0",
            "odom_y": "0",
            "odom_z": "0",
            "odom_qw": "1",
            "odom_qx": "0",
            "odom_qy": "0",
            "odom_qz": "0",
        },
        {
            "time_ros_s": "11.0",
            "odom_valid": "0",
            "odom_x": "0",
            "odom_y": "0",
            "odom_z": "0",
            "odom_qw": "0",
            "odom_qx": "1",
            "odom_qy": "0",
            "odom_qz": "0",
        },
        {
            "time_ros_s": "12.0",
            "odom_valid": "1",
            "odom_x": "0",
            "odom_y": "0",
            "odom_z": "0",
            "odom_qw": "0",
            "odom_qx": "0",
            "odom_qy": "0",
            "odom_qz": "0",
        },
    ]
    assert orientation_error_series(
        rows,
        "odom",
        10.0,
        (1.0, 0.0, 0.0, 0.0),
    ) == [(0.0, pytest.approx(0.0))]


def test_hold_window_reports_rms_max_and_whole_trial_endpoints():
    metrics = summarize_orientation_error(
        [(0.0, 20.0), (3.9, 8.0), (4.0, 6.0), (7.0, 4.0), (9.0, 2.0)],
        5.0,
    )

    assert metrics["orientation_error_initial_deg"] == pytest.approx(20.0)
    assert metrics["orientation_error_min_deg"] == pytest.approx(2.0)
    assert metrics["orientation_error_final_deg"] == pytest.approx(2.0)
    assert metrics["orientation_error_hold_rms_deg"] == pytest.approx(
        math.sqrt((6.0**2 + 4.0**2 + 2.0**2) / 3.0)
    )
    assert metrics["orientation_error_hold_max_deg"] == pytest.approx(6.0)
    assert metrics["orientation_error_hold_samples"] == 3.0
    assert metrics["orientation_error_hold_coverage_s"] == pytest.approx(5.0)
