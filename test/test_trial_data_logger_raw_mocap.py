"""Focused tests for raw MoCap capture in the trial CSV logger."""

from types import SimpleNamespace

from geometry_msgs.msg import PoseStamped
import pytest

from bluerov2_control.research.trial_data_logger import (
    LatestMessage,
    PayloadRetrievalDataLogger,
    SAMPLE_FIELDS,
)


RAW_MOCAP_FIELDS = [
    "raw_mocap_valid",
    "raw_mocap_age_s",
    "raw_mocap_x",
    "raw_mocap_y",
    "raw_mocap_z",
    "raw_mocap_qw",
    "raw_mocap_qx",
    "raw_mocap_qy",
    "raw_mocap_qz",
]


def test_sample_schema_contains_one_complete_raw_mocap_pose():
    assert len(SAMPLE_FIELDS) == len(set(SAMPLE_FIELDS))
    assert [field for field in SAMPLE_FIELDS if field.startswith("raw_mocap_")] == (
        RAW_MOCAP_FIELDS
    )


def test_raw_mocap_pose_is_written_without_ekf_or_frame_conversion():
    pose = PoseStamped()
    pose.pose.position.x = 4.52
    pose.pose.position.y = -0.33
    pose.pose.position.z = 1.41
    pose.pose.orientation.x = 0.11
    pose.pose.orientation.y = -0.22
    pose.pose.orientation.z = 0.33
    pose.pose.orientation.w = -0.44

    fake = SimpleNamespace(
        latest={"raw_mocap": LatestMessage(pose, 10.0, "ignored")},
        get_parameter=lambda name: SimpleNamespace(
            value={"stale_after_s": 0.30}[name]
        ),
    )
    fake._latest = lambda key: PayloadRetrievalDataLogger._latest(fake, key)
    fake._set_valid_age = lambda row, prefix, latest, now_s: (
        PayloadRetrievalDataLogger._set_valid_age(
            fake,
            row,
            prefix,
            latest,
            now_s,
        )
    )
    row = {field: "" for field in SAMPLE_FIELDS}

    PayloadRetrievalDataLogger._put_pose(
        fake,
        row,
        "raw_mocap",
        10.125,
    )

    assert row["raw_mocap_valid"] == 1
    assert float(row["raw_mocap_age_s"]) == pytest.approx(0.125)
    assert row["raw_mocap_x"] == pytest.approx(4.52)
    assert row["raw_mocap_y"] == pytest.approx(-0.33)
    assert row["raw_mocap_z"] == pytest.approx(1.41)
    assert row["raw_mocap_qw"] == pytest.approx(-0.44)
    assert row["raw_mocap_qx"] == pytest.approx(0.11)
    assert row["raw_mocap_qy"] == pytest.approx(-0.22)
    assert row["raw_mocap_qz"] == pytest.approx(0.33)

