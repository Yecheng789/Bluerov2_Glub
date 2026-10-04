#!/usr/bin/env python3
"""Record an averaged target pose from MoCap odometry or pose messages."""

import argparse
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import pstdev
from typing import Dict, List, Optional, Sequence, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]
QUATERNION_MIN_NORM = 1e-6


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def normalize_quat(q: Quaternion) -> Quaternion:
    if not all(math.isfinite(value) for value in q):
        raise ValueError("quaternion contains a non-finite value")
    norm = math.sqrt(sum(v * v for v in q))
    if norm < QUATERNION_MIN_NORM:
        raise ValueError(
            f"quaternion norm {norm:.3e} is below "
            f"{QUATERNION_MIN_NORM:.3e}"
        )
    return tuple(v / norm for v in q)  # type: ignore[return-value]


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def mean_vector(samples: Sequence[Vector3]) -> Vector3:
    return (
        mean([sample[0] for sample in samples]),
        mean([sample[1] for sample in samples]),
        mean([sample[2] for sample in samples]),
    )


def std_vector(samples: Sequence[Vector3]) -> Vector3:
    if len(samples) <= 1:
        return (0.0, 0.0, 0.0)
    return (
        pstdev([sample[0] for sample in samples]),
        pstdev([sample[1] for sample in samples]),
        pstdev([sample[2] for sample in samples]),
    )


def mean_quaternion(samples: Sequence[Quaternion]) -> Quaternion:
    reference = samples[0]
    aligned: List[Quaternion] = []
    for sample in samples:
        dot = sum(a * b for a, b in zip(sample, reference))
        if dot < 0.0:
            aligned.append(tuple(-v for v in sample))  # type: ignore[arg-type]
        else:
            aligned.append(sample)

    averaged = (
        mean([sample[0] for sample in aligned]),
        mean([sample[1] for sample in aligned]),
        mean([sample[2] for sample in aligned]),
        mean([sample[3] for sample in aligned]),
    )
    return normalize_quat(averaged)


def quaternion_angle_rad(q1: Quaternion, q2: Quaternion) -> float:
    """Return the shortest rotation angle between two unit quaternions."""
    dot = abs(sum(a * b for a, b in zip(q1, q2)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def orientation_statistics(samples: Sequence[Quaternion]) -> Dict[str, float]:
    """Return geodesic orientation stability metrics in degrees."""
    averaged = mean_quaternion(samples)
    deviations = [quaternion_angle_rad(sample, averaged) for sample in samples]
    angular_std_rad = math.sqrt(mean([angle * angle for angle in deviations]))

    pairwise_spread_rad = 0.0
    for index, first in enumerate(samples):
        for second in samples[index + 1:]:
            pairwise_spread_rad = max(
                pairwise_spread_rad,
                quaternion_angle_rad(first, second),
            )

    return {
        "angular_std_deg": math.degrees(angular_std_rad),
        "max_deviation_deg": math.degrees(max(deviations, default=0.0)),
        "pairwise_spread_deg": math.degrees(pairwise_spread_rad),
    }


def header_stamp_ns(msg) -> Optional[int]:
    """Return a non-zero ROS header timestamp in nanoseconds, if present."""
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return stamp_ns if stamp_ns != 0 else None


def pose_to_sample(msg) -> Tuple[Vector3, Quaternion, str]:
    pose = msg.pose.pose if isinstance(msg, Odometry) else msg.pose
    position = (
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
    )
    if not all(math.isfinite(value) for value in position):
        raise ValueError("position contains a non-finite value")
    quat_xyzw = normalize_quat(
        (
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        )
    )
    frame_id = str(getattr(msg.header, "frame_id", ""))
    return position, quat_xyzw, frame_id


class TargetPoseRecorder(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("record_mocap_target_pose")
        self.args = args
        self.positions: List[Vector3] = []
        self.quaternions: List[Quaternion] = []
        self.header_stamps_ns: List[Optional[int]] = []
        self.receipt_times_sec: List[float] = []
        self.message_ages_sec: List[float] = []
        self.rejection_counts: Counter = Counter()
        self._last_header_stamp_ns: Optional[int] = None
        self.frame_id = ""
        self._frame_id_initialized = False

        msg_type = Odometry if args.message_type == "odom" else PoseStamped
        self.create_subscription(
            msg_type,
            args.topic,
            self._on_msg,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f"recording {args.samples} samples from {args.topic} "
            f"as {args.message_type}"
        )

    def _reject(self, reason: str, detail: str) -> None:
        self.rejection_counts[reason] += 1
        self.get_logger().warn(
            f"ignored sample ({reason}): {detail}",
            throttle_duration_sec=1.0,
        )

    def _on_msg(self, msg) -> None:
        if len(self.positions) >= self.args.samples:
            return

        stamp_ns = header_stamp_ns(msg)
        if (
            stamp_ns is not None
            and self._last_header_stamp_ns is not None
            and stamp_ns <= self._last_header_stamp_ns
        ):
            self._reject(
                "non_increasing_header_stamp",
                "header timestamp is duplicate or older than the last "
                "accepted sample",
            )
            return

        now_ros_ns = self.get_clock().now().nanoseconds
        message_age_sec: Optional[float] = None
        if stamp_ns is None:
            if self.args.max_message_age_sec > 0.0:
                self._reject(
                    "missing_header_stamp",
                    "a timestamp is required while --max-message-age-sec "
                    "is enabled",
                )
                return
        else:
            message_age_sec = (now_ros_ns - stamp_ns) * 1e-9
            if (
                self.args.max_message_age_sec > 0.0
                and message_age_sec > self.args.max_message_age_sec
            ):
                self._reject(
                    "stale_header_stamp",
                    f"age {message_age_sec:.3f}s exceeds "
                    f"{self.args.max_message_age_sec:.3f}s",
                )
                return
            if message_age_sec < -self.args.max_future_skew_sec:
                self._reject(
                    "future_header_stamp",
                    f"timestamp is {-message_age_sec:.3f}s in the future; "
                    "allowed "
                    f"skew is {self.args.max_future_skew_sec:.3f}s",
                )
                return

        try:
            position, quat_xyzw, frame_id = pose_to_sample(msg)
        except (TypeError, ValueError) as exc:
            self._reject("invalid_pose", str(exc))
            return

        if self._frame_id_initialized and frame_id != self.frame_id:
            self._reject(
                "frame_id_changed",
                f"frame changed from {self.frame_id!r} to {frame_id!r}",
            )
            return
        if not self._frame_id_initialized:
            self.frame_id = frame_id
            self._frame_id_initialized = True

        self.positions.append(position)
        self.quaternions.append(quat_xyzw)
        self.header_stamps_ns.append(stamp_ns)
        self.receipt_times_sec.append(time.monotonic())
        if message_age_sec is not None:
            self.message_ages_sec.append(message_age_sec)
        if stamp_ns is not None:
            self._last_header_stamp_ns = stamp_ns

        if len(self.positions) % max(1, self.args.samples // 5) == 0:
            self.get_logger().info(
                f"collected {len(self.positions)}/{self.args.samples} samples"
            )

    @property
    def done(self) -> bool:
        return len(self.positions) >= self.args.samples


def recording_metrics(node: TargetPoseRecorder) -> dict:
    """Calculate metrics for the accepted sample window."""
    position_std = std_vector(node.positions)
    orientation = orientation_statistics(node.quaternions)

    all_timestamped = (
        len(node.header_stamps_ns) == len(node.positions)
        and all(stamp is not None for stamp in node.header_stamps_ns)
    )
    if all_timestamped and len(node.header_stamps_ns) > 1:
        first_stamp = node.header_stamps_ns[0]
        last_stamp = node.header_stamps_ns[-1]
        assert first_stamp is not None and last_stamp is not None
        sampling_span_sec = (last_stamp - first_stamp) * 1e-9
        sampling_span_source = "header_timestamp"
    elif len(node.receipt_times_sec) > 1:
        sampling_span_sec = (
            node.receipt_times_sec[-1] - node.receipt_times_sec[0]
        )
        sampling_span_source = "monotonic_receipt_time"
    else:
        sampling_span_sec = 0.0
        sampling_span_source = (
            "header_timestamp" if all_timestamped else "monotonic_receipt_time"
        )

    if node.message_ages_sec:
        message_age = {
            "min": min(node.message_ages_sec),
            "mean": mean(node.message_ages_sec),
            "max": max(node.message_ages_sec),
        }
    else:
        message_age = {"min": None, "mean": None, "max": None}

    return {
        "accepted_sample_count": len(node.positions),
        "rejected_sample_count": sum(node.rejection_counts.values()),
        "rejected_samples_by_reason": dict(
            sorted(node.rejection_counts.items())
        ),
        "header_timestamped_sample_count": sum(
            stamp is not None for stamp in node.header_stamps_ns
        ),
        "distinct_header_timestamp_count": len(
            {stamp for stamp in node.header_stamps_ns if stamp is not None}
        ),
        "sampling_span_sec": sampling_span_sec,
        "sampling_span_source": sampling_span_source,
        "message_age_sec": message_age,
        "position_standard_deviation_m": {
            "x": position_std[0],
            "y": position_std[1],
            "z": position_std[2],
        },
        "max_axis_position_standard_deviation_m": max(position_std),
        "orientation": orientation,
    }


def validation_failures(metrics: dict, args: argparse.Namespace) -> List[str]:
    """Return human-readable reasons why the recording must not be saved."""
    failures: List[str] = []
    if metrics["sampling_span_sec"] < args.min_sampling_span_sec:
        failures.append(
            f"sampling span {metrics['sampling_span_sec']:.3f}s is below "
            f"{args.min_sampling_span_sec:.3f}s"
        )
    if (
        metrics["max_axis_position_standard_deviation_m"]
        > args.max_position_std_m
    ):
        failures.append(
            "maximum position standard deviation "
            f"{metrics['max_axis_position_standard_deviation_m']:.4f}m "
            "exceeds "
            f"{args.max_position_std_m:.4f}m"
        )
    orientation = metrics["orientation"]
    if orientation["angular_std_deg"] > args.max_orientation_std_deg:
        failures.append(
            f"orientation angular std {orientation['angular_std_deg']:.3f}deg "
            f"exceeds {args.max_orientation_std_deg:.3f}deg"
        )
    if orientation["pairwise_spread_deg"] > args.max_orientation_spread_deg:
        failures.append(
            f"orientation spread {orientation['pairwise_spread_deg']:.3f}deg "
            f"exceeds {args.max_orientation_spread_deg:.3f}deg"
        )
    return failures


def validation_settings(args: argparse.Namespace) -> dict:
    return {
        "required_sample_count": args.samples,
        "max_message_age_sec": args.max_message_age_sec,
        "max_future_timestamp_skew_sec": args.max_future_skew_sec,
        "min_sampling_span_sec": args.min_sampling_span_sec,
        "max_position_standard_deviation_m": args.max_position_std_m,
        "max_orientation_angular_std_deg": args.max_orientation_std_deg,
        "max_orientation_pairwise_spread_deg": args.max_orientation_spread_deg,
        "quaternion_minimum_norm": QUATERNION_MIN_NORM,
    }


def build_payload(
    node: TargetPoseRecorder,
    metrics: Optional[dict] = None,
) -> dict:
    metrics = metrics if metrics is not None else recording_metrics(node)
    position = mean_vector(node.positions)
    quat_xyzw = mean_quaternion(node.quaternions)

    return {
        "description": "Averaged fixed hook target pose recorded from MoCap.",
        "frame_id": node.frame_id,
        "source_topic": node.args.topic,
        "message_type": (
            "nav_msgs/Odometry"
            if node.args.message_type == "odom"
            else "geometry_msgs/PoseStamped"
        ),
        "recorded_wall_utc": utc_now_iso(),
        "sample_count": len(node.positions),
        "target_pose": {
            "position": {
                "x": position[0],
                "y": position[1],
                "z": position[2],
            },
            "orientation_xyzw": {
                "x": quat_xyzw[0],
                "y": quat_xyzw[1],
                "z": quat_xyzw[2],
                "w": quat_xyzw[3],
            },
        },
        "sample_standard_deviation": {
            "position_m": metrics["position_standard_deviation_m"],
            "orientation_angular_std_deg": metrics["orientation"][
                "angular_std_deg"
            ],
        },
        "validation": {
            "passed": True,
            "settings": validation_settings(node.args),
            "metrics": metrics,
            "failures": [],
        },
        "notes": [
            "Record this while the robot is physically held at the "
            "hook-engaged pose.",
            "Use the same topic that the validation logger records.",
        ],
    }


def parse_args(argv=None) -> Tuple[argparse.Namespace, Sequence[str]]:
    parser = argparse.ArgumentParser(
        description="Average MoCap samples and save a target pose JSON file."
    )
    parser.add_argument("--topic", default="/mocap/glub/pose")
    parser.add_argument(
        "--message-type", choices=["odom", "pose"], default="pose"
    )
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--timeout-sec", type=float, default=15.0)
    parser.add_argument(
        "--max-message-age-sec",
        type=float,
        default=0.2,
        help=(
            "Reject timestamped messages older than this; a positive value "
            "also "
            "requires a non-zero header timestamp. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--max-future-skew-sec",
        type=float,
        default=0.1,
        help="Reject header timestamps farther than this into the future.",
    )
    parser.add_argument(
        "--min-sampling-span-sec",
        type=float,
        default=0.75,
        help="Minimum time from the first to last accepted sample.",
    )
    parser.add_argument(
        "--max-position-std-m",
        type=float,
        default=0.015,
        help=(
            "Maximum population standard deviation allowed on any position "
            "axis."
        ),
    )
    parser.add_argument(
        "--max-orientation-std-deg",
        type=float,
        default=1.5,
        help="Maximum RMS geodesic orientation deviation from the mean.",
    )
    parser.add_argument(
        "--max-orientation-spread-deg",
        type=float,
        default=5.0,
        help="Maximum pairwise orientation angular distance.",
    )
    parser.add_argument(
        "--output-file",
        default=(
            "/home/yecheng/bluerov_ws/src/bluerov2_control/"
            "experiments/payload_retrieval/config/"
            "hooked_box_target_pose_raw.json"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacing an existing output file.",
    )
    return parser.parse_known_args(argv)


def main(argv=None) -> None:
    args, ros_args = parse_args(argv)
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.timeout_sec <= 0.0:
        raise ValueError("--timeout-sec must be positive")
    nonnegative_settings = {
        "--max-message-age-sec": args.max_message_age_sec,
        "--max-future-skew-sec": args.max_future_skew_sec,
        "--min-sampling-span-sec": args.min_sampling_span_sec,
        "--max-position-std-m": args.max_position_std_m,
        "--max-orientation-std-deg": args.max_orientation_std_deg,
        "--max-orientation-spread-deg": args.max_orientation_spread_deg,
    }
    for name, value in nonnegative_settings.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    output_path = Path(args.output_file).expanduser()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing target file: {output_path}; "
            "choose another --output-file or pass --overwrite explicitly"
        )

    rclpy.init(args=ros_args)
    node = TargetPoseRecorder(args)
    start = time.monotonic()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
            now = time.monotonic()
            if now - start > args.timeout_sec:
                raise TimeoutError(
                    f"timed out after {args.timeout_sec:.1f}s with "
                    f"{len(node.positions)}/{args.samples} accepted samples; "
                    f"rejections={dict(node.rejection_counts)}"
                )

        metrics = recording_metrics(node)
        failures = validation_failures(metrics, args)
        if failures:
            raise ValueError(
                "target pose validation failed; no file was saved: "
                + "; ".join(failures)
            )

        payload = build_payload(node, metrics)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_mode = "w" if args.overwrite else "x"
        with output_path.open(write_mode, encoding="utf-8") as output_file:
            output_file.write(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
        node.get_logger().info(f"saved target pose to {output_path}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
