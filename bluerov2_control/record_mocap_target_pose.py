#!/usr/bin/env python3
"""Record an averaged target pose from MoCap odometry or pose messages."""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import pstdev
from typing import List, Sequence, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node


Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def normalize_quat(q: Quaternion) -> Quaternion:
    norm = math.sqrt(sum(v * v for v in q))
    if norm <= 1e-12 or not math.isfinite(norm):
        return (0.0, 0.0, 0.0, 1.0)
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


def pose_to_sample(msg) -> Tuple[Vector3, Quaternion, str]:
    pose = msg.pose.pose if isinstance(msg, Odometry) else msg.pose
    position = (
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
    )
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
        self.frame_id = ""

        msg_type = Odometry if args.message_type == "odom" else PoseStamped
        self.create_subscription(msg_type, args.topic, self._on_msg, 10)
        self.get_logger().info(
            f"recording {args.samples} samples from {args.topic} "
            f"as {args.message_type}"
        )

    def _on_msg(self, msg) -> None:
        if len(self.positions) >= self.args.samples:
            return

        position, quat_xyzw, frame_id = pose_to_sample(msg)
        if not all(math.isfinite(v) for v in (*position, *quat_xyzw)):
            self.get_logger().warn("ignored non-finite pose sample")
            return

        self.positions.append(position)
        self.quaternions.append(quat_xyzw)
        self.frame_id = frame_id or self.frame_id

        if len(self.positions) % max(1, self.args.samples // 5) == 0:
            self.get_logger().info(
                f"collected {len(self.positions)}/{self.args.samples} samples"
            )

    @property
    def done(self) -> bool:
        return len(self.positions) >= self.args.samples


def build_payload(node: TargetPoseRecorder) -> dict:
    position = mean_vector(node.positions)
    position_std = std_vector(node.positions)
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
            "position_m": {
                "x": position_std[0],
                "y": position_std[1],
                "z": position_std[2],
            }
        },
        "notes": [
            "Record this while the robot is physically held at the hook-engaged pose.",
            "Use the same topic that the validation logger records.",
        ],
    }


def parse_args(argv=None) -> Tuple[argparse.Namespace, Sequence[str]]:
    parser = argparse.ArgumentParser(
        description="Average MoCap samples and save a target pose JSON file."
    )
    parser.add_argument("--topic", default="/mocap/glub/odom_ekf")
    parser.add_argument("--message-type", choices=["odom", "pose"], default="odom")
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--timeout-sec", type=float, default=15.0)
    parser.add_argument(
        "--output-file",
        default=(
            "/home/yecheng/bluerov_ws/src/bluerov2_control/"
            "experiments/payload_retrieval/config/"
            "hooked_box_target_pose_from_ekf.json"
        ),
    )
    return parser.parse_known_args(argv)


def main(argv=None) -> None:
    args, ros_args = parse_args(argv)
    if args.samples <= 0:
        raise ValueError("--samples must be positive")

    rclpy.init(args=ros_args)
    node = TargetPoseRecorder(args)
    start = node.get_clock().now().nanoseconds * 1e-9
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
            now = node.get_clock().now().nanoseconds * 1e-9
            if now - start > args.timeout_sec:
                raise TimeoutError(
                    f"timed out after {args.timeout_sec:.1f}s with "
                    f"{len(node.positions)}/{args.samples} samples"
                )

        payload = build_payload(node)
        output_path = Path(args.output_file).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        node.get_logger().info(f"saved target pose to {output_path}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
