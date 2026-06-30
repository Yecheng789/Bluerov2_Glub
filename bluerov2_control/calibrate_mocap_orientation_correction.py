#!/usr/bin/env python3
"""Estimate a fixed MoCap rigid-body orientation correction."""

import argparse
import math
from statistics import mean, pstdev
from typing import List, Sequence, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


Quaternion = Tuple[float, float, float, float]


def normalize_quat(q: Sequence[float]) -> Quaternion:
    norm = math.sqrt(sum(float(v) * float(v) for v in q))
    if norm <= 1e-12 or not math.isfinite(norm):
        return (0.0, 0.0, 0.0, 1.0)
    x, y, z, w = (float(v) / norm for v in q)
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    return (x, y, z, w)


def quat_conjugate(q: Quaternion) -> Quaternion:
    x, y, z, w = normalize_quat(q)
    return (-x, -y, -z, w)


def quat_multiply(q1: Quaternion, q2: Quaternion) -> Quaternion:
    x1, y1, z1, w1 = normalize_quat(q1)
    x2, y2, z2, w2 = normalize_quat(q2)
    return normalize_quat(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )
    )


def quat_to_rpy(q: Quaternion) -> Tuple[float, float, float]:
    x, y, z, w = normalize_quat(q)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def yaw_to_quat(yaw: float) -> Quaternion:
    return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


def mean_quaternion(samples: Sequence[Quaternion]) -> Quaternion:
    reference = normalize_quat(samples[0])
    aligned: List[Quaternion] = []
    for sample in samples:
        sample = normalize_quat(sample)
        if sum(a * b for a, b in zip(sample, reference)) < 0.0:
            aligned.append(tuple(-v for v in sample))  # type: ignore[arg-type]
        else:
            aligned.append(sample)
    return normalize_quat(
        (
            mean([q[0] for q in aligned]),
            mean([q[1] for q in aligned]),
            mean([q[2] for q in aligned]),
            mean([q[3] for q in aligned]),
        )
    )


def largest_quaternion_cluster(
    samples: Sequence[Quaternion],
    cluster_angle_rad: float,
) -> List[Quaternion]:
    if not samples:
        return []

    best: List[Quaternion] = []
    for seed in samples:
        cluster = [
            sample
            for sample in samples
            if quat_angle(sample, seed) <= cluster_angle_rad
        ]
        if len(cluster) > len(best):
            best = cluster
    return best


def quat_angle(q1: Quaternion, q2: Quaternion) -> float:
    q1 = normalize_quat(q1)
    q2 = normalize_quat(q2)
    dot = abs(sum(a * b for a, b in zip(q1, q2)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def base_link_z_axis_angle_rad(q: Quaternion) -> float:
    x, y, _z, _w = normalize_quat(q)
    rzz = 1.0 - 2.0 * (x * x + y * y)
    return math.acos(max(-1.0, min(1.0, rzz)))


class CorrectionCalibrator(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("calibrate_mocap_orientation_correction")
        self.args = args
        self.quaternions: List[Quaternion] = []
        self.create_subscription(PoseStamped, args.topic, self._on_pose, 10)
        self.get_logger().info(
            f"collecting {args.samples} pose samples from {args.topic}; "
            "keep the robot physically level and still"
        )

    def _on_pose(self, msg: PoseStamped) -> None:
        if len(self.quaternions) >= self.args.samples:
            return
        q = normalize_quat(
            (
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            )
        )
        if not all(math.isfinite(v) for v in q):
            self.get_logger().warn("ignored non-finite quaternion sample")
            return
        self.quaternions.append(q)
        step = max(1, self.args.samples // 5)
        if len(self.quaternions) % step == 0:
            self.get_logger().info(
                f"collected {len(self.quaternions)}/{self.args.samples} samples"
            )

    @property
    def done(self) -> bool:
        return len(self.quaternions) >= self.args.samples


def build_report(samples: Sequence[Quaternion], args: argparse.Namespace) -> str:
    raw_samples = list(samples)
    cluster_angle_rad = math.radians(float(args.cluster_angle_deg))
    cluster = largest_quaternion_cluster(raw_samples, cluster_angle_rad)

    if len(cluster) >= int(args.min_cluster_samples):
        samples = cluster
    else:
        samples = raw_samples

    q_raw_avg = mean_quaternion(samples)
    roll, pitch, yaw = quat_to_rpy(q_raw_avg)

    q_desired = yaw_to_quat(yaw)
    q_correction = quat_multiply(quat_conjugate(q_raw_avg), q_desired)
    corrected = [quat_multiply(q, q_correction) for q in samples]

    raw_z = [base_link_z_axis_angle_rad(q) for q in samples]
    corrected_z = [base_link_z_axis_angle_rad(q) for q in corrected]
    spread = [quat_angle(q, q_raw_avg) for q in samples]

    def deg(v: float) -> float:
        return math.degrees(v)

    raw_all_avg = mean_quaternion(raw_samples)
    raw_all_spread = [quat_angle(q, raw_all_avg) for q in raw_samples]
    cluster_fraction = len(samples) / len(raw_samples)
    cluster_spread = [quat_angle(q, q_raw_avg) for q in samples]

    stable = (
        cluster_fraction >= float(args.min_cluster_fraction)
        and math.degrees(max(cluster_spread)) <= float(args.max_cluster_spread_deg)
    )

    lines = [
        "MoCap orientation correction calibration",
        "",
        "Assumption: robot was physically level during sampling; yaw is preserved.",
        "",
        f"Collected samples: {len(raw_samples)}",
        "All-sample quaternion spread from all-sample average deg: "
        f"mean={deg(mean(raw_all_spread)):.3f}, "
        f"max={deg(max(raw_all_spread)):.3f}",
        "Selected cluster: "
        f"{len(samples)}/{len(raw_samples)} samples "
        f"({100.0 * cluster_fraction:.1f}%), "
        f"cluster_angle={float(args.cluster_angle_deg):.1f} deg",
        "Selected cluster spread from cluster average deg: "
        f"mean={deg(mean(cluster_spread)):.3f}, max={deg(max(cluster_spread)):.3f}",
        "Calibration quality: "
        f"{'OK' if stable else 'UNSTABLE - do not use for MPC'}",
        "",
        "Raw average RPY deg: "
        f"roll={deg(roll):.3f}, pitch={deg(pitch):.3f}, yaw={deg(yaw):.3f}",
        "Raw base_link z-axis angle deg: "
        f"mean={deg(mean(raw_z)):.3f}, std={deg(pstdev(raw_z)) if len(raw_z) > 1 else 0.0:.3f}, "
        f"max={deg(max(raw_z)):.3f}",
        "Sample quaternion spread from average deg: "
        f"mean={deg(mean(spread)):.3f}, max={deg(max(spread)):.3f}",
        "",
        "orientation_correction_quat_xyzw:",
        (
            f"{q_correction[0]:.16g} {q_correction[1]:.16g} "
            f"{q_correction[2]:.16g} {q_correction[3]:.16g}"
        ),
        "",
        "Corrected base_link z-axis angle deg: "
        f"mean={deg(mean(corrected_z)):.3f}, "
        f"std={deg(pstdev(corrected_z)) if len(corrected_z) > 1 else 0.0:.3f}, "
        f"max={deg(max(corrected_z)):.3f}",
        "",
        "Test command:",
        (
            "ros2 launch bluerov2_control mocap_ekf_odom.launch.py "
            "rigid_body_name:=glub "
            "orientation_correction_quat_xyzw:="
            f"\"{q_correction[0]:.16g} {q_correction[1]:.16g} "
            f"{q_correction[2]:.16g} {q_correction[3]:.16g}\""
        ),
    ]
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Estimate orientation_correction_quat_xyzw from raw MoCap pose."
    )
    parser.add_argument("--topic", default="/mocap/glub/pose")
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--min-samples", type=int, default=40)
    parser.add_argument("--cluster-angle-deg", type=float, default=20.0)
    parser.add_argument("--min-cluster-samples", type=int, default=30)
    parser.add_argument("--min-cluster-fraction", type=float, default=0.6)
    parser.add_argument("--max-cluster-spread-deg", type=float, default=10.0)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    return parser.parse_known_args(argv)


def main(argv=None) -> None:
    args, ros_args = parse_args(argv)
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.min_samples <= 0:
        raise ValueError("--min-samples must be positive")
    if args.min_samples > args.samples:
        raise ValueError("--min-samples cannot be greater than --samples")
    if args.min_cluster_samples <= 0:
        raise ValueError("--min-cluster-samples must be positive")

    rclpy.init(args=ros_args)
    node = CorrectionCalibrator(args)
    start = node.get_clock().now().nanoseconds * 1e-9
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
            now = node.get_clock().now().nanoseconds * 1e-9
            if now - start > args.timeout_sec:
                if len(node.quaternions) < args.min_samples:
                    raise TimeoutError(
                        f"timed out after {args.timeout_sec:.1f}s with "
                        f"{len(node.quaternions)}/{args.samples} samples; "
                        f"need at least {args.min_samples}"
                    )
                node.get_logger().warn(
                    f"timed out after {args.timeout_sec:.1f}s with "
                    f"{len(node.quaternions)}/{args.samples} samples; "
                    "using collected samples"
                )
                break
        print(build_report(node.quaternions, args))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
