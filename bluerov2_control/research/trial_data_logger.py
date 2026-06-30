#!/usr/bin/env python3
"""ROS 2 logger for payload retrieval experiments.

The node samples the latest values from controller, vehicle, perception and
task-event topics into a trial directory. The output is intentionally simple
CSV/JSON so it can be used from Python, MATLAB, R or a spreadsheet when writing
the thesis results chapter.
"""

import csv
import getpass
import json
import os
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from px4_msgs.msg import (
    VehicleAttitudeSetpoint,
    VehicleControlMode,
    VehicleOdometry,
    VehicleThrustSetpoint,
    VehicleTorqueSetpoint,
)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32, String


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _default_trial_id() -> str:
    return "retrieval_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_str(value: Any) -> str:
    return "" if value is None else str(value)


def _vector_values(values: Any, count: int) -> List[float]:
    if values is None:
        return [0.0] * count
    out = []
    for index in range(count):
        try:
            out.append(float(values[index]))
        except (IndexError, TypeError, ValueError):
            out.append(0.0)
    return out


def _prefix_fields(prefix: str, names: Iterable[str]) -> List[str]:
    return [f"{prefix}_{name}" for name in names]


def _odom_fields(prefix: str) -> List[str]:
    return _prefix_fields(
        prefix,
        (
            "valid",
            "age_s",
            "x",
            "y",
            "z",
            "qw",
            "qx",
            "qy",
            "qz",
            "vx",
            "vy",
            "vz",
            "wx",
            "wy",
            "wz",
        ),
    )


def _pose_fields(prefix: str) -> List[str]:
    return _prefix_fields(
        prefix,
        ("valid", "age_s", "x", "y", "z", "qw", "qx", "qy", "qz"),
    )


def _twist_fields(prefix: str) -> List[str]:
    return _prefix_fields(
        prefix,
        (
            "valid",
            "age_s",
            "linear_x",
            "linear_y",
            "linear_z",
            "angular_x",
            "angular_y",
            "angular_z",
        ),
    )


def _xyz_fields(prefix: str) -> List[str]:
    return _prefix_fields(prefix, ("valid", "age_s", "x", "y", "z"))


SAMPLE_FIELDS = (
    [
        "trial_id",
        "sample_index",
        "time_ros_s",
        "elapsed_s",
        "time_wall_iso",
        "last_event",
        "last_event_age_s",
        "armed",
        "offboard",
        "manual_enabled",
        "attitude_enabled",
        "position_enabled",
        "velocity_enabled",
    ]
    + _odom_fields("odom")
    + _odom_fields("mocap")
    + _twist_fields("cmd")
    + _xyz_fields("thrust")
    + _xyz_fields("torque")
    + _prefix_fields(
        "attitude_sp",
        (
            "valid",
            "age_s",
            "qw",
            "qx",
            "qy",
            "qz",
            "thrust_x",
            "thrust_y",
            "thrust_z",
        ),
    )
    + _pose_fields("handle")
    + _prefix_fields("handle_confidence", ("valid", "age_s", "value"))
    + _prefix_fields("handle_detected", ("valid", "age_s", "value"))
    + _pose_fields("payload")
    + _pose_fields("dock")
)

EVENT_FIELDS = [
    "trial_id",
    "time_ros_s",
    "elapsed_s",
    "time_wall_iso",
    "event",
    "note",
    "source",
    "raw",
]


@dataclass
class LatestMessage:
    msg: Optional[Any] = None
    received_ros_s: Optional[float] = None
    received_wall_iso: str = ""

    def valid(self) -> bool:
        return self.msg is not None and self.received_ros_s is not None

    def age_s(self, now_s: float) -> Any:
        if self.received_ros_s is None:
            return ""
        return max(0.0, now_s - self.received_ros_s)


class PayloadRetrievalDataLogger(Node):
    """Sample experiment topics into a reproducible trial folder."""

    def __init__(self) -> None:
        super().__init__("payload_retrieval_data_logger")

        self.declare_parameter("trial_id", "")
        self.declare_parameter(
            "output_dir",
            "/home/yecheng/bluerov_ws/bluerov2_payload_retrieval_trials",
        )
        self.declare_parameter("sample_period_s", 0.05)
        self.declare_parameter("flush_every_n_samples", 10)
        self.declare_parameter("metadata_file", "")
        self.declare_parameter("controller_name", "")
        self.declare_parameter("environment", "sim_or_pool")
        self.declare_parameter("operator", getpass.getuser())
        self.declare_parameter("notes", "")

        self.declare_parameter("odom_topic", "/itrl_rov_1/fmu/out/vehicle_odometry")
        self.declare_parameter("mocap_odom_topic", "")
        self.declare_parameter("cmd_vel_topic", "/itrl_rov_1/cmd_vel")
        self.declare_parameter("thrust_sp_topic", "/itrl_rov_1/fmu/in/vehicle_thrust_setpoint")
        self.declare_parameter("torque_sp_topic", "/itrl_rov_1/fmu/in/vehicle_torque_setpoint")
        self.declare_parameter("attitude_sp_topic", "")
        self.declare_parameter("control_mode_topic", "/itrl_rov_1/fmu/out/vehicle_control_mode")
        self.declare_parameter("handle_pose_topic", "/payload/handle_pose")
        self.declare_parameter("handle_confidence_topic", "/payload/handle_confidence")
        self.declare_parameter("handle_detected_topic", "/payload/handle_detected")
        self.declare_parameter("payload_pose_topic", "/payload/pose")
        self.declare_parameter("dock_pose_topic", "/dock/pose")
        self.declare_parameter("task_event_topic", "/bluerov2/trial_event")

        trial_id = _as_str(self.get_parameter("trial_id").value).strip() or _default_trial_id()
        output_dir = Path(_as_str(self.get_parameter("output_dir").value)).expanduser()
        self.trial_id = trial_id
        self.trial_dir = output_dir / trial_id
        self.trial_dir.mkdir(parents=True, exist_ok=True)

        self.sample_path = self.trial_dir / "samples.csv"
        self.event_path = self.trial_dir / "events.csv"
        self.metadata_path = self.trial_dir / "metadata.json"

        self.sample_file = self.sample_path.open("w", newline="", encoding="utf-8")
        self.event_file = self.event_path.open("w", newline="", encoding="utf-8")
        self.sample_writer = csv.DictWriter(self.sample_file, fieldnames=SAMPLE_FIELDS)
        self.event_writer = csv.DictWriter(self.event_file, fieldnames=EVENT_FIELDS)
        self.sample_writer.writeheader()
        self.event_writer.writeheader()

        self.latest: Dict[str, LatestMessage] = {}
        self._topic_subscriptions = []
        self.sample_index = 0
        self.first_ros_s: Optional[float] = None
        self.closed = False
        self.last_event = ""
        self.last_event_ros_s: Optional[float] = None

        self._write_metadata()
        self._create_subscriptions()
        self._write_event("logger_start", "data logger started", "payload_retrieval_data_logger", "")

        sample_period_s = max(0.001, _as_float(self.get_parameter("sample_period_s").value, 0.05))
        self.timer = self.create_timer(sample_period_s, self._sample)

        self.get_logger().info(
            f"logging trial '{self.trial_id}' to {self.trial_dir} at {1.0 / sample_period_s:.1f} Hz"
        )

    def _topic_param(self, name: str) -> str:
        return _as_str(self.get_parameter(name).value).strip()

    def _ros_now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _elapsed_s(self, now_s: float) -> float:
        if self.first_ros_s is None:
            self.first_ros_s = now_s
        return now_s - self.first_ros_s

    def _write_metadata(self) -> None:
        metadata_file = self._topic_param("metadata_file")
        external_metadata: Dict[str, Any] = {}
        if metadata_file:
            try:
                with Path(metadata_file).expanduser().open("r", encoding="utf-8") as src:
                    external_metadata = json.load(src)
            except (OSError, json.JSONDecodeError) as exc:
                self.get_logger().warn(f"could not read metadata_file={metadata_file}: {exc}")

        topic_params = [
            "odom_topic",
            "mocap_odom_topic",
            "cmd_vel_topic",
            "thrust_sp_topic",
            "torque_sp_topic",
            "attitude_sp_topic",
            "control_mode_topic",
            "handle_pose_topic",
            "handle_confidence_topic",
            "handle_detected_topic",
            "payload_pose_topic",
            "dock_pose_topic",
            "task_event_topic",
        ]

        metadata = {
            "trial_id": self.trial_id,
            "created_wall_utc": _utc_now_iso(),
            "output_dir": str(self.trial_dir),
            "controller_name": _as_str(self.get_parameter("controller_name").value),
            "environment": _as_str(self.get_parameter("environment").value),
            "operator": _as_str(self.get_parameter("operator").value),
            "notes": _as_str(self.get_parameter("notes").value),
            "sample_period_s": _as_float(self.get_parameter("sample_period_s").value, 0.05),
            "topics": {name: self._topic_param(name) for name in topic_params},
            "machine": {
                "hostname": platform.node(),
                "platform": platform.platform(),
                "cwd": os.getcwd(),
            },
            "external_metadata": external_metadata,
        }

        with self.metadata_path.open("w", encoding="utf-8") as dst:
            json.dump(metadata, dst, indent=2, sort_keys=True)
            dst.write("\n")

    def _create_subscriptions(self) -> None:
        px4_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        general_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        event_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self._subscribe("odom", self._topic_param("odom_topic"), VehicleOdometry, px4_qos)
        self._subscribe("mocap", self._topic_param("mocap_odom_topic"), Odometry, general_qos)
        self._subscribe("cmd", self._topic_param("cmd_vel_topic"), Twist, general_qos)
        self._subscribe("thrust", self._topic_param("thrust_sp_topic"), VehicleThrustSetpoint, px4_qos)
        self._subscribe("torque", self._topic_param("torque_sp_topic"), VehicleTorqueSetpoint, px4_qos)
        self._subscribe("attitude_sp", self._topic_param("attitude_sp_topic"), VehicleAttitudeSetpoint, px4_qos)
        self._subscribe("control_mode", self._topic_param("control_mode_topic"), VehicleControlMode, px4_qos)
        self._subscribe("handle", self._topic_param("handle_pose_topic"), PoseStamped, general_qos)
        self._subscribe("handle_confidence", self._topic_param("handle_confidence_topic"), Float32, general_qos)
        self._subscribe("handle_detected", self._topic_param("handle_detected_topic"), Bool, general_qos)
        self._subscribe("payload", self._topic_param("payload_pose_topic"), PoseStamped, general_qos)
        self._subscribe("dock", self._topic_param("dock_pose_topic"), PoseStamped, general_qos)

        event_topic = self._topic_param("task_event_topic")
        if event_topic:
            self._topic_subscriptions.append(
                self.create_subscription(String, event_topic, self._on_event_msg, event_qos)
            )

    def _subscribe(self, key: str, topic: str, msg_type: Any, qos: QoSProfile) -> None:
        if not topic:
            return

        def callback(msg: Any, stored_key: str = key) -> None:
            self.latest[stored_key] = LatestMessage(msg, self._ros_now_s(), _utc_now_iso())

        self._topic_subscriptions.append(
            self.create_subscription(msg_type, topic, callback, qos)
        )

    def _on_event_msg(self, msg: String) -> None:
        raw = msg.data
        event = raw
        note = ""
        source = "ros_topic"
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                event = _as_str(payload.get("event", event)).strip() or event
                note = _as_str(payload.get("note", ""))
                source = _as_str(payload.get("source", source)) or source
        except json.JSONDecodeError:
            pass
        self._write_event(event, note, source, raw)

    def _write_event(self, event: str, note: str, source: str, raw: str) -> None:
        now_s = self._ros_now_s()
        elapsed_s = self._elapsed_s(now_s)
        event = _as_str(event).strip()
        self.last_event = event
        self.last_event_ros_s = now_s
        self.event_writer.writerow(
            {
                "trial_id": self.trial_id,
                "time_ros_s": f"{now_s:.9f}",
                "elapsed_s": f"{elapsed_s:.9f}",
                "time_wall_iso": _utc_now_iso(),
                "event": event,
                "note": _as_str(note),
                "source": _as_str(source),
                "raw": _as_str(raw),
            }
        )
        self.event_file.flush()
        self.get_logger().info(f"event: {event}")

    def _sample(self) -> None:
        now_s = self._ros_now_s()
        elapsed_s = self._elapsed_s(now_s)
        row: Dict[str, Any] = {field: "" for field in SAMPLE_FIELDS}
        row.update(
            {
                "trial_id": self.trial_id,
                "sample_index": self.sample_index,
                "time_ros_s": f"{now_s:.9f}",
                "elapsed_s": f"{elapsed_s:.9f}",
                "time_wall_iso": _utc_now_iso(),
                "last_event": self.last_event,
                "last_event_age_s": ""
                if self.last_event_ros_s is None
                else f"{max(0.0, now_s - self.last_event_ros_s):.9f}",
            }
        )

        self._put_control_mode(row, now_s)
        self._put_px4_odom(row, "odom", now_s)
        self._put_nav_odom(row, "mocap", now_s)
        self._put_twist(row, "cmd", now_s)
        self._put_xyz(row, "thrust", now_s)
        self._put_xyz(row, "torque", now_s)
        self._put_attitude_setpoint(row, now_s)
        self._put_pose(row, "handle", now_s)
        self._put_float(row, "handle_confidence", now_s)
        self._put_bool(row, "handle_detected", now_s)
        self._put_pose(row, "payload", now_s)
        self._put_pose(row, "dock", now_s)

        self.sample_writer.writerow(row)
        self.sample_index += 1
        flush_every = int(_as_float(self.get_parameter("flush_every_n_samples").value, 10))
        if flush_every <= 1 or self.sample_index % flush_every == 0:
            self.sample_file.flush()

    def _latest(self, key: str) -> LatestMessage:
        return self.latest.get(key, LatestMessage())

    def _set_valid_age(self, row: Dict[str, Any], prefix: str, latest: LatestMessage, now_s: float) -> bool:
        row[f"{prefix}_valid"] = 1 if latest.valid() else 0
        row[f"{prefix}_age_s"] = "" if not latest.valid() else f"{latest.age_s(now_s):.9f}"
        return latest.valid()

    def _put_control_mode(self, row: Dict[str, Any], now_s: float) -> None:
        latest = self._latest("control_mode")
        msg = latest.msg
        row["armed"] = "" if msg is None else int(bool(getattr(msg, "flag_armed", False)))
        row["offboard"] = "" if msg is None else int(bool(getattr(msg, "flag_control_offboard_enabled", False)))
        row["manual_enabled"] = "" if msg is None else int(bool(getattr(msg, "flag_control_manual_enabled", False)))
        row["attitude_enabled"] = "" if msg is None else int(bool(getattr(msg, "flag_control_attitude_enabled", False)))
        row["position_enabled"] = "" if msg is None else int(bool(getattr(msg, "flag_control_position_enabled", False)))
        row["velocity_enabled"] = "" if msg is None else int(bool(getattr(msg, "flag_control_velocity_enabled", False)))
        if latest.valid():
            row["last_event_age_s"] = row["last_event_age_s"]

    def _put_px4_odom(self, row: Dict[str, Any], prefix: str, now_s: float) -> None:
        latest = self._latest(prefix)
        if not self._set_valid_age(row, prefix, latest, now_s):
            return
        msg = latest.msg
        px, py, pz = _vector_values(getattr(msg, "position", None), 3)
        qw, qx, qy, qz = _vector_values(getattr(msg, "q", None), 4)
        vx, vy, vz = _vector_values(getattr(msg, "velocity", None), 3)
        wx, wy, wz = _vector_values(getattr(msg, "angular_velocity", None), 3)
        row.update(
            {
                f"{prefix}_x": px,
                f"{prefix}_y": py,
                f"{prefix}_z": pz,
                f"{prefix}_qw": qw,
                f"{prefix}_qx": qx,
                f"{prefix}_qy": qy,
                f"{prefix}_qz": qz,
                f"{prefix}_vx": vx,
                f"{prefix}_vy": vy,
                f"{prefix}_vz": vz,
                f"{prefix}_wx": wx,
                f"{prefix}_wy": wy,
                f"{prefix}_wz": wz,
            }
        )

    def _put_nav_odom(self, row: Dict[str, Any], prefix: str, now_s: float) -> None:
        latest = self._latest(prefix)
        if not self._set_valid_age(row, prefix, latest, now_s):
            return
        msg = latest.msg
        pos = msg.pose.pose.position
        quat = msg.pose.pose.orientation
        lin = msg.twist.twist.linear
        ang = msg.twist.twist.angular
        row.update(
            {
                f"{prefix}_x": float(pos.x),
                f"{prefix}_y": float(pos.y),
                f"{prefix}_z": float(pos.z),
                f"{prefix}_qw": float(quat.w),
                f"{prefix}_qx": float(quat.x),
                f"{prefix}_qy": float(quat.y),
                f"{prefix}_qz": float(quat.z),
                f"{prefix}_vx": float(lin.x),
                f"{prefix}_vy": float(lin.y),
                f"{prefix}_vz": float(lin.z),
                f"{prefix}_wx": float(ang.x),
                f"{prefix}_wy": float(ang.y),
                f"{prefix}_wz": float(ang.z),
            }
        )

    def _put_twist(self, row: Dict[str, Any], prefix: str, now_s: float) -> None:
        latest = self._latest(prefix)
        if not self._set_valid_age(row, prefix, latest, now_s):
            return
        msg = latest.msg
        row.update(
            {
                f"{prefix}_linear_x": float(msg.linear.x),
                f"{prefix}_linear_y": float(msg.linear.y),
                f"{prefix}_linear_z": float(msg.linear.z),
                f"{prefix}_angular_x": float(msg.angular.x),
                f"{prefix}_angular_y": float(msg.angular.y),
                f"{prefix}_angular_z": float(msg.angular.z),
            }
        )

    def _put_xyz(self, row: Dict[str, Any], prefix: str, now_s: float) -> None:
        latest = self._latest(prefix)
        if not self._set_valid_age(row, prefix, latest, now_s):
            return
        x, y, z = _vector_values(getattr(latest.msg, "xyz", None), 3)
        row.update({f"{prefix}_x": x, f"{prefix}_y": y, f"{prefix}_z": z})

    def _put_attitude_setpoint(self, row: Dict[str, Any], now_s: float) -> None:
        prefix = "attitude_sp"
        latest = self._latest(prefix)
        if not self._set_valid_age(row, prefix, latest, now_s):
            return
        qw, qx, qy, qz = _vector_values(getattr(latest.msg, "q_d", None), 4)
        tx, ty, tz = _vector_values(getattr(latest.msg, "thrust_body", None), 3)
        row.update(
            {
                "attitude_sp_qw": qw,
                "attitude_sp_qx": qx,
                "attitude_sp_qy": qy,
                "attitude_sp_qz": qz,
                "attitude_sp_thrust_x": tx,
                "attitude_sp_thrust_y": ty,
                "attitude_sp_thrust_z": tz,
            }
        )

    def _put_pose(self, row: Dict[str, Any], prefix: str, now_s: float) -> None:
        latest = self._latest(prefix)
        if not self._set_valid_age(row, prefix, latest, now_s):
            return
        msg = latest.msg
        pos = msg.pose.position
        quat = msg.pose.orientation
        row.update(
            {
                f"{prefix}_x": float(pos.x),
                f"{prefix}_y": float(pos.y),
                f"{prefix}_z": float(pos.z),
                f"{prefix}_qw": float(quat.w),
                f"{prefix}_qx": float(quat.x),
                f"{prefix}_qy": float(quat.y),
                f"{prefix}_qz": float(quat.z),
            }
        )

    def _put_float(self, row: Dict[str, Any], prefix: str, now_s: float) -> None:
        latest = self._latest(prefix)
        if not self._set_valid_age(row, prefix, latest, now_s):
            return
        row[f"{prefix}_value"] = float(latest.msg.data)

    def _put_bool(self, row: Dict[str, Any], prefix: str, now_s: float) -> None:
        latest = self._latest(prefix)
        if not self._set_valid_age(row, prefix, latest, now_s):
            return
        row[f"{prefix}_value"] = int(bool(latest.msg.data))

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self._write_event("logger_stop", "data logger stopped", "payload_retrieval_data_logger", "")
        except Exception:  # noqa: BLE001 - shutdown should always close files
            pass
        self.sample_file.flush()
        self.event_file.flush()
        self.sample_file.close()
        self.event_file.close()


def main() -> None:
    rclpy.init()
    node = PayloadRetrievalDataLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
