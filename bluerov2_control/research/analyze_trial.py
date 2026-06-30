#!/usr/bin/env python3
"""Offline analysis for payload retrieval trial logs."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


Vector3 = Tuple[float, float, float]


SUCCESS_EVENTS = {"success", "mission_success", "recovered", "delivered", "complete", "completed"}
FAILURE_EVENTS = {"failure", "failed", "abort", "aborted", "timeout"}
STAGE_EVENTS = [
    "start",
    "first_detection",
    "approach_start",
    "hook_attempt",
    "hooked",
    "line_through_handle",
    "return_start",
    "docked",
    "success",
    "failure",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute thesis-ready metrics from payload retrieval CSV logs."
    )
    parser.add_argument("trial", help="Trial directory, or path to samples.csv.")
    parser.add_argument("--pose-source", choices=["auto", "odom", "mocap"], default="auto")
    parser.add_argument("--payload-target", default="", help="Static payload target x,y,z if not logged.")
    parser.add_argument("--dock-target", default="", help="Static dock target x,y,z if not logged.")
    parser.add_argument("--tank-bounds", default="", help="xmin,xmax,ymin,ymax,zmin,zmax for safety analysis.")
    parser.add_argument("--stale-max-s", type=float, default=0.5, help="Max age for perception/pose data.")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--gap-max-s", type=float, default=1.0, help="Ignore path/effort gaps above this.")
    parser.add_argument("--output-dir", default="", help="Analysis output directory.")
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib figures.")
    return parser.parse_args()


def as_float(row: Dict[str, str], key: str) -> Optional[float]:
    value = row.get(key, "")
    if value in ("", None):
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def as_bool(row: Dict[str, str], key: str) -> Optional[bool]:
    value = row.get(key, "")
    if value in ("", None):
        return None
    try:
        return bool(int(float(value)))
    except ValueError:
        return value.strip().lower() in {"true", "yes", "on"}


def parse_vector(text: str) -> Optional[Vector3]:
    if not text:
        return None
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"expected x,y,z, got: {text!r}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def parse_bounds(text: str) -> Optional[Tuple[float, float, float, float, float, float]]:
    if not text:
        return None
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 6:
        raise ValueError(f"expected xmin,xmax,ymin,ymax,zmin,zmax, got: {text!r}")
    return tuple(float(part) for part in parts)  # type: ignore[return-value]


def dist(a: Vector3, b: Vector3) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as src:
        return list(csv.DictReader(src))


def trial_paths(trial_arg: str) -> Tuple[Path, Path, Path]:
    path = Path(trial_arg).expanduser().resolve()
    if path.is_file():
        trial_dir = path.parent
        sample_path = path
    else:
        trial_dir = path
        sample_path = path / "samples.csv"
    event_path = trial_dir / "events.csv"
    if not sample_path.exists():
        raise FileNotFoundError(f"samples.csv not found: {sample_path}")
    return trial_dir, sample_path, event_path


def first_time(rows: Sequence[Dict[str, str]]) -> float:
    for row in rows:
        t = as_float(row, "time_ros_s")
        if t is not None:
            return t
    return 0.0


def relative_time(row: Dict[str, str], t0: float) -> Optional[float]:
    t = as_float(row, "time_ros_s")
    if t is None:
        return None
    return t - t0


def pose_valid(row: Dict[str, str], prefix: str, stale_max_s: Optional[float] = None) -> bool:
    valid = as_bool(row, f"{prefix}_valid")
    if not valid:
        return False
    if stale_max_s is not None:
        age = as_float(row, f"{prefix}_age_s")
        if age is None or age > stale_max_s:
            return False
    return (
        as_float(row, f"{prefix}_x") is not None
        and as_float(row, f"{prefix}_y") is not None
        and as_float(row, f"{prefix}_z") is not None
    )


def pose_from_row(row: Dict[str, str], prefix: str) -> Optional[Vector3]:
    if (
        as_float(row, f"{prefix}_x") is None
        or as_float(row, f"{prefix}_y") is None
        or as_float(row, f"{prefix}_z") is None
    ):
        return None
    return (
        as_float(row, f"{prefix}_x") or 0.0,
        as_float(row, f"{prefix}_y") or 0.0,
        as_float(row, f"{prefix}_z") or 0.0,
    )


def count_valid_pose(rows: Sequence[Dict[str, str]], prefix: str) -> int:
    return sum(1 for row in rows if pose_valid(row, prefix))


def choose_pose_prefix(rows: Sequence[Dict[str, str]], requested: str) -> str:
    if requested != "auto":
        return requested
    mocap_count = count_valid_pose(rows, "mocap")
    odom_count = count_valid_pose(rows, "odom")
    return "mocap" if mocap_count > 0.5 * max(1, odom_count) else "odom"


def pose_series(
    rows: Sequence[Dict[str, str]],
    prefix: str,
    t0: float,
    stale_max_s: Optional[float] = None,
) -> List[Tuple[float, Vector3]]:
    series = []
    for row in rows:
        t = relative_time(row, t0)
        pose = pose_from_row(row, prefix)
        if t is not None and pose is not None and pose_valid(row, prefix, stale_max_s):
            series.append((t, pose))
    return series


def path_length(series: Sequence[Tuple[float, Vector3]], gap_max_s: float) -> float:
    total = 0.0
    for (t0, p0), (t1, p1) in zip(series, series[1:]):
        if 0.0 < t1 - t0 <= gap_max_s:
            total += dist(p0, p1)
    return total


def finite_difference_speed_series(
    series: Sequence[Tuple[float, Vector3]],
    gap_max_s: float,
) -> List[Tuple[float, float]]:
    speeds = []
    for (t0, p0), (t1, p1) in zip(series, series[1:]):
        dt = t1 - t0
        if 0.0 < dt <= gap_max_s:
            speeds.append((t1, dist(p0, p1) / dt))
    return speeds


def values_for_keys(row: Dict[str, str], keys: Iterable[str]) -> Optional[List[float]]:
    values = []
    for key in keys:
        value = as_float(row, key)
        if value is None:
            return None
        values.append(value)
    return values


def vector_norm_series(
    rows: Sequence[Dict[str, str]],
    keys: Iterable[str],
    valid_key: str,
    t0: float,
) -> List[Tuple[float, float]]:
    series = []
    for row in rows:
        if not as_bool(row, valid_key):
            continue
        values = values_for_keys(row, keys)
        t = relative_time(row, t0)
        if values is not None and t is not None:
            series.append((t, math.sqrt(sum(value * value for value in values))))
    return series


def time_weighted_mean(series: Sequence[Tuple[float, float]], gap_max_s: float) -> Optional[float]:
    if len(series) < 2:
        return None
    total = 0.0
    duration = 0.0
    for (t0, v0), (t1, _v1) in zip(series, series[1:]):
        dt = t1 - t0
        if 0.0 < dt <= gap_max_s:
            total += v0 * dt
            duration += dt
    if duration <= 0.0:
        return None
    return total / duration


def time_weighted_rms(series: Sequence[Tuple[float, float]], gap_max_s: float) -> Optional[float]:
    if len(series) < 2:
        return None
    total = 0.0
    duration = 0.0
    for (t0, v0), (t1, _v1) in zip(series, series[1:]):
        dt = t1 - t0
        if 0.0 < dt <= gap_max_s:
            total += v0 * v0 * dt
            duration += dt
    if duration <= 0.0:
        return None
    return math.sqrt(total / duration)


def time_integral(series: Sequence[Tuple[float, float]], gap_max_s: float) -> Optional[float]:
    if len(series) < 2:
        return None
    total = 0.0
    for (t0, v0), (t1, _v1) in zip(series, series[1:]):
        dt = t1 - t0
        if 0.0 < dt <= gap_max_s:
            total += abs(v0) * dt
    return total


def max_value(series: Sequence[Tuple[float, float]]) -> Optional[float]:
    if not series:
        return None
    return max(value for _time, value in series)


def detection_predicate(
    row: Dict[str, str],
    stale_max_s: float,
    confidence_threshold: float,
) -> bool:
    detected = as_bool(row, "handle_detected_value")
    detected_age = as_float(row, "handle_detected_age_s")
    if detected is True and detected_age is not None and detected_age <= stale_max_s:
        return True
    confidence = as_float(row, "handle_confidence_value")
    confidence_age = as_float(row, "handle_confidence_age_s")
    if (
        confidence is not None
        and confidence_age is not None
        and confidence_age <= stale_max_s
        and confidence >= confidence_threshold
    ):
        return True
    return pose_valid(row, "handle", stale_max_s)


def time_fraction(
    rows: Sequence[Dict[str, str]],
    predicate: Callable[[Dict[str, str]], bool],
    t0: float,
    gap_max_s: float,
) -> Optional[float]:
    if len(rows) < 2:
        return None
    true_time = 0.0
    total_time = 0.0
    for row0, row1 in zip(rows, rows[1:]):
        a = relative_time(row0, t0)
        b = relative_time(row1, t0)
        if a is None or b is None:
            continue
        dt = b - a
        if 0.0 < dt <= gap_max_s:
            total_time += dt
            if predicate(row0):
                true_time += dt
    if total_time <= 0.0:
        return None
    return true_time / total_time


def first_detection_time(
    rows: Sequence[Dict[str, str]],
    t0: float,
    stale_max_s: float,
    confidence_threshold: float,
) -> Optional[float]:
    for row in rows:
        if detection_predicate(row, stale_max_s, confidence_threshold):
            return relative_time(row, t0)
    return None


def confidence_values(rows: Sequence[Dict[str, str]], stale_max_s: float) -> List[float]:
    values = []
    for row in rows:
        confidence = as_float(row, "handle_confidence_value")
        age = as_float(row, "handle_confidence_age_s")
        if confidence is not None and age is not None and age <= stale_max_s:
            values.append(confidence)
    return values


def distances_to_dynamic_target(
    rows: Sequence[Dict[str, str]],
    robot_prefix: str,
    target_prefix: str,
    t0: float,
    stale_max_s: float,
) -> List[Tuple[float, float]]:
    out = []
    for row in rows:
        t = relative_time(row, t0)
        robot = pose_from_row(row, robot_prefix)
        target = pose_from_row(row, target_prefix)
        if (
            t is not None
            and robot is not None
            and target is not None
            and pose_valid(row, robot_prefix)
            and pose_valid(row, target_prefix, stale_max_s)
        ):
            out.append((t, dist(robot, target)))
    return out


def distances_to_static_target(
    series: Sequence[Tuple[float, Vector3]],
    target: Vector3,
) -> List[Tuple[float, float]]:
    return [(t, dist(pose, target)) for t, pose in series]


def summarize_distance(series: Sequence[Tuple[float, float]], prefix: str) -> Dict[str, Optional[float]]:
    if not series:
        return {
            f"{prefix}_initial_m": None,
            f"{prefix}_min_m": None,
            f"{prefix}_final_m": None,
            f"{prefix}_time_of_min_s": None,
        }
    min_t, min_d = min(series, key=lambda item: item[1])
    return {
        f"{prefix}_initial_m": series[0][1],
        f"{prefix}_min_m": min_d,
        f"{prefix}_final_m": series[-1][1],
        f"{prefix}_time_of_min_s": min_t,
    }


def read_events(event_path: Path, t0: float) -> List[Dict[str, Any]]:
    if not event_path.exists():
        return []
    events = []
    for row in read_csv(event_path):
        event = (row.get("event") or "").strip()
        t = as_float(row, "time_ros_s")
        events.append(
            {
                "event": event,
                "event_lower": event.lower(),
                "elapsed_s": None if t is None else t - t0,
                "note": row.get("note", ""),
                "source": row.get("source", ""),
            }
        )
    return events


def first_event_times(events: Sequence[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for event_name in STAGE_EVENTS:
        out[f"event_{event_name}_s"] = None
        for event in events:
            if event["event_lower"] == event_name and event["elapsed_s"] is not None:
                out[f"event_{event_name}_s"] = float(event["elapsed_s"])
                break
    return out


def infer_success(events: Sequence[Dict[str, Any]]) -> Optional[bool]:
    lowers = {event["event_lower"] for event in events}
    if lowers & SUCCESS_EVENTS:
        return True
    if lowers & FAILURE_EVENTS:
        return False
    return None


def tank_clearance(
    series: Sequence[Tuple[float, Vector3]],
    bounds: Tuple[float, float, float, float, float, float],
) -> Dict[str, Optional[float]]:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    clearances = []
    violations = 0
    for _t, (x, y, z) in series:
        c = min(x - xmin, xmax - x, y - ymin, ymax - y, z - zmin, zmax - z)
        clearances.append(c)
        if c < 0.0:
            violations += 1
    return {
        "tank_min_clearance_m": min(clearances) if clearances else None,
        "tank_violation_samples": float(violations),
    }


def safe_mean(values: Sequence[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def safe_median(values: Sequence[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def scalar(value: Optional[float]) -> Any:
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return value


def compute_metrics(
    rows: Sequence[Dict[str, str]],
    events: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, List[Tuple[float, float]]], str]:
    t0 = first_time(rows)
    pose_prefix = choose_pose_prefix(rows, args.pose_source)
    robot_series = pose_series(rows, pose_prefix, t0)

    times = [relative_time(row, t0) for row in rows]
    times = [time for time in times if time is not None]
    dt_values = [b - a for a, b in zip(times, times[1:]) if b > a]
    duration_s = max(times) - min(times) if times else 0.0

    velocity_series = vector_norm_series(
        rows,
        (f"{pose_prefix}_vx", f"{pose_prefix}_vy", f"{pose_prefix}_vz"),
        f"{pose_prefix}_valid",
        t0,
    )
    if not velocity_series:
        velocity_series = finite_difference_speed_series(robot_series, args.gap_max_s)

    angular_rate_series = vector_norm_series(
        rows,
        (f"{pose_prefix}_wx", f"{pose_prefix}_wy", f"{pose_prefix}_wz"),
        f"{pose_prefix}_valid",
        t0,
    )
    thrust_series = vector_norm_series(rows, ("thrust_x", "thrust_y", "thrust_z"), "thrust_valid", t0)
    torque_series = vector_norm_series(rows, ("torque_x", "torque_y", "torque_z"), "torque_valid", t0)
    cmd_linear_series = vector_norm_series(
        rows,
        ("cmd_linear_x", "cmd_linear_y", "cmd_linear_z"),
        "cmd_valid",
        t0,
    )
    cmd_angular_series = vector_norm_series(
        rows,
        ("cmd_angular_x", "cmd_angular_y", "cmd_angular_z"),
        "cmd_valid",
        t0,
    )

    payload_target = parse_vector(args.payload_target)
    dock_target = parse_vector(args.dock_target)
    payload_distance = (
        distances_to_static_target(robot_series, payload_target)
        if payload_target is not None
        else distances_to_dynamic_target(rows, pose_prefix, "payload", t0, args.stale_max_s)
    )
    handle_distance = distances_to_dynamic_target(rows, pose_prefix, "handle", t0, args.stale_max_s)
    dock_distance = (
        distances_to_static_target(robot_series, dock_target)
        if dock_target is not None
        else distances_to_dynamic_target(rows, pose_prefix, "dock", t0, args.stale_max_s)
    )

    conf_values = confidence_values(rows, args.stale_max_s)
    first_det = first_detection_time(rows, t0, args.stale_max_s, args.confidence_threshold)
    detection_ratio = time_fraction(
        rows,
        lambda row: detection_predicate(row, args.stale_max_s, args.confidence_threshold),
        t0,
        args.gap_max_s,
    )

    metrics: Dict[str, Any] = {
        "samples": float(len(rows)),
        "duration_s": duration_s,
        "mean_sample_dt_s": safe_mean(dt_values),
        "median_sample_dt_s": safe_median(dt_values),
        "max_sample_dt_s": max(dt_values) if dt_values else None,
        "pose_source": pose_prefix,
        "path_length_m": path_length(robot_series, args.gap_max_s),
        "mean_speed_mps": time_weighted_mean(velocity_series, args.gap_max_s),
        "rms_speed_mps": time_weighted_rms(velocity_series, args.gap_max_s),
        "max_speed_mps": max_value(velocity_series),
        "max_angular_rate_radps": max_value(angular_rate_series),
        "thrust_l1_integral_s": time_integral(thrust_series, args.gap_max_s),
        "thrust_rms_norm": time_weighted_rms(thrust_series, args.gap_max_s),
        "thrust_max_norm": max_value(thrust_series),
        "torque_l1_integral_s": time_integral(torque_series, args.gap_max_s),
        "torque_rms_norm": time_weighted_rms(torque_series, args.gap_max_s),
        "torque_max_norm": max_value(torque_series),
        "cmd_linear_rms": time_weighted_rms(cmd_linear_series, args.gap_max_s),
        "cmd_angular_rms": time_weighted_rms(cmd_angular_series, args.gap_max_s),
        "first_detection_s": first_det,
        "detection_availability_ratio": detection_ratio,
        "detection_confidence_mean": safe_mean(conf_values),
        "detection_confidence_median": safe_median(conf_values),
        "detection_confidence_min": min(conf_values) if conf_values else None,
        "detection_confidence_max": max(conf_values) if conf_values else None,
        "success": infer_success(events),
    }
    metrics.update(first_event_times(events))
    metrics.update(summarize_distance(payload_distance, "distance_to_payload"))
    metrics.update(summarize_distance(handle_distance, "distance_to_handle"))
    metrics.update(summarize_distance(dock_distance, "distance_to_dock"))

    bounds = parse_bounds(args.tank_bounds)
    if bounds is not None:
        metrics.update(tank_clearance(robot_series, bounds))

    series = {
        "speed": velocity_series,
        "thrust_norm": thrust_series,
        "torque_norm": torque_series,
        "payload_distance": payload_distance,
        "handle_distance": handle_distance,
        "dock_distance": dock_distance,
    }
    return metrics, series, pose_prefix


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    def clean(value: Any) -> Any:
        if isinstance(value, float):
            return scalar(value)
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    with path.open("w", encoding="utf-8") as dst:
        json.dump(clean(payload), dst, indent=2, sort_keys=True)
        dst.write("\n")


def write_metric_csv(path: Path, metrics: Dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.writer(dst)
        writer.writerow(["metric", "value"])
        for key in sorted(metrics):
            writer.writerow([key, "" if metrics[key] is None else metrics[key]])


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 1000.0 or (0.0 < abs(number) < 0.001):
        return f"{number:.{digits}e}"
    return f"{number:.{digits}f}"


def write_markdown(path: Path, metrics: Dict[str, Any], events: Sequence[Dict[str, Any]]) -> None:
    lines = [
        "# Payload Retrieval Trial Analysis",
        "",
        "## Key Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Success | {fmt(metrics.get('success'))} |",
        f"| Duration (s) | {fmt(metrics.get('duration_s'))} |",
        f"| Pose source | {fmt(metrics.get('pose_source'))} |",
        f"| Path length (m) | {fmt(metrics.get('path_length_m'))} |",
        f"| Max speed (m/s) | {fmt(metrics.get('max_speed_mps'))} |",
        f"| RMS speed (m/s) | {fmt(metrics.get('rms_speed_mps'))} |",
        f"| First detection (s) | {fmt(metrics.get('first_detection_s'))} |",
        f"| Detection availability | {fmt(metrics.get('detection_availability_ratio'))} |",
        f"| Mean detection confidence | {fmt(metrics.get('detection_confidence_mean'))} |",
        f"| Min distance to handle (m) | {fmt(metrics.get('distance_to_handle_min_m'))} |",
        f"| Min distance to payload (m) | {fmt(metrics.get('distance_to_payload_min_m'))} |",
        f"| Final distance to dock (m) | {fmt(metrics.get('distance_to_dock_final_m'))} |",
        f"| Thrust L1 integral (norm*s) | {fmt(metrics.get('thrust_l1_integral_s'))} |",
        f"| Torque L1 integral (norm*s) | {fmt(metrics.get('torque_l1_integral_s'))} |",
        "",
        "## Stage Times",
        "",
        "| Event | First time (s) |",
        "| --- | ---: |",
    ]
    for event_name in STAGE_EVENTS:
        lines.append(f"| {event_name} | {fmt(metrics.get(f'event_{event_name}_s'))} |")

    lines += [
        "",
        "## Event Log",
        "",
        "| Time (s) | Event | Note | Source |",
        "| ---: | --- | --- | --- |",
    ]
    for event in events:
        lines.append(
            f"| {fmt(event.get('elapsed_s'))} | {event.get('event', '')} | "
            f"{event.get('note', '')} | {event.get('source', '')} |"
        )

    lines += [
        "",
        "## Thesis Use",
        "",
        "Report detection reliability, task completion time, final docking error, path length, "
        "control effort and safety margins for each trial. Then aggregate these metrics over "
        "repeated trials for simulation, pool tests and controller variants.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def maybe_make_plots(
    out_dir: Path,
    rows: Sequence[Dict[str, str]],
    series: Dict[str, List[Tuple[float, float]]],
    pose_prefix: str,
    t0: float,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    robot = pose_series(rows, pose_prefix, t0)
    if robot:
        xs = [pose[0] for _t, pose in robot]
        ys = [pose[1] for _t, pose in robot]
        plt.figure()
        plt.plot(xs, ys, label=pose_prefix)
        for target_prefix, marker in (("handle", "x"), ("payload", "s"), ("dock", "^")):
            target = pose_series(rows, target_prefix, t0, stale_max_s=0.5)
            if target:
                plt.scatter([target[-1][1][0]], [target[-1][1][1]], marker=marker, label=target_prefix)
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "plan_view.png", dpi=160)
        plt.close()

    plt.figure()
    plotted = False
    for name in ("handle_distance", "payload_distance", "dock_distance"):
        values = series.get(name, [])
        if values:
            plt.plot([item[0] for item in values], [item[1] for item in values], label=name)
            plotted = True
    if plotted:
        plt.xlabel("time [s]")
        plt.ylabel("distance [m]")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "target_distances.png", dpi=160)
    plt.close()

    plt.figure()
    plotted = False
    for name in ("speed", "thrust_norm", "torque_norm"):
        values = series.get(name, [])
        if values:
            plt.plot([item[0] for item in values], [item[1] for item in values], label=name)
            plotted = True
    if plotted:
        plt.xlabel("time [s]")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "motion_and_control.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    trial_dir, sample_path, event_path = trial_paths(args.trial)
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else trial_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(sample_path)
    if not rows:
        raise RuntimeError(f"no samples in {sample_path}")

    t0 = first_time(rows)
    events = read_events(event_path, t0)
    metrics, series, pose_prefix = compute_metrics(rows, events, args)

    metadata_path = trial_dir / "metadata.json"
    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    payload = {
        "trial_dir": str(trial_dir),
        "sample_path": str(sample_path),
        "event_path": str(event_path),
        "metadata": metadata,
        "metrics": metrics,
        "events": events,
    }
    write_json(out_dir / "summary_metrics.json", payload)
    write_metric_csv(out_dir / "summary_metrics.csv", metrics)
    write_markdown(out_dir / "thesis_results_summary.md", metrics, events)
    if not args.no_plots:
        maybe_make_plots(out_dir, rows, series, pose_prefix, t0)

    print(f"Wrote analysis to {out_dir}")
    print(f"Duration: {fmt(metrics.get('duration_s'))} s")
    print(f"Success: {fmt(metrics.get('success'))}")
    print(f"Min handle distance: {fmt(metrics.get('distance_to_handle_min_m'))} m")


if __name__ == "__main__":
    main()
