#!/usr/bin/env python3
"""Generate publication-ready figures for three successful retrieval trials."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PHASE_ORDER = ("Approach", "Forward", "Wait", "Retreat")
PHASE_COLORS = {
    "Approach": "#0072B2",
    "Forward": "#E69F00",
    "Wait": "#CC79A7",
    "Retreat": "#009E73",
}
TRIAL_COLORS = ("#0072B2", "#D55E00", "#009E73")
DISPLAY_RATE_HZ = 5.0


@dataclass
class Trial:
    """Loaded samples and derived fixed-hook coordinates for one trial."""

    spec: Dict[str, Any]
    metadata: Dict[str, Any]
    notes: Dict[str, Any]
    samples: pd.DataFrame
    window_samples: pd.DataFrame
    data: pd.DataFrame
    prehook: np.ndarray
    hook: np.ndarray
    line_unit: np.ndarray
    line_normal: np.ndarray
    line_length_m: float
    color: str

    @property
    def label(self) -> str:
        return str(self.spec["short_label"])

    @property
    def panel_label(self) -> str:
        return str(self.spec["panel_label"])

    @property
    def complete(self) -> bool:
        return bool(self.spec["controller_complete"])

    @property
    def go_time(self) -> float:
        return float(self.spec["go_forward_ros_s"])

    @property
    def wait_time(self) -> float:
        return float(self.spec["wait_hook_ros_s"])

    @property
    def back_time(self) -> float:
        return float(self.spec["go_back_ros_s"])

    @property
    def end_time(self) -> float:
        return float(self.spec["analysis_end_ros_s"])


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    workspace = script_path.parents[3]
    default_root = workspace / "bluerov2_payload_retrieval_trials"
    parser = argparse.ArgumentParser(
        description=(
            "Create vector and high-resolution raster paper figures from "
            "the three operator-confirmed Splash retrieval trials."
        )
    )
    parser.add_argument(
        "--trials-root",
        type=Path,
        default=default_root,
        help="Directory containing the three trial folders.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=script_path.with_name("three_success_trial_annotations.json"),
        help="Controller-state timestamp annotations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_root / "paper_figures_three_success",
        help="Destination for figures, metrics, and captions.",
    )
    return parser.parse_args()


def configure_plot_style() -> None:
    """Use a compact journal style with embedded TrueType text in PDF."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8.2,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.35,
            "lines.linewidth": 1.15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def normalized_quaternion_error_deg(
    frame: pd.DataFrame,
    target_wxyz: np.ndarray,
) -> np.ndarray:
    """Return shortest full-attitude error for every valid WXYZ sample."""
    quaternions = frame[
        ["odom_qw", "odom_qx", "odom_qy", "odom_qz"]
    ].to_numpy(dtype=float, copy=True)
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12):
        raise ValueError("invalid odometry quaternion in plotted interval")
    quaternions /= norms[:, None]
    target = np.asarray(target_wxyz, dtype=float)
    target /= np.linalg.norm(target)
    dot = np.clip(np.abs(quaternions @ target), -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def load_trial(
    spec: Dict[str, Any],
    trials_root: Path,
    color: str,
) -> Trial:
    """Load one trial and derive phase, line, error, and reference columns."""
    trial_dir = trials_root / str(spec["directory"])
    metadata = json.loads((trial_dir / "metadata.json").read_text())
    notes = json.loads(metadata["notes"])
    if metadata["trial_id"] != spec["trial_id"]:
        raise ValueError(
            f"{trial_dir}: trial_id mismatch between metadata and annotations"
        )

    samples = pd.read_csv(trial_dir / "samples.csv")
    required = [
        "time_ros_s",
        "odom_valid",
        "odom_x",
        "odom_y",
        "odom_z",
        "odom_qw",
        "odom_qx",
        "odom_qy",
        "odom_qz",
        "thrust_x",
        "thrust_y",
        "thrust_z",
        "torque_x",
        "torque_y",
        "torque_z",
    ]
    missing = sorted(set(required) - set(samples.columns))
    if missing:
        raise ValueError(f"{trial_dir}: missing columns: {missing}")

    start_time = float(spec["mission_enable_ros_s"])
    end_time = float(spec["analysis_end_ros_s"])
    if not (
        start_time
        < float(spec["go_forward_ros_s"])
        < float(spec["wait_hook_ros_s"])
        < float(spec["go_back_ros_s"])
        < end_time
    ):
        raise ValueError(f"{trial_dir}: phase timestamps are not ordered")

    window = samples[
        (samples["time_ros_s"] >= start_time)
        & (samples["time_ros_s"] <= end_time)
    ].copy()
    pose_columns = required[2:9]
    valid = window["odom_valid"].eq(1) & window[pose_columns].notna().all(axis=1)
    data = window.loc[valid].copy().sort_values("time_ros_s")
    if data.empty:
        raise ValueError(f"{trial_dir}: no valid odometry in annotated window")

    prehook = np.asarray(notes["pre_approach_ned_m"], dtype=float)
    hook = np.asarray(notes["goal_ned_m"], dtype=float)
    line_delta = hook - prehook
    line_length = float(np.linalg.norm(line_delta[:2]))
    if not math.isfinite(line_length) or line_length <= 1e-9:
        raise ValueError(f"{trial_dir}: invalid pre-hook to Hook line")
    line_unit = np.array(
        [line_delta[0] / line_length, line_delta[1] / line_length, 0.0]
    )
    line_normal = np.array([-line_unit[1], line_unit[0], 0.0])

    position = data[["odom_x", "odom_y", "odom_z"]].to_numpy(dtype=float)
    relative = position - prehook
    data["line_progress_m"] = relative @ line_unit
    data["cross_track_m"] = relative @ line_normal
    data["depth_error_m"] = position[:, 2] - hook[2]
    data["attitude_error_deg"] = normalized_quaternion_error_deg(
        data,
        np.asarray(notes["goal_quaternion_wxyz_ned_frd"], dtype=float),
    )
    data["time_from_go_s"] = data["time_ros_s"] - float(
        spec["go_forward_ros_s"]
    )

    phase_conditions = [
        data["time_ros_s"] < float(spec["go_forward_ros_s"]),
        data["time_ros_s"] < float(spec["wait_hook_ros_s"]),
        data["time_ros_s"] < float(spec["go_back_ros_s"]),
    ]
    data["phase"] = np.select(
        phase_conditions,
        ["Approach", "Forward", "Wait"],
        default="Retreat",
    )

    speed = float(notes["final_approach_speed_mps"])
    time_values = data["time_ros_s"].to_numpy(dtype=float)
    reference = np.zeros(len(data), dtype=float)
    forward_mask = (
        (time_values >= float(spec["go_forward_ros_s"]))
        & (time_values < float(spec["wait_hook_ros_s"]))
    )
    wait_mask = (
        (time_values >= float(spec["wait_hook_ros_s"]))
        & (time_values < float(spec["go_back_ros_s"]))
    )
    retreat_mask = time_values >= float(spec["go_back_ros_s"])
    reference[forward_mask] = np.clip(
        speed * (time_values[forward_mask] - float(spec["go_forward_ros_s"])),
        0.0,
        line_length,
    )
    reference[wait_mask] = line_length
    reference[retreat_mask] = np.clip(
        line_length
        - speed
        * (time_values[retreat_mask] - float(spec["go_back_ros_s"])),
        0.0,
        line_length,
    )
    data["line_reference_m"] = reference

    return Trial(
        spec=spec,
        metadata=metadata,
        notes=notes,
        samples=samples,
        window_samples=window,
        data=data,
        prehook=prehook,
        hook=hook,
        line_unit=line_unit,
        line_normal=line_normal,
        line_length_m=line_length,
        color=color,
    )


def display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Downsample to about 5 Hz for compact vector graphics, without smoothing."""
    if len(frame) <= 2:
        return frame
    median_dt = float(frame["time_ros_s"].diff().median())
    stride = max(1, int(round(1.0 / (DISPLAY_RATE_HZ * median_dt))))
    indices = list(range(0, len(frame), stride))
    if indices[-1] != len(frame) - 1:
        indices.append(len(frame) - 1)
    return frame.iloc[indices]


def phase_frame(trial: Trial, phase: str) -> pd.DataFrame:
    return trial.data.loc[trial.data["phase"].eq(phase)]


def path_length_m(frame: pd.DataFrame) -> float:
    """Integrate valid 3-D steps without bridging logging/odometry gaps."""
    if len(frame) < 2:
        return 0.0
    time_values = frame["time_ros_s"].to_numpy(dtype=float)
    position = frame[["odom_x", "odom_y", "odom_z"]].to_numpy(dtype=float)
    dt = np.diff(time_values)
    steps = np.linalg.norm(np.diff(position, axis=0), axis=1)
    return float(np.sum(steps[(dt > 0.0) & (dt <= 0.2)]))


def rms(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.sqrt(np.mean(array * array))) if len(array) else math.nan


def phase_metrics(trial: Trial, phase: str) -> Dict[str, Any]:
    frame = phase_frame(trial, phase)
    if frame.empty:
        raise ValueError(f"{trial.label}: no data for phase {phase}")
    target = trial.prehook if phase == "Retreat" else trial.hook
    final_position = frame[["odom_x", "odom_y", "odom_z"]].iloc[-1].to_numpy(
        dtype=float
    )
    thrust_x = frame["thrust_x"].dropna().to_numpy(dtype=float)
    return {
        "scenario": trial.label,
        "trial_id": trial.spec["trial_id"],
        "phase": phase,
        "sample_count": int(len(frame)),
        "duration_s": float(
            frame["time_ros_s"].iloc[-1] - frame["time_ros_s"].iloc[0]
        ),
        "ekf_cumulative_path_m": path_length_m(frame),
        "cross_track_rmse_cm": 100.0 * rms(frame["cross_track_m"]),
        "cross_track_max_abs_cm": 100.0
        * float(frame["cross_track_m"].abs().max()),
        "depth_rmse_cm": 100.0 * rms(frame["depth_error_m"]),
        "depth_max_abs_cm": 100.0 * float(frame["depth_error_m"].abs().max()),
        "attitude_rmse_deg": rms(frame["attitude_error_deg"]),
        "attitude_max_deg": float(frame["attitude_error_deg"].max()),
        "endpoint_position_error_cm": 100.0
        * float(np.linalg.norm(final_position - target)),
        "surge_setpoint_rms": rms(thrust_x),
        "surge_saturation_ratio": (
            float(
                np.mean(
                    np.abs(thrust_x)
                    >= 0.95 * float(trial.notes["thrust_sat"])
                )
            )
            if len(thrust_x)
            else math.nan
        ),
    }


def summary_metrics(trial: Trial, phase_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_phase = {
        row["phase"]: row for row in phase_rows if row["scenario"] == trial.label
    }
    window = trial.window_samples
    valid_ratio = float(window["odom_valid"].fillna(0).eq(1).mean())
    if {"raw_mocap_valid", "raw_mocap_age_s"}.issubset(window.columns):
        maximum_raw_age = float(trial.notes["max_raw_mocap_message_age_sec"])
        raw_fresh = window["raw_mocap_valid"].fillna(0).eq(1) & window[
            "raw_mocap_age_s"
        ].le(maximum_raw_age)
        raw_fresh_ratio = float(raw_fresh.mean())
    else:
        raw_fresh_ratio = math.nan
    return {
        "scenario": trial.label,
        "trial_id": trial.spec["trial_id"],
        "controller_complete": trial.complete,
        "operator_confirmed_physical_success": bool(
            trial.spec["operator_confirmed_physical_success"]
        ),
        "analysis_end_reason": trial.spec["analysis_end_reason"],
        "approach_duration_s": trial.go_time
        - float(trial.spec["mission_enable_ros_s"]),
        "forward_duration_s": trial.wait_time - trial.go_time,
        "wait_duration_s": trial.back_time - trial.wait_time,
        "retreat_duration_s": trial.end_time - trial.back_time,
        "annotated_mission_duration_s": trial.end_time
        - float(trial.spec["mission_enable_ros_s"]),
        "odom_valid_ratio": valid_ratio,
        "raw_mocap_fresh_ratio": raw_fresh_ratio,
        "forward_ekf_cumulative_path_m": by_phase["Forward"][
            "ekf_cumulative_path_m"
        ],
        "retreat_ekf_cumulative_path_m": by_phase["Retreat"][
            "ekf_cumulative_path_m"
        ],
        "forward_cross_track_rmse_cm": by_phase["Forward"][
            "cross_track_rmse_cm"
        ],
        "retreat_cross_track_rmse_cm": by_phase["Retreat"][
            "cross_track_rmse_cm"
        ],
        "forward_depth_rmse_cm": by_phase["Forward"]["depth_rmse_cm"],
        "retreat_depth_rmse_cm": by_phase["Retreat"]["depth_rmse_cm"],
        "forward_attitude_rmse_deg": by_phase["Forward"][
            "attitude_rmse_deg"
        ],
        "retreat_attitude_rmse_deg": by_phase["Retreat"][
            "attitude_rmse_deg"
        ],
        "retreat_endpoint_position_error_cm": by_phase["Retreat"][
            "endpoint_position_error_cm"
        ],
        "retreat_surge_saturation_percent": 100.0
        * by_phase["Retreat"]["surge_saturation_ratio"],
    }


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    """Write editable vectors plus a 600 dpi review/Word raster."""
    for suffix in ("pdf", "svg"):
        figure.savefig(output_dir / f"{stem}.{suffix}")
    figure.savefig(output_dir / f"{stem}.png", dpi=600)
    plt.close(figure)


def add_phase_background(axis: plt.Axes, trial: Trial) -> None:
    origin = trial.go_time
    intervals = (
        (0.0, trial.wait_time - origin, "Forward"),
        (trial.wait_time - origin, trial.back_time - origin, "Wait"),
        (trial.back_time - origin, trial.end_time - origin, "Retreat"),
    )
    for start, end, phase in intervals:
        axis.axvspan(
            start,
            end,
            facecolor=PHASE_COLORS[phase],
            alpha=0.055,
            linewidth=0.0,
            zorder=0,
        )
    axis.axvline(
        trial.wait_time - origin,
        color=PHASE_COLORS["Wait"],
        linewidth=0.75,
        linestyle=":",
        zorder=1,
    )
    axis.axvline(
        trial.back_time - origin,
        color=PHASE_COLORS["Retreat"],
        linewidth=0.75,
        linestyle=":",
        zorder=1,
    )


def make_plan_view(trials: List[Trial], output_dir: Path) -> None:
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.25, 2.75),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    all_x = np.concatenate([trial.data["odom_x"].to_numpy() for trial in trials])
    all_y = np.concatenate([trial.data["odom_y"].to_numpy() for trial in trials])
    x_padding = max(0.08, 0.04 * float(np.ptp(all_x)))
    y_padding = max(0.08, 0.06 * float(np.ptp(all_y)))

    for axis, trial in zip(axes, trials):
        for phase in PHASE_ORDER:
            frame = display_frame(phase_frame(trial, phase))
            axis.plot(
                frame["odom_x"],
                frame["odom_y"],
                color=PHASE_COLORS[phase],
                label=phase,
                zorder=2,
            )
        axis.plot(
            [trial.prehook[0], trial.hook[0]],
            [trial.prehook[1], trial.hook[1]],
            color="0.25",
            linestyle="--",
            linewidth=0.9,
            zorder=1,
        )
        first = trial.data.iloc[0]
        last = trial.data.iloc[-1]
        axis.scatter(
            first["odom_x"],
            first["odom_y"],
            s=27,
            marker="o",
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )
        axis.scatter(
            trial.prehook[0],
            trial.prehook[1],
            s=31,
            marker="D",
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            zorder=6,
        )
        axis.scatter(
            trial.hook[0],
            trial.hook[1],
            s=70,
            marker="*",
            facecolor="#D62728",
            edgecolor="black",
            linewidth=0.5,
            zorder=7,
        )
        axis.scatter(
            last["odom_x"],
            last["odom_y"],
            s=33,
            marker="s" if trial.complete else "X",
            facecolor="white" if trial.complete else "#D62728",
            edgecolor="black",
            linewidth=0.7,
            zorder=7,
        )
        axis.set_title(f"{trial.panel_label} {trial.label}")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True)
        axis.set_xlim(float(np.min(all_x) - x_padding), float(np.max(all_x) + x_padding))
        axis.set_ylim(float(np.min(all_y) - y_padding), float(np.max(all_y) + y_padding))

    axes[0].set_ylabel(r"NED $y$ position (m)")
    figure.supxlabel(r"NED $x$ position (m)")
    legend_handles = [
        Line2D([0], [0], color=PHASE_COLORS[phase], label=phase)
        for phase in PHASE_ORDER
    ]
    legend_handles += [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            label="Mission start",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            label="Pre-hook",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="None",
            markerfacecolor="#D62728",
            markeredgecolor="black",
            markersize=9,
            label="Hook pose",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            linestyle="None",
            color="#D62728",
            label="MoCap/odom loss",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=8,
        frameon=False,
        handlelength=1.5,
    )
    save_figure(figure, output_dir, "fig01_plan_view_trajectories")


def make_hook_tracking(trials: List[Trial], output_dir: Path) -> None:
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(7.25, 6.4),
        constrained_layout=True,
    )
    all_hook_data = pd.concat(
        [trial.data.loc[trial.data["time_ros_s"] >= trial.go_time] for trial in trials]
    )
    max_error_cm = 100.0 * max(
        float(all_hook_data["cross_track_m"].abs().max()),
        float(all_hook_data["depth_error_m"].abs().max()),
    )
    error_limit = max(4.0, math.ceil(max_error_cm + 0.5))

    for row, trial in enumerate(trials):
        frame = display_frame(
            trial.data.loc[trial.data["time_ros_s"] >= trial.go_time]
        )
        progress_axis, error_axis = axes[row]
        for axis in (progress_axis, error_axis):
            add_phase_background(axis, trial)
            axis.grid(True)
            axis.set_xlim(0.0, trial.end_time - trial.go_time)

        progress_axis.plot(
            frame["time_from_go_s"],
            frame["line_progress_m"],
            color=trial.color,
            label="Measured",
        )
        progress_axis.plot(
            frame["time_from_go_s"],
            frame["line_reference_m"],
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="Reference",
        )
        progress_axis.axhline(0.0, color="0.45", linewidth=0.55)
        progress_axis.axhline(
            trial.line_length_m,
            color="0.45",
            linewidth=0.55,
        )
        progress_axis.set_ylim(-0.13, 0.68)
        progress_axis.set_ylabel(
            f"{trial.panel_label} {trial.label}\nAlong-track (m)"
        )

        error_axis.plot(
            frame["time_from_go_s"],
            100.0 * frame["cross_track_m"],
            color="#0072B2",
            label="Cross-track",
        )
        error_axis.plot(
            frame["time_from_go_s"],
            100.0 * frame["depth_error_m"],
            color="#D55E00",
            label="Depth (+ down)",
        )
        error_axis.axhline(0.0, color="black", linewidth=0.55)
        error_axis.set_ylim(-error_limit, error_limit)
        error_axis.set_ylabel("Error (cm)")

        if row == 0:
            progress_axis.set_title("Fixed-hook line progress")
            error_axis.set_title("Lateral and depth tracking")
            progress_axis.legend(loc="best", frameon=False)
            error_axis.legend(loc="best", frameon=False)
        if not trial.complete:
            progress_axis.text(
                0.985,
                0.04,
                "MoCap/odom loss",
                transform=progress_axis.transAxes,
                ha="right",
                va="bottom",
                color="#D62728",
                fontsize=7.0,
            )

    axes[-1, 0].set_xlabel(r"Time since GO_FORWARD (s)")
    axes[-1, 1].set_xlabel(r"Time since GO_FORWARD (s)")
    phase_handles = [
        Patch(
            facecolor=PHASE_COLORS[phase],
            alpha=0.20,
            edgecolor="none",
            label=phase,
        )
        for phase in ("Forward", "Wait", "Retreat")
    ]
    figure.legend(
        handles=phase_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=3,
        frameon=False,
    )
    save_figure(figure, output_dir, "fig02_hook_line_tracking")


def make_attitude_and_control(trials: List[Trial], output_dir: Path) -> None:
    figure, axes = plt.subplots(
        3,
        3,
        figsize=(7.25, 6.5),
        constrained_layout=True,
    )
    maximum_attitude = max(
        float(
            trial.data.loc[
                trial.data["time_ros_s"] >= trial.go_time,
                "attitude_error_deg",
            ].max()
        )
        for trial in trials
    )
    attitude_limit = 5.0 * math.ceil((maximum_attitude + 1.0) / 5.0)
    thrust_colors = ("#0072B2", "#D55E00", "#009E73")
    torque_colors = ("#56B4E9", "#E69F00", "#CC79A7")

    for row, trial in enumerate(trials):
        frame = display_frame(
            trial.data.loc[trial.data["time_ros_s"] >= trial.go_time]
        )
        trace_duration = trial.end_time - trial.go_time
        full_attitude_gate_deg = float(
            trial.notes["prehook_planner"][
                "reached_orientation_tolerance_deg"
            ]
        )
        thrust_limit = float(trial.notes["thrust_sat"])
        torque_limit = float(trial.notes["torque_sat"])
        for axis in axes[row]:
            add_phase_background(axis, trial)
            axis.grid(True)
            axis.set_xlim(0.0, trace_duration * (1.015 if not trial.complete else 1.0))
            if not trial.complete:
                axis.axvline(
                    trace_duration,
                    color="#D62728",
                    linestyle=":",
                    linewidth=1.0,
                    zorder=5,
                )

        attitude_axis, thrust_axis, torque_axis = axes[row]
        attitude_axis.plot(
            frame["time_from_go_s"],
            frame["attitude_error_deg"],
            color=trial.color,
        )
        attitude_axis.axhline(
            full_attitude_gate_deg,
            color="#D62728",
            linestyle="--",
            linewidth=0.9,
            label=(
                rf"${full_attitude_gate_deg:g}^\circ$ full-attitude gate"
            ),
        )
        attitude_axis.set_ylim(0.0, attitude_limit)
        attitude_axis.set_ylabel(
            f"{trial.panel_label} {trial.label}\nError (deg)"
        )

        for column, color, label in zip(
            ("thrust_x", "thrust_y", "thrust_z"),
            thrust_colors,
            (r"$u_x$", r"$u_y$", r"$u_z$"),
        ):
            thrust_axis.plot(
                frame["time_from_go_s"],
                frame[column],
                color=color,
                label=label,
            )
        thrust_axis.axhline(
            thrust_limit, color="0.35", linestyle="--", linewidth=0.6
        )
        thrust_axis.axhline(
            -thrust_limit, color="0.35", linestyle="--", linewidth=0.6
        )
        thrust_axis.set_ylim(-1.1 * thrust_limit, 1.1 * thrust_limit)
        thrust_axis.set_ylabel("Normalized thrust")

        for column, color, label in zip(
            ("torque_x", "torque_y", "torque_z"),
            torque_colors,
            (r"$u_\phi$", r"$u_\theta$", r"$u_\psi$"),
        ):
            torque_axis.plot(
                frame["time_from_go_s"],
                frame[column],
                color=color,
                label=label,
            )
        torque_axis.axhline(
            torque_limit, color="0.35", linestyle="--", linewidth=0.6
        )
        torque_axis.axhline(
            -torque_limit, color="0.35", linestyle="--", linewidth=0.6
        )
        torque_axis.set_ylim(-1.1 * torque_limit, 1.1 * torque_limit)
        torque_axis.set_ylabel("Normalized torque")

        if not trial.complete:
            attitude_axis.text(
                0.985,
                0.96,
                "MoCap/odom loss",
                transform=attitude_axis.transAxes,
                ha="right",
                va="top",
                fontsize=6.8,
                color="#D62728",
            )

        if row == 0:
            attitude_axis.set_title("Full-attitude error")
            thrust_axis.set_title("Force setpoints")
            torque_axis.set_title("Torque setpoints")
            attitude_axis.legend(loc="upper right", frameon=False)
            thrust_axis.legend(loc="upper right", ncol=3, frameon=False)
            torque_axis.legend(loc="upper right", ncol=3, frameon=False)

    for axis in axes[-1]:
        axis.set_xlabel(r"Time since GO_FORWARD (s)")
    phase_handles = [
        Patch(
            facecolor=PHASE_COLORS[phase],
            alpha=0.20,
            edgecolor="none",
            label=phase,
        )
        for phase in ("Forward", "Wait", "Retreat")
    ]
    figure.legend(
        handles=phase_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=3,
        frameon=False,
    )
    save_figure(figure, output_dir, "fig03_attitude_and_control")


def annotate_bars(axis: plt.Axes, bars: Any, digits: int = 1) -> None:
    for bar in bars:
        height = float(bar.get_height())
        axis.annotate(
            f"{height:.{digits}f}",
            (bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.6,
        )


def mark_censored_bars(bars: Any, trials: List[Trial]) -> None:
    """Hatch bars whose interval ends at data loss rather than COMPLETE."""
    for bar, trial in zip(bars, trials):
        if not trial.complete:
            bar.set_hatch("////")
            bar.set_edgecolor("0.2")
            bar.set_linewidth(0.7)


def make_summary(
    trials: List[Trial],
    summaries: List[Dict[str, Any]],
    phase_rows: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.25, 5.0),
        constrained_layout=True,
    )
    names = [f"{trial.label}{'' if trial.complete else '*'}" for trial in trials]
    x_values = np.arange(len(trials), dtype=float)
    summary_by_name = {row["scenario"]: row for row in summaries}
    phase_by_key = {
        (row["scenario"], row["phase"]): row for row in phase_rows
    }

    duration_axis = axes[0, 0]
    bottom = np.zeros(len(trials), dtype=float)
    duration_keys = {
        "Approach": "approach_duration_s",
        "Forward": "forward_duration_s",
        "Wait": "wait_duration_s",
        "Retreat": "retreat_duration_s",
    }
    for phase in PHASE_ORDER:
        values = np.array(
            [summary_by_name[trial.label][duration_keys[phase]] for trial in trials]
        )
        bars = duration_axis.bar(
            x_values,
            values,
            bottom=bottom,
            color=PHASE_COLORS[phase],
            label="Operator wait" if phase == "Wait" else phase,
            width=0.64,
        )
        if phase == "Retreat":
            mark_censored_bars(bars, trials)
        bottom += values
    for index, total in enumerate(bottom):
        duration_axis.text(
            index,
            total + 5.0,
            f"{total:.0f}",
            ha="center",
            va="bottom",
            fontsize=7.0,
        )
    duration_axis.set_ylabel("Duration (s)")
    duration_axis.set_title("(a) Controller-stage duration")
    duration_axis.set_xticks(x_values, names)
    duration_axis.grid(True, axis="y")
    handles, labels = duration_axis.get_legend_handles_labels()
    handles.append(
        Patch(
            facecolor="white",
            edgecolor="0.2",
            hatch="////",
            label="Censored at loss",
        )
    )
    labels.append("Censored at loss")
    duration_axis.legend(
        handles, labels, loc="upper left", frameon=False, ncol=1
    )

    width = 0.34
    comparison_phases = ("Forward", "Retreat")
    comparison_colors = (PHASE_COLORS["Forward"], PHASE_COLORS["Retreat"])
    panels = (
        (
            axes[0, 1],
            "ekf_cumulative_path_m",
            "Cumulative 3-D EKF path (m)",
            "(b) Cumulative motion",
        ),
        (
            axes[1, 0],
            "cross_track_rmse_cm",
            "Cross-track RMSE (cm)",
            "(c) Lateral tracking",
        ),
        (
            axes[1, 1],
            "depth_rmse_cm",
            "Depth RMSE (cm)",
            "(d) Vertical tracking",
        ),
    )
    for axis, metric, ylabel, title in panels:
        for phase_index, (phase, color) in enumerate(
            zip(comparison_phases, comparison_colors)
        ):
            values = [
                phase_by_key[(trial.label, phase)][metric] for trial in trials
            ]
            bars = axis.bar(
                x_values + (phase_index - 0.5) * width,
                values,
                width=width,
                color=color,
                label=phase,
            )
            if phase == "Retreat":
                mark_censored_bars(bars, trials)
            annotate_bars(axis, bars, digits=2)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.set_xticks(x_values, names)
        axis.grid(True, axis="y")
    axes[0, 1].legend(frameon=False)
    save_figure(figure, output_dir, "fig04_comparative_metrics")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def write_metrics(
    summaries: List[Dict[str, Any]],
    phase_rows: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    pd.DataFrame(summaries).to_csv(
        output_dir / "summary_metrics.csv",
        index=False,
        float_format="%.6f",
    )
    pd.DataFrame(phase_rows).to_csv(
        output_dir / "phase_metrics.csv",
        index=False,
        float_format="%.6f",
    )
    payload = {
        "summary": summaries,
        "phases": phase_rows,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_readme(
    trials: List[Trial],
    summaries: List[Dict[str, Any]],
    output_dir: Path,
    annotations_path: Path,
) -> None:
    scenario_lines = []
    for trial, summary in zip(trials, summaries):
        status = "controller COMPLETE" if trial.complete else "MoCap/odom loss before COMPLETE"
        scenario_lines.append(
            f"- **{trial.label}** (`{trial.spec['trial_id']}`): {status}; "
            f"operator-confirmed physical success; annotated duration "
            f"{summary['annotated_mission_duration_s']:.1f} s."
        )
    readme = f"""# Paper figures: three successful payload-retrieval experiments

These figures compare the three experiment folders selected by the operator.
Use the PDF files for LaTeX/publication, SVG for editing, and 600 dpi PNG for
Word or slide review.

## Included trials

{chr(10).join(scenario_lines)}

## Figure files

- `fig01_plan_view_trajectories`: common-axis NED plan view, colored by mission phase.
- `fig02_hook_line_tracking`: measured/reference along-track position and signed cross-track/depth error.
- `fig03_attitude_and_control`: full quaternion attitude error and all normalized force/torque setpoints.
- `fig04_comparative_metrics`: stage durations, cumulative EKF motion, lateral RMSE, and depth RMSE.

Every figure is available as `.pdf`, `.svg`, and `.png`. Numerical values are
in `summary_metrics.csv`, `phase_metrics.csv`, and `metrics.json`. The exact
timestamp annotations and their controller-log evidence are copied into this
directory so the analysis package remains auditable.

## Processing rules

- Position and attitude use `/mocap/splash/odom_ekf_fixed_hook` recorded in the
  `odom_*` columns.
- Invalid odometry rows are excluded. Metrics use all valid 20 Hz samples.
- `raw_mocap_fresh_ratio` requires the logged raw pose to be valid and no older
  than the controller's configured 0.20 s safety threshold.
- Display curves are decimated to approximately {DISPLAY_RATE_HZ:.0f} Hz without
  smoothing; no experimental value is interpolated or filtered for presentation.
- Cumulative path is the point-to-point sum of all valid 20 Hz EKF positions.
  It deliberately includes physical oscillation and residual EKF variation during
  long stages; it is not straight-line displacement or a filtered efficiency metric.
- Along-track distance is the projection on the configured pre-hook-to-Hook
  horizontal NED line. Cross-track error is signed in the perpendicular NED
  direction. Positive depth error means deeper/down in NED.
- Attitude error is the shortest full-quaternion angle
  `2 acos(|q_measured dot q_target|)`.
- The reference line is reconstructed from the recorded pre-hook/Hook endpoints,
  the configured `0.09 m/s` speed, and controller-state timestamps in
  `{annotations_path.name}`.
- The Direct-front trace is stopped at the controller's odometry-stale event.
  It must not be described as a clean controller `COMPLETE`: physical retrieval
  was reported successful, but MoCap/odometry was lost before the state machine
  completed GO_BACK. The figures mark this endpoint explicitly.
- Stage timestamps were recovered from the controller logs and stored in the
  self-contained annotation file. `events.csv` contains logger start/stop only.
- There is one run per initial-position condition. Bars are descriptive trial
  values, not replicate means; therefore no confidence intervals are shown.

## Suggested paper captions

**Figure 1.** Plan-view trajectories for three fixed-hook payload-retrieval
experiments with different initial robot positions. Colors indicate the A*/NMPC
approach, final straight engagement, operator-confirmed Hook hold, and straight
retreat. All trials use the same measured Hook and pre-hook poses.

**Figure 2.** Fixed-hook line tracking from GO_FORWARD onward. The dashed line is
the time-parameterized reference; measured along-track motion and signed lateral
and depth errors are calculated from EKF-fused MoCap odometry. Shading identifies
GO_FORWARD, WAIT_HOOK, and GO_BACK.

**Figure 3.** Full-quaternion attitude tracking error and normalized NMPC
force/torque setpoints during the fixed-hook phases. The red dashed line is the
10-degree full-attitude component of the pre-hook arrival gate; the controller
also applies separate 5-degree yaw and body-forward-axis gates, which are not
plotted. Gray dashed lines are the configured command limits.

**Figure 4.** Descriptive comparison of stage duration, cumulative 20 Hz EKF
motion, cross-track RMSE, and depth RMSE for one run in each initial-position
condition. WAIT_HOOK duration includes operator observation and H-key response.
The asterisk flags the incomplete Direct-front controller trace; its hatched
retreat values are right-censored at MoCap/odometry loss rather than a clean
controller `COMPLETE`. The operator nevertheless reported physical retrieval
success.

## Reproduce

```bash
cd /home/yecheng/bluerov_ws
MPLCONFIGDIR=/tmp/bluerov_paper_mpl python3 \\
  src/bluerov2_control/scripts/plot_three_success_trials.py
```
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    trials_root = args.trials_root.expanduser().resolve()
    annotations_path = args.annotations.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    trial_specs = annotations.get("trials", [])
    if len(trial_specs) != 3:
        raise ValueError("annotations must contain exactly three trials")
    configure_plot_style()
    trials = [
        load_trial(spec, trials_root, TRIAL_COLORS[index])
        for index, spec in enumerate(trial_specs)
    ]

    phase_rows = [
        phase_metrics(trial, phase)
        for trial in trials
        for phase in ("Forward", "Wait", "Retreat")
    ]
    summaries = [summary_metrics(trial, phase_rows) for trial in trials]
    write_metrics(summaries, phase_rows, output_dir)
    make_plan_view(trials, output_dir)
    make_hook_tracking(trials, output_dir)
    make_attitude_and_control(trials, output_dir)
    make_summary(trials, summaries, phase_rows, output_dir)
    write_readme(trials, summaries, output_dir, annotations_path)

    shutil.copy2(annotations_path, output_dir / annotations_path.name)
    evidence_name = annotations.get("timestamp_provenance", {}).get(
        "evidence_file"
    )
    if evidence_name:
        evidence_path = annotations_path.with_name(str(evidence_name))
        if not evidence_path.is_file():
            raise FileNotFoundError(f"controller evidence file missing: {evidence_path}")
        shutil.copy2(evidence_path, output_dir / evidence_path.name)

    with (output_dir / "generation_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        writer = csv.writer(destination)
        writer.writerow(
            [
                "scenario",
                "trial_directory",
                "controller_log",
                "controller_log_sha256",
            ]
        )
        for trial in trials:
            writer.writerow(
                [
                    trial.label,
                    trial.spec["directory"],
                    trial.spec["controller_log"],
                    trial.spec["controller_log_sha256"],
                ]
            )
    print(f"Wrote paper figures and metrics to {output_dir}")


if __name__ == "__main__":
    main()
