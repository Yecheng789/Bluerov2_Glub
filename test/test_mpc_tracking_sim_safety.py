"""Regression tests for controller feedback freshness and Gazebo startup."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from bluerov2_control import mpc_track_trajectory_acados as tracker_module
from bluerov2_control.mpc_track_trajectory_acados import (
    DEFAULT_CONTROL_MODE_TIMEOUT_SEC,
    MPCTrackTrajectoryAcados,
)


class _Logger:
    def __init__(self):
        self.warnings = []

    def warn(self, message, **_kwargs):
        self.warnings.append(message)


def _load_sim_launch():
    path = (
        Path(__file__).resolve().parents[1]
        / 'launch'
        / 'mpc_track_trajectory_acados_sim.launch.py'
    )
    spec = importlib.util.spec_from_file_location('sim_launch', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_tracker_timeout_tolerates_px4_two_hz_status_jitter():
    logger = _Logger()
    fake = SimpleNamespace(
        last_control_mode_monotonic=100.0,
        get_parameter=lambda _name: SimpleNamespace(
            value=DEFAULT_CONTROL_MODE_TIMEOUT_SEC
        ),
        get_logger=lambda: logger,
    )

    assert DEFAULT_CONTROL_MODE_TIMEOUT_SEC == pytest.approx(1.25)
    assert MPCTrackTrajectoryAcados._control_mode_feedback_fresh(
        fake,
        now_monotonic=101.25,
    )
    assert not MPCTrackTrajectoryAcados._control_mode_feedback_fresh(
        fake,
        now_monotonic=101.251,
    )
    assert len(logger.warnings) == 1


def test_tracker_declares_jitter_tolerant_timeout_as_default():
    source_path = (
        Path(__file__).resolve().parents[1]
        / 'bluerov2_control'
        / 'mpc_track_trajectory_acados.py'
    )
    tree = ast.parse(source_path.read_text(encoding='utf-8'))
    matching_calls = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'declare_parameter'
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == 'control_mode_timeout_s'
        )
    ]
    assert len(matching_calls) == 1
    assert isinstance(matching_calls[0].args[1], ast.Name)
    assert matching_calls[0].args[1].id == (
        'DEFAULT_CONTROL_MODE_TIMEOUT_SEC'
    )


def test_control_mode_callback_records_monotonic_receive_time(monkeypatch):
    monkeypatch.setattr(tracker_module.time, 'monotonic', lambda: 123.0)
    fake = SimpleNamespace(enabled=True)
    message = SimpleNamespace(
        flag_armed=True,
        flag_control_offboard_enabled=True,
    )
    MPCTrackTrajectoryAcados.on_control_mode(fake, message)
    assert fake.last_control_mode_monotonic == pytest.approx(123.0)


def test_sim_launch_uses_one_position_controller_and_no_wrench_publisher(
    monkeypatch,
):
    module = _load_sim_launch()
    monkeypatch.setattr(module, 'Node', lambda **kwargs: kwargs)
    monkeypatch.setattr(
        module,
        'LaunchConfiguration',
        lambda name: SimpleNamespace(perform=lambda context: context[name]),
    )

    nodes = module._launch_setup({'px4_namespace': ''})
    assert len(nodes) == 1
    controller = nodes[0]
    assert controller['executable'] == 'position_payload_retrieval'
    parameters = controller['parameters'][0]
    assert parameters['require_mission_enable'] is False
    assert parameters['control_mode_timeout_s'] == pytest.approx(1.25)
    assert parameters['hook_alignment_raise_m'] == pytest.approx(0.04)
    assert parameters['align_hold_lower_m'] == pytest.approx(0.01)
    assert parameters['align_speed_mps'] == pytest.approx(0.06)
    assert parameters['descent_speed_mps'] == pytest.approx(0.06)
    assert parameters['forward_pass_speed_mps'] == pytest.approx(0.05)
    assert parameters['backward_pass_speed_mps'] == pytest.approx(0.06)
    assert parameters['yaw_reference_rate_rad_s'] == pytest.approx(0.10)
    assert parameters['attitude_setpoint_topic'] == (
        '/fmu/in/vehicle_attitude_setpoint_v1'
    )
    assert parameters['offboard_control_mode_topic'] == (
        '/fmu/in/offboard_control_mode'
    )
    assert parameters['rates_setpoint_topic'] == (
        '/fmu/in/vehicle_rates_setpoint'
    )
    assert not any(
        'thrust_sp_topic' in item or 'torque_sp_topic' in item
        for item in parameters
    )


def test_sim_launch_applies_namespace_to_every_px4_topic(monkeypatch):
    module = _load_sim_launch()
    monkeypatch.setattr(module, 'Node', lambda **kwargs: kwargs)
    monkeypatch.setattr(
        module,
        'LaunchConfiguration',
        lambda name: SimpleNamespace(perform=lambda context: context[name]),
    )
    controller = module._launch_setup({
        'px4_namespace': '/itrl_rov_1/',
    })[0]
    parameters = controller['parameters'][0]
    for name in (
        'odom_topic',
        'control_mode_topic',
        'attitude_setpoint_topic',
        'offboard_control_mode_topic',
        'rates_setpoint_topic',
    ):
        assert parameters[name].startswith('/itrl_rov_1/fmu/')
