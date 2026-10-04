"""Regression tests for the current Glub Qualisys rigid-body name."""

import ast
import importlib.util
from pathlib import Path

from bluerov2_control.calibrate_mocap_orientation_correction import (
    parse_args as parse_calibration_args,
)
from launch import LaunchContext


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _declared_parameter_defaults(relative_path):
    tree = ast.parse(
        (PACKAGE_ROOT / relative_path).read_text(encoding='utf-8')
    )
    defaults = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not (
            isinstance(function, ast.Attribute)
            and function.attr == 'declare_parameter'
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Constant)
        ):
            continue
        try:
            defaults[node.args[0].value] = ast.literal_eval(node.args[1])
        except (TypeError, ValueError):
            continue
    return defaults


def _load_launch(relative_path, module_name):
    path = PACKAGE_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_standalone_mocap_ekf_defaults_to_current_glub_rigid_body():
    defaults = _declared_parameter_defaults(
        'bluerov2_control/mocap_ekf_odom.py'
    )
    assert defaults['rigid_body_name'] == 'glub'

    module = _load_launch(
        'launch/mocap_ekf_odom.launch.py',
        'mocap_ekf_odom_launch_glub_default',
    )
    context = LaunchContext()
    for entity in module.generate_launch_description().entities:
        if getattr(entity, 'name', None) is not None:
            entity.execute(context)
    assert context.launch_configurations['rigid_body_name'] == 'glub'


def test_standalone_odom_adapter_defaults_follow_current_glub_topics():
    defaults = _declared_parameter_defaults(
        'bluerov2_control/nav_odom_to_vehicle_odometry.py'
    )
    assert defaults['input_odom_topic'] == '/mocap/glub/odom_ekf'
    assert defaults['output_vehicle_odometry_topic'] == (
        '/mocap/glub/vehicle_odometry_ekf'
    )


def test_orientation_calibration_defaults_to_current_raw_glub_topic():
    args, ros_args = parse_calibration_args([])
    assert args.topic == '/mocap/glub/pose'
    assert ros_args == []
