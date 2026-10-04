"""Check the fixed-target restoration against the recorded August 31 trial.

Only launch descriptions and message classes are loaded. These tests never
start ROS nodes, solve an OCP, arm a vehicle, or publish commands.
"""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

from bluerov2_control.research.trial_data_logger import SAMPLE_FIELDS

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.utilities import (
    normalize_to_list_of_substitutions,
    perform_substitutions,
)

from launch_ros.actions import Node as LaunchNode
from launch_ros.utilities import evaluate_parameters

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TRIAL_ROOT = (
    PACKAGE_ROOT.parents[1]
    / 'bluerov2_payload_retrieval_trials'
    / 'retrieval_directly_in_front_of_the_box_success'
)
CONFIG_ROOT = PACKAGE_ROOT / 'experiments' / 'payload_retrieval' / 'config'


def _load_launch(filename):
    spec = importlib.util.spec_from_file_location(
        'restored_baseline_' + filename.replace('.', '_'),
        PACKAGE_ROOT / 'launch' / filename,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def baseline():
    """Read preserved experimental evidence, without rewriting any artifact."""
    metadata_path = TRIAL_ROOT / 'metadata.json'
    if not metadata_path.is_file():
        pytest.skip('The archived August 31 trial is required for this audit')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    return metadata, json.loads(metadata['notes'])


@pytest.fixture(scope='module')
def resolved_splash(baseline):
    """Resolve the actual Splash include and common defaults offline."""
    _, recorded_notes = baseline
    context = LaunchContext()
    context.launch_configurations['target_config'] = str(
        CONFIG_ROOT / Path(recorded_notes['target_config']).name
    )
    wrapper = _load_launch('fixed_hook_pose_validation_splash.launch.py')
    description = wrapper.generate_launch_description()
    for action in description.entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    include = next(
        action for action in description.entities
        if isinstance(action, IncludeLaunchDescription)
    )
    for name, value in vars(include)[
        '_IncludeLaunchDescription__launch_arguments'
    ]:
        context.launch_configurations[name] = perform_substitutions(
            context, normalize_to_list_of_substitutions(value)
        )
    common = _load_launch('fixed_hook_pose_validation.launch.py')
    for action in common.generate_launch_description().entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    parameters = {}
    for action in common._launch_setup(context):
        if isinstance(action, LaunchNode):
            executable = vars(action)['_Node__node_executable']
            parameters[executable] = evaluate_parameters(
                context, vars(action)['_Node__parameters']
            )[0]
    return parameters


def _assert_recorded_values(actual, expected, key='notes'):
    """Compare control settings, allowing roundoff and path relocation."""
    if isinstance(expected, dict):
        # The ROS overlay installation location is host-specific, not a
        # controller setting. Message versions remain part of the comparison.
        keys = set(expected) - {'px4_msgs_prefix'}
        assert set(actual) - {'px4_msgs_prefix'} == keys, key
        for child in keys:
            _assert_recorded_values(actual[child], expected[child], child)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), key
        for index, (left, right) in enumerate(zip(actual, expected)):
            _assert_recorded_values(left, right, f'{key}[{index}]')
    elif isinstance(expected, bool) or expected is None:
        assert actual is expected, key
    elif isinstance(expected, (float, int)):
        assert actual == pytest.approx(expected, rel=1e-10, abs=1e-12), key
    elif key in {'target_config', 'source_config', 'manifest_config'}:
        assert Path(actual).name == Path(expected).name, key
    else:
        assert actual == expected, key


def test_splash_settings_match_recorded_success(baseline, resolved_splash):
    """Preserve all saved A*, NMPC, geometry, H-key and safety settings."""
    _, expected = baseline
    actual = json.loads(
        resolved_splash['payload_retrieval_data_logger']['notes']
    )
    _assert_recorded_values(actual, expected)


def test_splash_target_still_matches_hash_locked_recording(baseline):
    """Retain the measured target and name-only transfer provenance."""
    metadata, notes = baseline
    provenance = notes['target_provenance']
    original = CONFIG_ROOT / Path(provenance['source_config']).name
    assert hashlib.sha256(original.read_bytes()).hexdigest() == (
        provenance['source_config_sha256']
    )
    recorded_pose = metadata['external_metadata']['target_pose']
    original_payload = json.loads(original.read_text(encoding='utf-8'))
    assert original_payload['target_pose'] == recorded_pose
    manifest = json.loads(
        (CONFIG_ROOT / Path(notes['target_config']).name).read_text(
            encoding='utf-8'
        )
    )
    assert manifest['target_pose'] == recorded_pose
    assert manifest['derived_target']['destination_pose_topic'] == (
        '/mocap/splash/pose'
    )
    assert manifest['derived_target']['source_config_sha256'] == (
        provenance['source_config_sha256']
    )


def test_fixed_target_pipeline_remains_independent(resolved_splash):
    """Keep the five-node fixed-target launch, without camera/YOLO nodes."""
    assert set(resolved_splash) == {
        'mocap_ekf_odom',
        'nav_odom_to_vehicle_odometry',
        'mpc_track_trajectory_acados',
        'payload_retrieval_data_logger',
        'offboard_enable',
    }
    mpc = resolved_splash['mpc_track_trajectory_acados']
    assert mpc['model_type'] == 'fossen_real'
    assert mpc['solve_rate_hz'] == 25.0
    assert mpc['planner_mode'] == 'astar'
    assert mpc['use_dynamic_prehook_planner'] is True
    assert mpc['prehook_planner_check_rate_hz'] == 2.0
    assert mpc['fixed_hook_line_position_mode'] is True
    assert mpc['hold_attitude'] is True
    assert mpc['require_operator_hook_confirmation'] is True
    assert mpc['return_to_pre_approach_after_hold'] is True
    assert mpc['final_approach_speed_mps'] == 0.09
    assert mpc['backward_pass_speed_mps'] == 0.09
    assert mpc['pre_approach_z'] == mpc['goal_z']
    for parameters in resolved_splash.values():
        assert 'mission_state_topic' not in parameters


def test_restored_logger_schema_matches_success_csv(baseline):
    """Keep paper-analysis input columns identical to the successful trial."""
    with (TRIAL_ROOT / 'samples.csv').open(
        newline='', encoding='utf-8'
    ) as stream:
        original_columns = next(csv.reader(stream))
    assert list(SAMPLE_FIELDS) == original_columns
