#!/usr/bin/env python3
"""Run the guarded real-pool fixed-hook target-pose validation."""

import json
import math
import os
from pathlib import Path

from ament_index_python.packages import get_package_prefix
from bluerov2_control.nav_odom_to_vehicle_odometry import (
    _axis_transform,
    _matrix_to_quat_wxyz,
    _quat_xyzw_to_matrix,
)
from bluerov2_control.offboard_enable import versioned_px4_topic
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import numpy as np
from px4_msgs.msg import VehicleCommandAck, VehicleStatus


MISSION_ENABLE_TOPIC = '/bluerov2/fixed_hook/mission_enable'
OFFBOARD_REQUEST_TOPIC = '/bluerov2/fixed_hook/offboard_request_enable'
CONTROLLER_HEARTBEAT_TOPIC = '/bluerov2/fixed_hook/controller_heartbeat'
TRIAL_EVENT_TOPIC = '/bluerov2/trial_event'
EXPECTED_ACK_MESSAGE_VERSION = 0
EXPECTED_STATUS_MESSAGE_VERSION = 1


def _argument(context, name):
    return LaunchConfiguration(name).perform(context).strip()


def _parse_bool(context, name):
    text = _argument(context, name).lower()
    if text == 'true':
        return True
    if text == 'false':
        return False
    raise RuntimeError(f'{name} must be either true or false')


def _parse_float(context, name, minimum=None, maximum=None):
    text = _argument(context, name)
    try:
        value = float(text)
    except ValueError as exc:
        raise RuntimeError(f'{name} must be a number') from exc
    if not math.isfinite(value):
        raise RuntimeError(f'{name} must be finite')
    if minimum is not None and value < minimum:
        raise RuntimeError(f'{name} must be >= {minimum}')
    if maximum is not None and value > maximum:
        raise RuntimeError(f'{name} must be <= {maximum}')
    return value


def _parse_int(context, name, minimum, maximum):
    text = _argument(context, name)
    if text.lower() == 'unconfigured':
        raise RuntimeError(
            f'{name} is unconfigured; read it from the version-matched '
            'PX4 VehicleStatus topic'
        )
    try:
        value = int(text)
    except ValueError as exc:
        raise RuntimeError(f'{name} must be an integer') from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(
            f'{name} must be in [{minimum}, {maximum}]'
        )
    return value


def _parse_choice(context, name, choices):
    value = _argument(context, name).lower()
    if value not in choices:
        expected = ', '.join(sorted(choices))
        raise RuntimeError(
            f'{name}={value!r} is not configured; choose one of: {expected}'
        )
    return value


def _validate_firmware_matched_px4_msgs():
    """Reject overlays not synchronized with the current real PX4 firmware."""
    ack_version = int(getattr(VehicleCommandAck, 'MESSAGE_VERSION', 0))
    status_version = int(getattr(VehicleStatus, 'MESSAGE_VERSION', 0))
    if (
        ack_version != EXPECTED_ACK_MESSAGE_VERSION
        or status_version != EXPECTED_STATUS_MESSAGE_VERSION
    ):
        prefix = get_package_prefix('px4_msgs')
        raise RuntimeError(
            'px4_msgs does not match the current real PX4 firmware: '
            f'active prefix={prefix}, ACK/Status versions='
            f'{ack_version}/{status_version}, expected '
            f'{EXPECTED_ACK_MESSAGE_VERSION}/'
            f'{EXPECTED_STATUS_MESSAGE_VERSION}. Source '
            'scripts/source_fixed_hook_real.bash and rebuild px4_msgs.'
        )
    return get_package_prefix('px4_msgs'), ack_version, status_version


def _normalise_namespace(context):
    namespace = _argument(context, 'robot_namespace')
    if not namespace:
        raise RuntimeError('robot_namespace must not be empty')
    parts = namespace.strip('/').split('/')
    if not all(part and part.replace('_', '').isalnum() for part in parts):
        raise RuntimeError('robot_namespace contains an invalid ROS name')
    return '/' + '/'.join(parts)


def _validate_rigid_body_name(context):
    name = _argument(context, 'rigid_body_name')
    if not name or not name.replace('_', '').isalnum():
        raise RuntimeError('rigid_body_name must be one ROS name component')
    return name


def _strict_quaternion_xyzw(text, argument_name):
    text = str(text).strip()
    if not text:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    try:
        values = np.array(
            [float(value) for value in text.replace(',', ' ').split()],
            dtype=float,
        )
    except ValueError as exc:
        raise RuntimeError(
            f'{argument_name} must contain four numeric values'
        ) from exc
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise RuntimeError(
            f'{argument_name} must contain four finite values: x y z w'
        )
    norm = float(np.linalg.norm(values))
    if norm <= 1e-6:
        raise RuntimeError(f'{argument_name} has a near-zero norm')
    values /= norm
    if values[3] < 0.0:
        values = -values
    return values


def _pool_bounds_to_ned(bounds_text, world_frame, safety_margin_m):
    """Convert a raw MoCap AABB into full and margin-shrunk NED bounds."""
    text = str(bounds_text).strip()
    if not text or text.lower() == 'unconfigured':
        raise RuntimeError(
            'pool_bounds_mocap is required in the raw MoCap world frame; '
            'provide: xmin xmax ymin ymax zmin zmax'
        )
    try:
        values = np.asarray(
            [float(value) for value in text.replace(',', ' ').split()],
            dtype=float,
        )
    except ValueError as exc:
        raise RuntimeError(
            'pool_bounds_mocap must contain six numeric values: '
            'xmin xmax ymin ymax zmin zmax'
        ) from exc
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise RuntimeError(
            'pool_bounds_mocap must contain six finite values: '
            'xmin xmax ymin ymax zmin zmax'
        )

    raw_min = values[[0, 2, 4]]
    raw_max = values[[1, 3, 5]]
    if not np.all(raw_min < raw_max):
        raise RuntimeError(
            'pool_bounds_mocap requires xmin < xmax, ymin < ymax, and '
            'zmin < zmax'
        )

    try:
        margin = float(safety_margin_m)
    except (TypeError, ValueError) as exc:
        raise RuntimeError('pool_safety_margin_m must be a number') from exc
    if not math.isfinite(margin) or margin < 0.0:
        raise RuntimeError('pool_safety_margin_m must be finite and >= 0')

    # Convert all eight corners rather than assuming which raw MoCap axis maps
    # to NED north/east/down or whether an axis changes sign.
    raw_corners = np.asarray(
        [
            [x, y, z]
            for x in (raw_min[0], raw_max[0])
            for y in (raw_min[1], raw_max[1])
            for z in (raw_min[2], raw_max[2])
        ],
        dtype=float,
    )
    world_to_ned = _axis_transform(world_frame)
    ned_corners = (world_to_ned @ raw_corners.T).T
    ned_min = np.min(ned_corners, axis=0)
    ned_max = np.max(ned_corners, axis=0)
    operating_min = ned_min + margin
    operating_max = ned_max - margin
    if not np.all(operating_min < operating_max):
        raise RuntimeError(
            f'pool_safety_margin_m={margin} leaves no valid NED operating '
            f'volume inside converted bounds min={ned_min.tolist()}, '
            f'max={ned_max.tolist()}'
        )

    return {
        'raw_min': raw_min,
        'raw_max': raw_max,
        'ned_min': ned_min,
        'ned_max': ned_max,
        'operating_min_ned': operating_min,
        'operating_max_ned': operating_max,
        'safety_margin_m': margin,
    }


def _require_point_inside_bounds(point, minimum, maximum, label):
    point = np.asarray(point, dtype=float)
    minimum = np.asarray(minimum, dtype=float)
    maximum = np.asarray(maximum, dtype=float)
    if (
        point.shape != (3,)
        or minimum.shape != (3,)
        or maximum.shape != (3,)
        or not np.all(np.isfinite(point))
        or not np.all(np.isfinite(minimum))
        or not np.all(np.isfinite(maximum))
        or not np.all(minimum < maximum)
    ):
        raise RuntimeError(f'{label} or its bounds are invalid')
    if np.any(point < minimum) or np.any(point > maximum):
        raise RuntimeError(
            f'{label}={point.tolist()} is outside the margin-shrunk NED '
            f'operating bounds min={minimum.tolist()}, max={maximum.tolist()}'
        )


def _finite_json_number(value, field_name):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f'target JSON field {field_name} is not numeric') from exc
    if not math.isfinite(number):
        raise RuntimeError(f'target JSON field {field_name} is not finite')
    return number


def _load_target_payload(path):
    try:
        with path.open('r', encoding='utf-8') as source:
            payload = json.load(source)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f'cannot read target_config={path}: {exc}') from exc
    if not isinstance(payload, dict):
        raise RuntimeError('target_config must contain a JSON object')
    return payload


def _validate_recording(payload, context):
    validation = payload.get('validation')
    if not isinstance(validation, dict) or validation.get('passed') is not True:
        raise RuntimeError(
            'target_config was not produced by a successful validated target '
            'recording; record a new file with record_mocap_target_pose'
        )
    metrics = validation.get('metrics')
    if not isinstance(metrics, dict):
        raise RuntimeError('target_config validation.metrics is missing')
    settings = validation.get('settings')
    if not isinstance(settings, dict):
        raise RuntimeError('target_config validation.settings is missing')

    live_max_message_age = _parse_float(
        context,
        'max_raw_mocap_message_age_sec',
        minimum=0.01,
        maximum=1.0,
    )
    recording_max_message_age = _finite_json_number(
        settings.get('max_message_age_sec'),
        'validation.settings.max_message_age_sec',
    )
    if (
        recording_max_message_age <= 0.0
        or recording_max_message_age > live_max_message_age
    ):
        raise RuntimeError(
            'target recorder max message age '
            f'{recording_max_message_age:.3f}s is incompatible with the '
            f'live EKF limit {live_max_message_age:.3f}s; record a new '
            'target with the same or stricter limit'
        )

    message_age = metrics.get('message_age_sec')
    if not isinstance(message_age, dict):
        raise RuntimeError('target message-age validation metrics are missing')
    recorded_max_age = _finite_json_number(
        message_age.get('max'),
        'validation.metrics.message_age_sec.max',
    )
    if recorded_max_age > live_max_message_age:
        raise RuntimeError(
            f'target samples reached {recorded_max_age:.3f}s age, above '
            f'the live EKF limit {live_max_message_age:.3f}s'
        )

    samples = int(metrics.get('accepted_sample_count', -1))
    min_samples = _parse_int(context, 'target_min_samples', 2, 100000)
    if samples < min_samples:
        raise RuntimeError(
            f'target recording has {samples} samples; at least '
            f'{min_samples} are required'
        )

    timestamped = int(metrics.get('header_timestamped_sample_count', -1))
    distinct = int(metrics.get('distinct_header_timestamp_count', -1))
    if timestamped != samples or distinct != samples:
        raise RuntimeError(
            'every target sample must have a distinct, non-zero header stamp'
        )

    span = _finite_json_number(
        metrics.get('sampling_span_sec'),
        'validation.metrics.sampling_span_sec',
    )
    min_span = _parse_float(
        context,
        'target_min_sampling_span_sec',
        minimum=0.1,
    )
    if span < min_span:
        raise RuntimeError(
            f'target recording span {span:.3f}s is below {min_span:.3f}s'
        )

    position_std = _finite_json_number(
        metrics.get('max_axis_position_standard_deviation_m'),
        'validation.metrics.max_axis_position_standard_deviation_m',
    )
    max_position_std = _parse_float(
        context,
        'target_max_position_std_m',
        minimum=0.0,
        maximum=0.05,
    )
    if position_std > max_position_std:
        raise RuntimeError(
            f'target position std {position_std:.4f}m exceeds '
            f'{max_position_std:.4f}m'
        )

    orientation = metrics.get('orientation')
    if not isinstance(orientation, dict):
        raise RuntimeError('target orientation validation metrics are missing')
    angular_std = _finite_json_number(
        orientation.get('angular_std_deg'),
        'validation.metrics.orientation.angular_std_deg',
    )
    spread = _finite_json_number(
        orientation.get('pairwise_spread_deg'),
        'validation.metrics.orientation.pairwise_spread_deg',
    )
    max_angular_std = _parse_float(
        context,
        'target_max_orientation_std_deg',
        minimum=0.0,
        maximum=10.0,
    )
    max_spread = _parse_float(
        context,
        'target_max_orientation_spread_deg',
        minimum=0.0,
        maximum=20.0,
    )
    if angular_std > max_angular_std or spread > max_spread:
        raise RuntimeError(
            'target orientation is not stable enough: '
            f'std={angular_std:.3f}deg, spread={spread:.3f}deg'
        )
    return metrics


def _target_pose_in_ned_frd(
    payload,
    world_frame,
    body_frame,
    correction_xyzw,
    correction_mode,
):
    target = payload.get('target_pose')
    if not isinstance(target, dict):
        raise RuntimeError('target_config.target_pose is missing')
    position = target.get('position')
    orientation = target.get('orientation_xyzw')
    if not isinstance(position, dict) or not isinstance(orientation, dict):
        raise RuntimeError('target_config target position/orientation is missing')

    position_raw = np.array(
        [
            _finite_json_number(position.get('x'), 'target_pose.position.x'),
            _finite_json_number(position.get('y'), 'target_pose.position.y'),
            _finite_json_number(position.get('z'), 'target_pose.position.z'),
        ],
        dtype=float,
    )
    quat_raw = _strict_quaternion_xyzw(
        ' '.join(
            str(orientation.get(axis)) for axis in ('x', 'y', 'z', 'w')
        ),
        'target_pose.orientation_xyzw',
    )

    message_type = str(payload.get('message_type', '')).strip()
    if correction_mode == 'auto':
        if message_type == 'geometry_msgs/PoseStamped':
            apply_correction = True
        elif message_type == 'nav_msgs/Odometry':
            apply_correction = False
        else:
            raise RuntimeError(
                'target_config message_type is unknown; set '
                'target_orientation_correction_mode explicitly'
            )
    else:
        apply_correction = correction_mode == 'apply'

    rotation_raw = _quat_xyzw_to_matrix(quat_raw)
    rotation_correction = _quat_xyzw_to_matrix(correction_xyzw)
    if rotation_raw is None or rotation_correction is None:
        raise RuntimeError('target or correction quaternion is invalid')
    rotation_corrected = rotation_raw
    if apply_correction:
        rotation_corrected = rotation_raw @ rotation_correction

    world_to_ned = _axis_transform(world_frame)
    body_to_frd = _axis_transform(body_frame, body=True)
    position_ned = world_to_ned @ position_raw
    rotation_ned_frd = (
        world_to_ned @ rotation_corrected @ body_to_frd.T
    )
    quat_wxyz = np.array(
        _matrix_to_quat_wxyz(rotation_ned_frd),
        dtype=float,
    )
    if not np.all(np.isfinite(quat_wxyz)):
        raise RuntimeError('target frame conversion produced an invalid pose')
    return position_ned, quat_wxyz, apply_correction


def _quat_wxyz_to_rpy(quat):
    w, x, y, z = [float(value) for value in quat]
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll, cos_roll)
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sin_yaw, cos_yaw)
    return roll, pitch, yaw


def _body_forward_pre_approach_position(
    target_position,
    goal_yaw,
    distance_m,
):
    """Place pre-hook behind the recorded horizontal body-forward axis."""
    target = np.asarray(target_position, dtype=float)
    yaw = float(goal_yaw)
    distance = float(distance_m)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise RuntimeError('target position must contain three finite values')
    if not math.isfinite(yaw):
        raise RuntimeError('target yaw must be finite')
    if not math.isfinite(distance) or distance <= 0.0:
        raise RuntimeError('pre-approach distance must be finite and > 0')

    horizontal_forward = np.array(
        [math.cos(yaw), math.sin(yaw), 0.0],
        dtype=float,
    )
    pre_approach = target - distance * horizontal_forward
    # Keep the full recorded attitude while commanding a level translation.
    pre_approach[2] = target[2]
    return pre_approach


def _critical_exit_handler(node, label):
    return RegisterEventHandler(
        OnProcessExit(
            target_action=node,
            on_exit=[
                EmitEvent(
                    event=Shutdown(
                        reason=f'critical fixed-hook process exited: {label}'
                    )
                )
            ],
        )
    )


def _launch_setup(context, *args, **kwargs):
    del args, kwargs
    (
        px4_msgs_prefix,
        ack_message_version,
        status_message_version,
    ) = _validate_firmware_matched_px4_msgs()
    namespace = _normalise_namespace(context)
    rigid_body_name = _validate_rigid_body_name(context)
    world_frame = _parse_choice(
        context, 'mocap_world_frame', {'ned', 'nwu', 'enu'}
    )
    pool_safety_margin = _parse_float(
        context, 'pool_safety_margin_m', minimum=0.0
    )
    pool_bounds = _pool_bounds_to_ned(
        _argument(context, 'pool_bounds_mocap'),
        world_frame,
        pool_safety_margin,
    )
    body_frame = _parse_choice(
        context, 'mocap_body_frame', {'frd', 'flu'}
    )
    robot_type = _parse_choice(
        context, 'robot_type', {'standard', 'heavy_tube'}
    )
    correction_mode = _parse_choice(
        context,
        'target_orientation_correction_mode',
        {'auto', 'apply', 'skip'},
    )
    correction_text = _argument(
        context, 'orientation_correction_quat_xyzw'
    )
    correction_xyzw = _strict_quaternion_xyzw(
        correction_text,
        'orientation_correction_quat_xyzw',
    )
    if (
        correction_mode == 'skip'
        and not np.allclose(
            correction_xyzw,
            np.array([0.0, 0.0, 0.0, 1.0]),
            atol=1e-9,
        )
    ):
        raise RuntimeError(
            'target_orientation_correction_mode=skip conflicts with a '
            'non-identity live EKF correction; use auto/apply so target and '
            'live orientation follow the same transform'
        )

    target_text = _argument(context, 'target_config')
    if not target_text:
        raise RuntimeError(
            'target_config is required; manually hook the target and record a '
            'new validated MoCap pose before launching this experiment'
        )
    target_path = Path(target_text).expanduser().resolve()
    payload = _load_target_payload(target_path)
    raw_pose_topic = f'/mocap/{rigid_body_name}/pose'
    if payload.get('message_type') != 'geometry_msgs/PoseStamped':
        raise RuntimeError(
            'fixed-hook reality validation requires a target recorded '
            'directly from geometry_msgs/PoseStamped, not filtered odometry'
        )
    if str(payload.get('source_topic', '')).strip() != raw_pose_topic:
        raise RuntimeError(
            f'target source_topic must be {raw_pose_topic}; record a new raw '
            'MoCap target for this rigid body'
        )
    target_frame_id = str(payload.get('frame_id', '')).strip()
    if not target_frame_id:
        raise RuntimeError(
            'target recording has an empty frame_id; fix the MoCap publisher '
            'before recording the real target'
        )
    metrics = _validate_recording(payload, context)
    target_position, target_quat, target_correction_applied = (
        _target_pose_in_ned_frd(
            payload,
            world_frame,
            body_frame,
            correction_xyzw,
            correction_mode,
        )
    )
    _require_point_inside_bounds(
        target_position,
        pool_bounds['operating_min_ned'],
        pool_bounds['operating_max_ned'],
        'converted target position',
    )
    target_down_alignment = 1.0 - 2.0 * (
        target_quat[1] * target_quat[1]
        + target_quat[2] * target_quat[2]
    )
    target_tilt_rad = math.acos(
        max(-1.0, min(1.0, float(target_down_alignment)))
    )
    if target_tilt_rad > 0.55:
        raise RuntimeError(
            f'converted target tilt is {target_tilt_rad:.3f}rad, above '
            '0.55rad; the declared world/body axes or orientation correction '
            'is inconsistent with a normally upright BlueROV'
        )
    goal_roll, goal_pitch, goal_yaw = _quat_wxyz_to_rpy(target_quat)

    traj_speed = _parse_float(
        context, 'traj_speed_mps', minimum=0.005, maximum=0.10
    )
    pre_approach_speed = _parse_float(
        context,
        'pre_approach_speed_mps',
        minimum=0.005,
        maximum=0.10,
    )
    final_approach_speed = _parse_float(
        context,
        'final_approach_speed_mps',
        minimum=0.005,
        maximum=0.10,
    )
    min_traj_duration = _parse_float(
        context, 'min_traj_duration_s', minimum=2.0, maximum=60.0
    )
    pre_approach_distance = _parse_float(
        context,
        'pre_approach_distance_m',
        minimum=0.05,
        maximum=2.0,
    )
    fixed_hook_depth_tolerance = _parse_float(
        context,
        'fixed_hook_depth_tolerance_m',
        minimum=0.005,
        maximum=0.05,
    )
    traj_angular_speed_deg = _parse_float(
        context,
        'traj_angular_speed_deg_s',
        minimum=1.0,
        maximum=90.0,
    )
    pre_approach_position = _body_forward_pre_approach_position(
        target_position,
        goal_yaw,
        pre_approach_distance,
    )
    pre_approach_line_distance = float(
        np.linalg.norm(pre_approach_position - target_position)
    )
    if not math.isclose(
        pre_approach_line_distance,
        pre_approach_distance,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            'body-forward pre-approach construction produced an unexpected '
            'line length'
        )
    _require_point_inside_bounds(
        pre_approach_position,
        pool_bounds['operating_min_ned'],
        pool_bounds['operating_max_ned'],
        'converted pre-approach position',
    )
    thrust_sat = _parse_float(
        context, 'thrust_sat', minimum=0.0, maximum=0.20
    )
    torque_sat = _parse_float(
        context, 'torque_sat', minimum=0.0, maximum=0.15
    )
    final_pose_hold_s = _parse_float(
        context, 'final_pose_hold_s', minimum=0.0, maximum=60.0
    )
    retreat_speed_mps = _parse_float(
        context, 'retreat_speed_mps', minimum=0.005, maximum=0.10
    )
    px4_angular_velocity_timeout_sec = _parse_float(
        context,
        'px4_angular_velocity_timeout_sec',
        minimum=0.03,
        maximum=0.30,
    )
    w_att = _parse_float(context, 'w_att', minimum=0.0, maximum=100.0)
    w_omega = _parse_float(
        context, 'w_omega', minimum=0.0, maximum=100.0
    )
    w_u_torque = _parse_float(
        context, 'w_u_torque', minimum=0.001, maximum=10.0
    )
    position_integral_gain = _parse_float(
        context,
        'position_integral_gain_N_per_m_s',
        minimum=0.0,
        maximum=10.0,
    )
    position_integral_force_limit_fraction = _parse_float(
        context,
        'position_integral_force_limit_fraction',
        minimum=0.0,
        maximum=0.20,
    )
    position_integral_activation_error = _parse_float(
        context,
        'position_integral_activation_error_m',
        minimum=0.05,
        maximum=1.0,
    )
    if robot_type == 'standard' and thrust_sat < 0.075:
        raise RuntimeError(
            'fossen_real standard assumes about 9.82 N positive buoyancy; '
            'thrust_sat must be >= 0.075 to make depth hold physically '
            'possible. Confirm the real ballast before running.'
        )

    max_initial_distance = _parse_float(
        context,
        'max_initial_goal_distance_m',
        minimum=0.0,
        maximum=2.0,
    )
    max_initial_orientation_deg = _parse_float(
        context,
        'max_initial_goal_orientation_error_deg',
        minimum=0.0,
        maximum=180.0,
    )
    max_position_jump = _parse_float(
        context,
        'max_odom_position_jump_m',
        minimum=0.01,
        maximum=1.0,
    )
    max_orientation_jump_deg = _parse_float(
        context,
        'max_odom_orientation_jump_deg',
        minimum=1.0,
        maximum=90.0,
    )
    max_raw_mocap_message_age = _parse_float(
        context,
        'max_raw_mocap_message_age_sec',
        minimum=0.01,
        maximum=1.0,
    )
    max_mocap_coast = _parse_float(
        context,
        'max_mocap_coast_sec',
        minimum=0.25,
        maximum=2.0,
    )
    target_system_id = _parse_int(
        context, 'target_system_id', 1, 255
    )
    target_component_id = _parse_int(
        context, 'target_component_id', 1, 255
    )
    source_system_id = _parse_int(
        context, 'source_system_id', 1, 255
    )
    source_component_id = _parse_int(
        context, 'source_component_id', 1, 65535
    )

    acados_source = Path(
        _argument(context, 'acados_source_dir')
    ).expanduser().resolve()
    acados_lib = acados_source / 'lib'
    if not acados_source.is_dir() or not acados_lib.is_dir():
        raise RuntimeError(
            f'acados_source_dir={acados_source} or its lib directory is missing'
        )
    ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    updated_ld_path = str(acados_lib)
    if ld_library_path:
        updated_ld_path += ':' + ld_library_path

    nav_odom_topic = f'/mocap/{rigid_body_name}/odom_ekf_fixed_hook'
    vehicle_odom_topic = (
        f'/mocap/{rigid_body_name}/vehicle_odometry_fixed_hook'
    )
    control_mode_topic = namespace + '/fmu/out/vehicle_control_mode'
    vehicle_command_ack_topic = versioned_px4_topic(
        namespace + '/fmu/out/vehicle_command_ack',
        VehicleCommandAck,
    )
    thrust_topic = namespace + '/fmu/in/vehicle_thrust_setpoint'
    torque_topic = namespace + '/fmu/in/vehicle_torque_setpoint'
    px4_vehicle_odometry_topic = namespace + '/fmu/out/vehicle_odometry'

    mocap_ekf = Node(
        package='bluerov2_control',
        executable='mocap_ekf_odom',
        name='mocap_ekf_odom_fixed_hook',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'rigid_body_name': rigid_body_name,
            'pose_topic': raw_pose_topic,
            'odom_topic': nav_odom_topic,
            'parent_frame': target_frame_id,
            'child_frame': f'{rigid_body_name}/body_ekf_{body_frame}',
            'publish_rate_hz': 80.0,
            'publish_tf': False,
            'publish_static_flu_tf': False,
            'use_imu_gyro': False,
            'orientation_correction_quat_xyzw': correction_text,
            # Keep publishing EKF-predicted odometry through a bounded raw
            # MoCap interruption.  MPC still requires this 80 Hz output to be
            # fresh, and the EKF stops fail-closed when this window expires.
            'max_coast_sec': max_mocap_coast,
            'max_rejected_samples': 0,
            'max_pose_message_age_sec': max_raw_mocap_message_age,
            'max_pose_future_skew_sec': 0.10,
            'require_increasing_pose_stamp': True,
            'expected_pose_frame': target_frame_id,
            'reject_unexpected_pose_frame': True,
            'max_position_innovation_m': 0.20,
            'max_orientation_innovation_rad': 0.55,
            # Tilt is checked after the explicit NED/FRD conversion in MPC.
            'max_base_link_z_axis_angle_rad': 0.0,
        }],
    )

    odom_adapter = Node(
        package='bluerov2_control',
        executable='nav_odom_to_vehicle_odometry',
        name='nav_odom_to_vehicle_odometry_fixed_hook',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'input_odom_topic': nav_odom_topic,
            'output_vehicle_odometry_topic': vehicle_odom_topic,
            'pose_frame': 'ned',
            'velocity_frame': 'body_frd',
            'input_world_frame': world_frame,
            'input_body_frame': body_frame,
            'require_explicit_frame_transform': True,
            # Position, orientation, and linear velocity remain MoCap-based.
            # Replace only the delayed MoCap-derived body rate with PX4's
            # low-latency BODY_FRD angular velocity. Missing/stale feedback
            # makes this adapter withhold odometry and stops MPC fail-closed.
            'angular_velocity_override_topic': (
                px4_vehicle_odometry_topic
            ),
            'angular_velocity_override_timeout_sec': (
                px4_angular_velocity_timeout_sec
            ),
            'angular_velocity_override_max_norm_rad_s': 5.0,
            'quality': 100,
        }],
    )

    mpc = Node(
        package='bluerov2_control',
        executable='mpc_track_trajectory_acados',
        name='mpc_fixed_hook_pose_validation',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'odom_topic': vehicle_odom_topic,
            'control_mode_topic': control_mode_topic,
            'thrust_sp_topic': thrust_topic,
            'torque_sp_topic': torque_topic,
            'mission_enable_topic': MISSION_ENABLE_TOPIC,
            'require_mission_enable': True,
            'controller_heartbeat_topic': CONTROLLER_HEARTBEAT_TOPIC,
            'goal_x': float(target_position[0]),
            'goal_y': float(target_position[1]),
            'goal_z': float(target_position[2]),
            'goal_roll': goal_roll,
            'goal_pitch': goal_pitch,
            'goal_yaw': goal_yaw,
            'hold_attitude': True,
            'traj_mode': 'linear',
            'traj_speed_mps': traj_speed,
            'pre_approach_speed_mps': pre_approach_speed,
            'final_approach_speed_mps': final_approach_speed,
            'traj_angular_speed_rad_s': math.radians(
                traj_angular_speed_deg
            ),
            'min_traj_duration_s': min_traj_duration,
            'use_pre_approach_waypoint': True,
            'pre_approach_x': float(pre_approach_position[0]),
            'pre_approach_y': float(pre_approach_position[1]),
            'pre_approach_z': float(pre_approach_position[2]),
            'fixed_hook_depth_tolerance_m': (
                fixed_hook_depth_tolerance
            ),
            'final_pose_hold_s': final_pose_hold_s,
            'return_to_pre_approach_after_hold': True,
            'backward_pass_speed_mps': retreat_speed_mps,
            'goal_reached_tol_m': 0.05,
            'goal_reached_orientation_tol_rad': math.radians(5.0),
            'regenerate_on_goal_change': False,
            'planner_mode': 'none',
            'use_box_recovery_mission': False,
            'Ts': 0.04,
            'N': 25,
            'solve_rate_hz': 25.0,
            'model_type': 'fossen_real',
            'robot_type': robot_type,
            'w_pos': 50.0,
            'w_vel': 15.0,
            'w_att': w_att,
            'w_omega': w_omega,
            'w_u_force': 0.1,
            'w_u_torque': w_u_torque,
            'position_integral_gain_N_per_m_s': (
                position_integral_gain
            ),
            'position_integral_force_limit_fraction': (
                position_integral_force_limit_fraction
            ),
            'position_integral_activation_error_m': (
                position_integral_activation_error
            ),
            'position_integral_max_dt_s': 0.10,
            'Fx_max_N': 88.0,
            'Fy_max_N': 88.0,
            'Fz_max_N': 137.0,
            'Mx_max_Nm': 30.0,
            'My_max_Nm': 16.5,
            'Mz_max_Nm': 21.0,
            'thrust_sat': thrust_sat,
            'torque_sat': torque_sat,
            'publish_dt': 0.02,
            'odom_timeout_s': 0.20,
            'control_mode_timeout_s': 1.25,
            'command_timeout_s': 0.20,
            'max_solve_gap_s': 0.30,
            'max_solve_duration_s': 0.20,
            'revoke_mission_on_solver_failure': True,
            'revoke_mission_on_state_failure': True,
            'max_initial_goal_distance_m': max_initial_distance,
            'max_initial_goal_orientation_error_rad': math.radians(
                max_initial_orientation_deg
            ),
            'max_odom_position_jump_m': max_position_jump,
            'max_odom_orientation_jump_rad': math.radians(
                max_orientation_jump_deg
            ),
            'max_tilt_rad': 0.55,
            'require_expected_odom_frames': True,
            'require_increasing_odom_timestamp': True,
            'min_odom_quality': 50,
            'operating_bounds_enable': True,
            'operating_bounds_xmin_m': float(
                pool_bounds['operating_min_ned'][0]
            ),
            'operating_bounds_xmax_m': float(
                pool_bounds['operating_max_ned'][0]
            ),
            'operating_bounds_ymin_m': float(
                pool_bounds['operating_min_ned'][1]
            ),
            'operating_bounds_ymax_m': float(
                pool_bounds['operating_max_ned'][1]
            ),
            'operating_bounds_zmin_m': float(
                pool_bounds['operating_min_ned'][2]
            ),
            'operating_bounds_zmax_m': float(
                pool_bounds['operating_max_ned'][2]
            ),
            'codegen_dir': (
                f'/tmp/bluerov2_acados_fixed_hook_{robot_type}'
            ),
            'rebuild_solver': _parse_bool(context, 'rebuild_solver'),
        }],
    )

    notes = json.dumps(
        {
            'stage': 'fixed_hook_pose_validation',
            'target_config': str(target_path),
            'target_source_topic': payload.get('source_topic', ''),
            'target_frame_id': payload.get('frame_id', ''),
            'target_correction_applied': target_correction_applied,
            'orientation_correction_quat_xyzw': correction_xyzw.tolist(),
            'target_orientation_correction_mode': correction_mode,
            'mocap_world_frame': world_frame,
            'mocap_body_frame': body_frame,
            'robot_type': robot_type,
            'robot_namespace': namespace,
            'px4_msgs_prefix': px4_msgs_prefix,
            'vehicle_command_ack_topic': vehicle_command_ack_topic,
            'vehicle_command_ack_message_version': ack_message_version,
            'vehicle_status_message_version': status_message_version,
            'target_system_id': target_system_id,
            'target_component_id': target_component_id,
            'source_system_id': source_system_id,
            'source_component_id': source_component_id,
            'goal_ned_m': target_position.tolist(),
            'goal_quaternion_wxyz_ned_frd': target_quat.tolist(),
            'use_pre_approach_waypoint': True,
            'pre_approach_distance_m': pre_approach_distance,
            'pre_approach_line_distance_m': (
                pre_approach_line_distance
            ),
            'pre_approach_geometry': (
                'recorded_yaw_horizontal_body_forward'
            ),
            'pre_approach_forward_yaw_rad': goal_yaw,
            'pre_approach_ned_m': pre_approach_position.tolist(),
            'fixed_hook_depth_tolerance_m': (
                fixed_hook_depth_tolerance
            ),
            'pool_nominal_dimensions_m': [9.0, 5.0, 3.0],
            'traj_speed_mps': traj_speed,
            'pre_approach_speed_mps': pre_approach_speed,
            'final_approach_speed_mps': final_approach_speed,
            'traj_angular_speed_deg_s': traj_angular_speed_deg,
            'min_traj_duration_s': min_traj_duration,
            'final_pose_hold_s': final_pose_hold_s,
            'return_to_pre_approach_after_hold': True,
            'retreat_speed_mps': retreat_speed_mps,
            'angular_velocity_source_topic': (
                px4_vehicle_odometry_topic
            ),
            'px4_angular_velocity_timeout_sec': (
                px4_angular_velocity_timeout_sec
            ),
            'w_att': w_att,
            'w_omega': w_omega,
            'w_u_torque': w_u_torque,
            'position_integral_gain_N_per_m_s': (
                position_integral_gain
            ),
            'position_integral_force_limit_fraction': (
                position_integral_force_limit_fraction
            ),
            'position_integral_activation_error_m': (
                position_integral_activation_error
            ),
            'thrust_sat': thrust_sat,
            'torque_sat': torque_sat,
            'max_initial_goal_distance_m': max_initial_distance,
            'max_initial_goal_orientation_error_deg': (
                max_initial_orientation_deg
            ),
            'max_odom_position_jump_m': max_position_jump,
            'max_odom_orientation_jump_deg': max_orientation_jump_deg,
            'max_raw_mocap_message_age_sec': max_raw_mocap_message_age,
            'pool_bounds_mocap_m': {
                'min_xyz': pool_bounds['raw_min'].tolist(),
                'max_xyz': pool_bounds['raw_max'].tolist(),
            },
            'pool_span_mocap_m': (
                pool_bounds['raw_max'] - pool_bounds['raw_min']
            ).tolist(),
            'pool_bounds_ned_m': {
                'min_ned': pool_bounds['ned_min'].tolist(),
                'max_ned': pool_bounds['ned_max'].tolist(),
            },
            'pool_safety_margin_m': pool_bounds['safety_margin_m'],
            'operating_bounds_ned_m': {
                'min_ned': pool_bounds['operating_min_ned'].tolist(),
                'max_ned': pool_bounds['operating_max_ned'].tolist(),
            },
            'operating_span_ned_m': (
                pool_bounds['operating_max_ned']
                - pool_bounds['operating_min_ned']
            ).tolist(),
        },
        sort_keys=True,
    )
    logger = Node(
        package='bluerov2_control',
        executable='payload_retrieval_data_logger',
        name='fixed_hook_pose_validation_logger',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'trial_id': _argument(context, 'trial_id'),
            'output_dir': _argument(context, 'trial_output_dir'),
            'sample_period_s': 0.05,
            'stale_after_s': 0.30,
            'metadata_file': str(target_path),
            'controller_name': 'mpc_fixed_hook_pose_validation',
            'environment': 'real_pool_9x5x3m',
            'notes': notes,
            'odom_topic': vehicle_odom_topic,
            'mocap_odom_topic': nav_odom_topic,
            'cmd_vel_topic': '',
            'thrust_sp_topic': thrust_topic,
            'torque_sp_topic': torque_topic,
            'attitude_sp_topic': '',
            'control_mode_topic': control_mode_topic,
            'offboard_request_enable_topic': OFFBOARD_REQUEST_TOPIC,
            'mission_enable_topic': MISSION_ENABLE_TOPIC,
            'handle_pose_topic': '',
            'handle_confidence_topic': '',
            'handle_detected_topic': '',
            'payload_pose_topic': '',
            'dock_pose_topic': '',
            'task_event_topic': TRIAL_EVENT_TOPIC,
        }],
    )

    offboard = Node(
        package='bluerov2_control',
        executable='offboard_enable',
        name='offboard_enable_fixed_hook',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'offboard_mode_topic': (
                namespace + '/fmu/in/offboard_control_mode'
            ),
            'vehicle_cmd_topic': namespace + '/fmu/in/vehicle_command',
            'vehicle_cmd_ack_topic': vehicle_command_ack_topic,
            'vehicle_control_mode_topic': control_mode_topic,
            'request_enable_topic': OFFBOARD_REQUEST_TOPIC,
            'require_request_enable': True,
            'controller_heartbeat_topic': CONTROLLER_HEARTBEAT_TOPIC,
            'require_controller_heartbeat': True,
            'controller_heartbeat_timeout_sec': 0.30,
            'enforce_control_publisher_exclusivity': True,
            'expected_controller_node_fqn': (
                '/mpc_fixed_hook_pose_validation'
            ),
            'exclusive_thrust_sp_topic': thrust_topic,
            'exclusive_torque_sp_topic': torque_topic,
            'auto_arm': False,
            'target_system_id': target_system_id,
            'target_component_id': target_component_id,
            'source_system_id': source_system_id,
            'source_component_id': source_component_id,
            'prestream_duration_sec': 2.0,
            'command_retry_interval_sec': 1.0,
            'control_mode_timeout_sec': 1.25,
        }],
    )

    nodes = [mocap_ekf, odom_adapter, mpc, logger, offboard]
    handlers = [
        _critical_exit_handler(node, label)
        for node, label in (
            (mocap_ekf, 'mocap_ekf'),
            (odom_adapter, 'odom_adapter'),
            (mpc, 'mpc'),
            (logger, 'logger'),
            (offboard, 'offboard_enable'),
        )
    ]
    target_summary = (
        'Validated target in NED/FRD: '
        f'p=[{target_position[0]:.3f}, {target_position[1]:.3f}, '
        f'{target_position[2]:.3f}] m, '
        f'rpy=[{goal_roll:.3f}, {goal_pitch:.3f}, {goal_yaw:.3f}] rad; '
        f'samples={metrics["accepted_sample_count"]}'
    )
    pre_approach_summary = (
        'Pre-approach waypoint in NED: '
        f'p=[{pre_approach_position[0]:.3f}, '
        f'{pre_approach_position[1]:.3f}, '
        f'{pre_approach_position[2]:.3f}] m, '
        f'body-forward distance={pre_approach_line_distance:.3f} m, '
        f'depth tolerance={fixed_hook_depth_tolerance:.3f} m; the controller '
        'will settle to the recorded attitude and common line depth before '
        'advancing along the recorded horizontal nose direction.'
    )
    mission_sequence_summary = (
        'Fixed-hook pose sequence: PRE_APPROACH -> FINAL_APPROACH -> '
        f'FINAL_HOLD({final_pose_hold_s:.1f}s) -> straight RETREAT to '
        'PRE_APPROACH -> COMPLETE; speeds: '
        f'pre={pre_approach_speed:.3f}m/s, '
        f'final={final_approach_speed:.3f}m/s, '
        f'retreat={retreat_speed_mps:.3f}m/s.'
    )
    rate_feedback_summary = (
        'Attitude-rate damping uses PX4 BODY_FRD angular velocity from '
        f'{px4_vehicle_odometry_topic}; timeout='
        f'{px4_angular_velocity_timeout_sec:.3f}s (fail-closed).'
    )
    position_integral_summary = (
        'Bounded offset-free position correction: gain='
        f'{position_integral_gain:.2f}N/(m*s), bias limit='
        f'{position_integral_force_limit_fraction:.3f} of each force axis, '
        f'activation error<={position_integral_activation_error:.2f}m; '
        'MPC plus integral remains inside thrust_sat.'
    )
    px4_msgs_summary = (
        f'Firmware-matched px4_msgs: {px4_msgs_prefix}; '
        f'ACK/Status versions={ack_message_version}/{status_message_version}.'
    )
    permission_summary = (
        f'After preflight, enable OFFBOARD requests on '
        f'{OFFBOARD_REQUEST_TOPIC}; enable motion separately on '
        f'{MISSION_ENABLE_TOPIC}. auto_arm is always false.'
    )
    bounds_summary = (
        'Online NED operating bounds (after '
        f'{pool_bounds["safety_margin_m"]:.3f} m margin): '
        f'min={pool_bounds["operating_min_ned"].tolist()}, '
        f'max={pool_bounds["operating_max_ned"].tolist()}'
    )
    return [
        SetEnvironmentVariable('ACADOS_SOURCE_DIR', str(acados_source)),
        SetEnvironmentVariable('LD_LIBRARY_PATH', updated_ld_path),
        *handlers,
        LogInfo(msg=target_summary),
        LogInfo(msg=pre_approach_summary),
        LogInfo(msg=mission_sequence_summary),
        LogInfo(msg=rate_feedback_summary),
        LogInfo(msg=position_integral_summary),
        LogInfo(msg=px4_msgs_summary),
        LogInfo(msg=bounds_summary),
        LogInfo(msg=permission_summary),
        *nodes,
    ]


def generate_launch_description():
    """Declare guarded real-experiment arguments."""
    return LaunchDescription([
        DeclareLaunchArgument(
            'rigid_body_name',
            default_value='glub_fb',
            description='MoCap rigid-body name used under /mocap.',
        ),
        DeclareLaunchArgument(
            'robot_namespace',
            default_value='/glub',
            description=(
                'PX4 DDS ROS namespace; independent of rigid_body_name.'
            ),
        ),
        DeclareLaunchArgument(
            'target_config',
            default_value=(
                '/home/yecheng/bluerov_ws/src/bluerov2_control/'
                'experiments/payload_retrieval/config/'
                'hooked_box_target_pose_20260802_195146.json'
            ),
            description='Validated glub_fb target recorded at the hooked pose.',
        ),
        DeclareLaunchArgument(
            'mocap_world_frame',
            default_value='ned',
            description='Raw MoCap world axes: ned, nwu, or enu.',
        ),
        DeclareLaunchArgument(
            'pool_bounds_mocap',
            default_value='0 9 -2.5 2.5 0 3',
            description=(
                'Raw MoCap pool bounds: '
                'xmin xmax ymin ymax zmin zmax.'
            ),
        ),
        DeclareLaunchArgument(
            'pool_safety_margin_m',
            default_value='0.25',
            description='Margin removed from every converted NED boundary.',
        ),
        DeclareLaunchArgument(
            'mocap_body_frame',
            default_value='frd',
            description='EKF body axes after correction: frd or flu.',
        ),
        DeclareLaunchArgument(
            'robot_type',
            default_value='standard',
            description='Real dynamics preset: standard or heavy_tube.',
        ),
        DeclareLaunchArgument(
            'orientation_correction_quat_xyzw',
            default_value='',
            description='Optional raw-marker-to-EKF-body right correction.',
        ),
        DeclareLaunchArgument(
            'target_orientation_correction_mode',
            default_value='auto',
            description='auto/apply/skip correction for the recorded target.',
        ),
        DeclareLaunchArgument('traj_speed_mps', default_value='0.05'),
        DeclareLaunchArgument(
            'pre_approach_speed_mps',
            default_value='0.06',
            description=(
                'Straight-line reference speed while travelling to the '
                'pre-hooking waypoint.'
            ),
        ),
        DeclareLaunchArgument(
            'final_approach_speed_mps',
            default_value='0.09',
            description=(
                'Straight-line reference speed for the final hook approach.'
            ),
        ),
        DeclareLaunchArgument('min_traj_duration_s', default_value='5.0'),
        DeclareLaunchArgument(
            'pre_approach_distance_m',
            default_value='0.50',
            description=(
                'Horizontal distance behind the recorded final body-forward '
                'direction used to construct the pre-hook waypoint.'
            ),
        ),
        DeclareLaunchArgument(
            'fixed_hook_depth_tolerance_m',
            default_value='0.03',
            description=(
                'Independent NED-depth tolerance required before fixed-hook '
                'stage transitions; forward and retreat references lock to '
                'the recorded hook depth.'
            ),
        ),
        DeclareLaunchArgument(
            'traj_angular_speed_deg_s',
            default_value='8.0',
            description='Maximum reference-attitude slew rate in deg/s.',
        ),
        DeclareLaunchArgument(
            'final_pose_hold_s',
            default_value='5.0',
            description=(
                'Continuous in-tolerance hold at the recorded hook pose '
                'before retreating.'
            ),
        ),
        DeclareLaunchArgument(
            'retreat_speed_mps',
            default_value='0.09',
            description=(
                'Straight-line position-reference speed from the recorded '
                'hook pose back to pre-approach.'
            ),
        ),
        DeclareLaunchArgument(
            'px4_angular_velocity_timeout_sec',
            default_value='0.10',
            description=(
                'Maximum receive age of PX4 BODY_FRD angular velocity; '
                'stale feedback blocks MPC odometry.'
            ),
        ),
        DeclareLaunchArgument(
            'w_att',
            default_value='10.0',
            description='Conservative real-robot attitude-error weight.',
        ),
        DeclareLaunchArgument(
            'w_omega',
            default_value='20.0',
            description=(
                'Body-rate damping weight used with low-latency PX4 rate.'
            ),
        ),
        DeclareLaunchArgument(
            'w_u_torque',
            default_value='0.5',
            description='Real-robot torque command penalty.',
        ),
        DeclareLaunchArgument(
            'position_integral_gain_N_per_m_s',
            default_value='3.0',
            description=(
                'Bounded offset-free position gain, enabled only after the '
                'nominal reference finishes.'
            ),
        ),
        DeclareLaunchArgument(
            'position_integral_force_limit_fraction',
            default_value='0.07',
            description=(
                'Maximum integral force bias per physical force axis; the '
                'combined MPC plus bias remains inside thrust_sat.'
            ),
        ),
        DeclareLaunchArgument(
            'position_integral_activation_error_m',
            default_value='0.50',
            description=(
                'Maximum goal error at which offset-free integration is '
                'allowed; larger errors remain fail-safe and unmasked.'
            ),
        ),
        DeclareLaunchArgument('thrust_sat', default_value='0.12'),
        DeclareLaunchArgument('torque_sat', default_value='0.02'),
        DeclareLaunchArgument(
            'max_initial_goal_distance_m',
            default_value='0',
            description=(
                'Optional legacy spherical start-distance gate; 0 disables it.'
            ),
        ),
        DeclareLaunchArgument(
            'max_initial_goal_orientation_error_deg',
            default_value='0',
            description=(
                'Optional legacy start-attitude gate in degrees; 0 disables it.'
            ),
        ),
        DeclareLaunchArgument(
            'max_odom_position_jump_m', default_value='0.20'
        ),
        DeclareLaunchArgument(
            'max_odom_orientation_jump_deg', default_value='20.0'
        ),
        DeclareLaunchArgument('target_min_samples', default_value='80'),
        DeclareLaunchArgument(
            'target_min_sampling_span_sec', default_value='0.75'
        ),
        DeclareLaunchArgument(
            'target_max_position_std_m', default_value='0.015'
        ),
        DeclareLaunchArgument(
            'target_max_orientation_std_deg', default_value='1.5'
        ),
        DeclareLaunchArgument(
            'target_max_orientation_spread_deg', default_value='5.0'
        ),
        DeclareLaunchArgument(
            'max_raw_mocap_message_age_sec',
            default_value='0.20',
            description=(
                'Shared raw MoCap age limit for target recording validation '
                'and the live EKF.'
            ),
        ),
        DeclareLaunchArgument(
            'max_mocap_coast_sec',
            default_value='2.0',
            description=(
                'Maximum bounded EKF prediction interval during a raw MoCap '
                'interruption; longer interruptions still stop odometry.'
            ),
        ),
        DeclareLaunchArgument(
            'target_system_id',
            default_value='3',
            description='PX4 MAVLink system_id from VehicleStatus.',
        ),
        DeclareLaunchArgument(
            'target_component_id',
            default_value='1',
            description='PX4 MAVLink component_id from VehicleStatus.',
        ),
        DeclareLaunchArgument('source_system_id', default_value='1'),
        DeclareLaunchArgument('source_component_id', default_value='191'),
        DeclareLaunchArgument(
            'trial_output_dir',
            default_value=(
                '/home/yecheng/bluerov_ws/'
                'bluerov2_payload_retrieval_trials'
            ),
        ),
        DeclareLaunchArgument('trial_id', default_value=''),
        DeclareLaunchArgument(
            'acados_source_dir', default_value='/home/yecheng/acados'
        ),
        DeclareLaunchArgument('rebuild_solver', default_value='false'),
        OpaqueFunction(function=_launch_setup),
    ])
