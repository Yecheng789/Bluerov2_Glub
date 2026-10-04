#!/usr/bin/env python3
"""Run the guarded real-pool fixed-hook target-pose validation."""

import hashlib
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
from bluerov2_control.planner_astar import (
    OccupancyGrid2D as PlannerOccupancyGrid2D,
    line_segment_is_free as planner_line_segment_is_free,
)
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
HOOK_CONFIRMATION_SERVICE = '/bluerov2/fixed_hook/confirm_hook'
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


def _parse_optional_timeout(context, name):
    """Parse zero=disabled or a guarded timeout of at least five seconds."""
    value = _parse_float(context, name, minimum=0.0, maximum=120.0)
    if 0.0 < value < 5.0:
        raise RuntimeError(
            f'{name} must be 0 (disabled) or >= 5.0'
        )
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


def _parse_static_obstacle_rectangles(context):
    """Parse optional real-NED AABBs as xmin xmax ymin ymax groups."""
    text = _argument(context, 'prehook_static_obstacles_ned_xyxy')
    if not text:
        return [], ''
    rectangles = []
    for index, group in enumerate(text.split(';'), start=1):
        words = group.replace(',', ' ').split()
        if len(words) != 4:
            raise RuntimeError(
                'prehook_static_obstacles_ned_xyxy rectangle '
                f'{index} must contain xmin xmax ymin ymax'
            )
        try:
            rectangle = tuple(float(word) for word in words)
        except ValueError as exc:
            raise RuntimeError(
                'prehook_static_obstacles_ned_xyxy must contain numbers'
            ) from exc
        if not all(math.isfinite(value) for value in rectangle):
            raise RuntimeError(
                'prehook_static_obstacles_ned_xyxy must contain finite values'
            )
        xmin, xmax, ymin, ymax = rectangle
        if xmin >= xmax or ymin >= ymax:
            raise RuntimeError(
                'prehook_static_obstacles_ned_xyxy requires xmin < xmax '
                'and ymin < ymax for every rectangle'
            )
        rectangles.append(rectangle)
    normalized = ';'.join(
        ' '.join(f'{value:.9g}' for value in rectangle)
        for rectangle in rectangles
    )
    return rectangles, normalized


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


def _sha256_file(path):
    """Return the SHA-256 digest of a target provenance file."""
    digest = hashlib.sha256()
    try:
        with path.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(block)
    except OSError as exc:
        raise RuntimeError(f'cannot hash source target config={path}: {exc}') \
            from exc
    return digest.hexdigest()


def _strict_identity_transfer(values, expected, field_name):
    """Validate an explicitly declared identity pose-transfer component."""
    if not isinstance(values, list) or len(values) != len(expected):
        raise RuntimeError(
            f'derived target {field_name} must contain {len(expected)} values'
        )
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in values
    ):
        raise RuntimeError(
            f'derived target {field_name} must contain JSON numbers'
        )
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)) or not np.array_equal(
        array, np.asarray(expected, dtype=float)
    ):
        raise RuntimeError(
            f'derived target {field_name} must be the exact identity value '
            f'{list(expected)}'
        )


def _resolve_target_payload(target_path, expected_pose_topic, allow_derived):
    """Resolve a direct recording or a hash-locked identity derivation."""
    manifest = _load_target_payload(target_path)
    if manifest.get('status') == 'invalidated_for_control':
        raise RuntimeError(
            'target_config is explicitly marked invalidated_for_control; '
            'record a new target from the active robot MoCap rigid body'
        )
    derivation = manifest.get('derived_target')
    if derivation is None:
        source_topic = str(manifest.get('source_topic', '')).strip()
        if source_topic != expected_pose_topic:
            raise RuntimeError(
                f'target source_topic must be {expected_pose_topic}; record '
                'a new raw MoCap target for this rigid body'
            )
        return manifest, {
            'kind': 'direct_mocap_recording',
            'source_config': str(target_path),
            'source_pose_topic': source_topic,
            'destination_pose_topic': expected_pose_topic,
        }

    if not allow_derived:
        raise RuntimeError(
            'target_config is an identity-derived target, but this launch '
            'profile does not explicitly allow derived targets'
        )
    if not isinstance(derivation, dict):
        raise RuntimeError('derived_target must contain a JSON object')
    if derivation.get('schema_version') != 1:
        raise RuntimeError('derived target schema_version must be 1')
    if derivation.get('type') != 'identity_pose_transfer':
        raise RuntimeError(
            'derived target type must be identity_pose_transfer'
        )
    if derivation.get('operator_confirmed_equivalent_geometry') is not True:
        raise RuntimeError(
            'derived target requires explicit operator confirmation of '
            'equivalent robot and rigid-body geometry'
        )
    if derivation.get('operator_confirmed_same_target_pose') is not True:
        raise RuntimeError(
            'derived target requires explicit operator confirmation that '
            'the fixed Hook target pose is unchanged'
        )

    destination_topic = str(
        derivation.get('destination_pose_topic', '')
    ).strip()
    if destination_topic != expected_pose_topic:
        raise RuntimeError(
            'derived target destination_pose_topic must be '
            f'{expected_pose_topic}'
        )
    source_topic = str(derivation.get('source_pose_topic', '')).strip()
    if not source_topic or source_topic == destination_topic:
        raise RuntimeError(
            'derived target source_pose_topic must identify the distinct '
            'original recording topic'
        )

    _strict_identity_transfer(
        derivation.get('position_offset_m'),
        [0.0, 0.0, 0.0],
        'position_offset_m',
    )
    _strict_identity_transfer(
        derivation.get('orientation_offset_xyzw'),
        [0.0, 0.0, 0.0, 1.0],
        'orientation_offset_xyzw',
    )

    source_name = str(derivation.get('source_config', '')).strip()
    if not source_name or Path(source_name).name != source_name:
        raise RuntimeError(
            'derived target source_config must be a filename in the same '
            'directory as target_config'
        )
    source_path = (target_path.parent / source_name).resolve()
    if source_path.parent != target_path.parent or source_path == target_path:
        raise RuntimeError('derived target source_config path is invalid')

    expected_sha256 = str(
        derivation.get('source_config_sha256', '')
    ).strip().lower()
    if (
        len(expected_sha256) != 64
        or any(character not in '0123456789abcdef'
               for character in expected_sha256)
    ):
        raise RuntimeError(
            'derived target source_config_sha256 must be 64 hexadecimal '
            'characters'
        )
    actual_sha256 = _sha256_file(source_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            'derived target source_config SHA-256 mismatch; do not use a '
            'modified or substituted source recording'
        )

    source_payload = _load_target_payload(source_path)
    if source_payload.get('derived_target') is not None:
        raise RuntimeError('derived target chains are not allowed')
    if str(source_payload.get('source_topic', '')).strip() != source_topic:
        raise RuntimeError(
            'derived target source_pose_topic does not match the source '
            'recording provenance'
        )
    for field_name in (
        'frame_id',
        'message_type',
        'source_topic',
        'target_pose',
    ):
        if manifest.get(field_name) != source_payload.get(field_name):
            raise RuntimeError(
                f'derived target {field_name} snapshot does not exactly '
                'match the hash-locked source recording'
            )

    return source_payload, {
        'kind': 'identity_pose_transfer',
        'manifest_config': str(target_path),
        'source_config': str(source_path),
        'source_config_sha256': actual_sha256,
        'source_pose_topic': source_topic,
        'destination_pose_topic': destination_topic,
        'position_offset_m': [0.0, 0.0, 0.0],
        'orientation_offset_xyzw': [0.0, 0.0, 0.0, 1.0],
        'operator_confirmed_equivalent_geometry': True,
        'operator_confirmed_same_target_pose': True,
    }


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
    if not target_text or target_text.lower() == 'unconfigured':
        raise RuntimeError(
            'target_config is required; manually hook the target and record a '
            'new validated MoCap pose before launching this experiment'
        )
    target_path = Path(target_text).expanduser().resolve()
    raw_pose_topic = f'/mocap/{rigid_body_name}/pose'
    payload, target_provenance = _resolve_target_payload(
        target_path,
        raw_pose_topic,
        _parse_bool(context, 'allow_identity_derived_target'),
    )
    if payload.get('message_type') != 'geometry_msgs/PoseStamped':
        raise RuntimeError(
            'fixed-hook reality validation requires a target recorded '
            'directly from geometry_msgs/PoseStamped, not filtered odometry'
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
    planner_check_rate_hz = _parse_float(
        context,
        'prehook_planner_check_rate_hz',
        minimum=2.0,
        maximum=5.0,
    )
    replan_deviation_m = _parse_float(
        context, 'prehook_replan_deviation_m', minimum=0.05, maximum=1.0
    )
    replan_deviation_hold_s = _parse_float(
        context,
        'prehook_replan_deviation_hold_s',
        minimum=0.0,
        maximum=5.0,
    )
    replan_min_switch_interval_s = _parse_float(
        context,
        'prehook_replan_min_switch_interval_s',
        minimum=0.0,
        maximum=10.0,
    )
    replan_min_improvement_m = _parse_float(
        context,
        'prehook_replan_min_improvement_m',
        minimum=0.0,
        maximum=1.0,
    )
    replan_min_improvement_ratio = _parse_float(
        context,
        'prehook_replan_min_improvement_ratio',
        minimum=0.0,
        maximum=1.0,
    )
    replan_optimization_period_s = _parse_float(
        context,
        'prehook_replan_optimization_period_s',
        minimum=0.0,
        maximum=30.0,
    )
    prehook_reached_hold_s = _parse_float(
        context, 'prehook_reached_hold_s', minimum=0.0, maximum=10.0
    )
    prehook_attitude_reference_mode = _parse_choice(
        context,
        'prehook_attitude_reference_mode',
        {'recorded_hook', 'capture_start_trim'},
    )
    prehook_reached_orientation_tol_deg = _parse_float(
        context,
        'prehook_reached_orientation_tol_deg',
        minimum=1.0,
        maximum=20.0,
    )
    prehook_reached_forward_axis_tol_deg = _parse_float(
        context,
        'prehook_reached_forward_axis_tol_deg',
        minimum=0.5,
        maximum=20.0,
    )
    prehook_reached_yaw_tol_deg = _parse_float(
        context,
        'prehook_reached_yaw_tol_deg',
        minimum=0.5,
        maximum=20.0,
    )
    fixed_hook_line_position_mode = _parse_bool(
        context,
        'fixed_hook_line_position_mode',
    )
    fixed_hook_line_yaw_tol_deg = _parse_float(
        context,
        'fixed_hook_line_yaw_tol_deg',
        minimum=0.5,
        maximum=20.0,
    )
    fixed_hook_line_cross_track_tol_m = _parse_float(
        context,
        'fixed_hook_line_cross_track_tol_m',
        minimum=0.005,
        maximum=0.20,
    )
    fixed_hook_line_interlock_release_ratio = _parse_float(
        context,
        'fixed_hook_line_interlock_release_ratio',
        minimum=0.0,
        maximum=1.0,
    )
    if fixed_hook_line_interlock_release_ratio <= 0.0:
        raise RuntimeError(
            'fixed_hook_line_interlock_release_ratio must be > 0'
        )
    fixed_hook_line_max_reference_lead_m = _parse_float(
        context,
        'fixed_hook_line_max_reference_lead_m',
        minimum=0.005,
        maximum=0.20,
    )
    fixed_hook_line_velocity_weight_multiplier = _parse_float(
        context,
        'fixed_hook_line_velocity_weight_multiplier',
        minimum=1.0,
        maximum=50.0,
    )
    prehook_attitude_wait_exit_hysteresis_ratio = _parse_float(
        context,
        'prehook_attitude_wait_exit_hysteresis_ratio',
        minimum=1.0,
        maximum=3.0,
    )
    prehook_attitude_alignment_timeout_s = _parse_optional_timeout(
        context,
        'prehook_attitude_alignment_timeout_s',
    )
    astar_resolution = _parse_float(
        context, 'prehook_astar_resolution_m', minimum=0.03, maximum=0.25
    )
    astar_robot_radius = _parse_float(
        context, 'prehook_robot_radius_m', minimum=0.05, maximum=0.60
    )
    astar_obstacle_margin = _parse_float(
        context, 'prehook_obstacle_margin_m', minimum=0.0, maximum=0.50
    )
    smoothing_iterations = _parse_int(
        context, 'prehook_path_smoothing_iterations', 0, 4
    )
    smoothing_corner_fraction = _parse_float(
        context,
        'prehook_path_smoothing_corner_fraction',
        minimum=0.01,
        maximum=0.49,
    )
    smoothing_samples = _parse_int(
        context, 'prehook_path_smoothing_samples_per_corner', 2, 12
    )
    static_obstacles, static_obstacles_text = (
        _parse_static_obstacle_rectangles(context)
    )
    obstacle_inflation = astar_robot_radius + astar_obstacle_margin
    if pool_safety_margin + 1e-9 < obstacle_inflation:
        raise RuntimeError(
            'pool_safety_margin_m must be at least '
            'prehook_robot_radius_m + prehook_obstacle_margin_m so the '
            'real-NED A* pool-wall clearance covers the configured robot '
            'collision envelope'
        )
    inflated_obstacles = []
    for index, rectangle in enumerate(static_obstacles, start=1):
        xmin, xmax, ymin, ymax = rectangle
        inflated = (
            xmin - obstacle_inflation,
            xmax + obstacle_inflation,
            ymin - obstacle_inflation,
            ymax + obstacle_inflation,
        )
        inflated_obstacles.append(inflated)
        if (
            inflated[0] <= pre_approach_position[0] <= inflated[1]
            and inflated[2] <= pre_approach_position[1] <= inflated[3]
        ):
            raise RuntimeError(
                f'pre-hook waypoint lies inside inflated static obstacle '
                f'{index}; correct the measured real-NED obstacle geometry'
            )
    final_corridor_grid = PlannerOccupancyGrid2D(
        bounds=(
            float(pool_bounds['operating_min_ned'][0]),
            float(pool_bounds['operating_max_ned'][0]),
            float(pool_bounds['operating_min_ned'][1]),
            float(pool_bounds['operating_max_ned'][1]),
        ),
        resolution=astar_resolution,
        obstacles=inflated_obstacles,
    )
    if not planner_line_segment_is_free(
        final_corridor_grid,
        tuple(float(value) for value in pre_approach_position[0:2]),
        tuple(float(value) for value in target_position[0:2]),
    ):
        raise RuntimeError(
            'the fixed pre-hook-to-hook GO_FORWARD/GO_BACK corridor '
            'intersects an inflated no-contact static obstacle; correct the '
            'real-NED obstacle geometry or move the fixed hook corridor'
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
            # Reject a physically implausible MoCap attitude before its
            # prediction can reach MPC; keep automatic EKF reinitialization
            # disabled so a repeated bad pose cannot become the new state.
            'max_base_link_z_axis_angle_rad': 0.55,
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
            'require_operator_hook_confirmation': True,
            'hook_confirmation_service': HOOK_CONFIRMATION_SERVICE,
            'hook_confirmation_min_wait_s': 0.25,
            'backward_pass_speed_mps': retreat_speed_mps,
            'goal_reached_tol_m': 0.05,
            'goal_reached_orientation_tol_rad': math.radians(5.0),
            'regenerate_on_goal_change': False,
            'planner_mode': 'astar',
            'use_dynamic_prehook_planner': True,
            'prehook_planner_check_rate_hz': planner_check_rate_hz,
            'prehook_replan_deviation_m': replan_deviation_m,
            'prehook_replan_deviation_hold_s': (
                replan_deviation_hold_s
            ),
            'prehook_replan_min_switch_interval_s': (
                replan_min_switch_interval_s
            ),
            'prehook_replan_min_improvement_m': (
                replan_min_improvement_m
            ),
            'prehook_replan_min_improvement_ratio': (
                replan_min_improvement_ratio
            ),
            'prehook_replan_optimization_period_s': (
                replan_optimization_period_s
            ),
            'prehook_reached_hold_s': prehook_reached_hold_s,
            'prehook_attitude_reference_mode': (
                prehook_attitude_reference_mode
            ),
            'prehook_reached_orientation_tol_rad': math.radians(
                prehook_reached_orientation_tol_deg
            ),
            'prehook_reached_forward_axis_tol_rad': math.radians(
                prehook_reached_forward_axis_tol_deg
            ),
            'prehook_reached_yaw_tol_rad': math.radians(
                prehook_reached_yaw_tol_deg
            ),
            'fixed_hook_line_yaw_tol_rad': math.radians(
                fixed_hook_line_yaw_tol_deg
            ),
            'fixed_hook_line_position_mode': (
                fixed_hook_line_position_mode
            ),
            'fixed_hook_line_cross_track_tol_m': (
                fixed_hook_line_cross_track_tol_m
            ),
            'fixed_hook_line_interlock_release_ratio': (
                fixed_hook_line_interlock_release_ratio
            ),
            'fixed_hook_line_max_reference_lead_m': (
                fixed_hook_line_max_reference_lead_m
            ),
            'fixed_hook_line_velocity_weight_multiplier': (
                fixed_hook_line_velocity_weight_multiplier
            ),
            'prehook_attitude_wait_exit_hysteresis_ratio': (
                prehook_attitude_wait_exit_hysteresis_ratio
            ),
            'prehook_attitude_alignment_timeout_s': (
                prehook_attitude_alignment_timeout_s
            ),
            'prehook_static_obstacles_ned_xyxy': static_obstacles_text,
            'astar_resolution': astar_resolution,
            'astar_robot_radius': astar_robot_radius,
            'astar_obstacle_margin': astar_obstacle_margin,
            'astar_diagonal_motion': True,
            'prehook_path_smoothing_iterations': smoothing_iterations,
            'prehook_path_smoothing_corner_fraction': (
                smoothing_corner_fraction
            ),
            'prehook_path_smoothing_samples_per_corner': (
                smoothing_samples
            ),
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
            'target_live_pose_topic': raw_pose_topic,
            'target_provenance': target_provenance,
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
            'prehook_planner': {
                'mode': 'astar_real_ned',
                'active_states': [
                    'PLAN_TO_PREHOOK',
                    'TRACK_TO_PREHOOK',
                ],
                'disabled_states': [
                    'PREHOOK_REACHED',
                    'GO_FORWARD',
                    'WAIT_HOOK',
                    'GO_BACK',
                    'COMPLETE',
                ],
                'check_rate_hz': planner_check_rate_hz,
                'replan_deviation_m': replan_deviation_m,
                'replan_deviation_hold_s': replan_deviation_hold_s,
                'min_switch_interval_s': replan_min_switch_interval_s,
                'min_improvement_m': replan_min_improvement_m,
                'min_improvement_ratio': replan_min_improvement_ratio,
                'optimization_period_s': replan_optimization_period_s,
                'prehook_reached_hold_s': prehook_reached_hold_s,
                'attitude_reference_mode': prehook_attitude_reference_mode,
                'reached_orientation_tolerance_deg': (
                    prehook_reached_orientation_tol_deg
                ),
                'reached_forward_axis_tolerance_deg': (
                    prehook_reached_forward_axis_tol_deg
                ),
                'attitude_gate_semantics': (
                    'full_attitude_safety_envelope_and_body_x_forward_axis_'
                    'and_yaw'
                ),
                'reached_yaw_tolerance_deg': (
                    prehook_reached_yaw_tol_deg
                ),
                'attitude_wait_exit_hysteresis_ratio': (
                    prehook_attitude_wait_exit_hysteresis_ratio
                ),
                'attitude_wait_exit_hysteresis_semantics': (
                    'position_depth_exit_gate_ratio'
                ),
                'attitude_alignment_timeout_s': (
                    prehook_attitude_alignment_timeout_s
                ),
                'attitude_alignment_timeout_enabled': (
                    prehook_attitude_alignment_timeout_s > 0.0
                ),
                'attitude_timeout_action': (
                    (
                        'latched_current_pose_hold_preserve_world_z_bias_'
                        'keep_offboard_heartbeat'
                    )
                    if prehook_attitude_alignment_timeout_s > 0.0
                    else 'continuous_nmpc_alignment_no_elapsed_timeout'
                ),
                'attitude_timeout_integral_semantics': (
                    (
                        'clear_xy_preserve_and_continue_bounded_world_z_bias'
                    )
                    if prehook_attitude_alignment_timeout_s > 0.0
                    else 'not_applicable_no_timeout_transition'
                ),
                'resolution_m': astar_resolution,
                'robot_radius_m': astar_robot_radius,
                'obstacle_margin_m': astar_obstacle_margin,
                'obstacle_inflation_m': obstacle_inflation,
                'static_obstacles_ned_xyxy': [
                    list(rectangle) for rectangle in static_obstacles
                ],
                'static_obstacle_semantics': 'no_contact_all_mission_phases',
                'fixed_hook_corridor_collision_validated': True,
                'bounds_semantics': (
                    'already_margin_reduced_center_feasible_ned'
                ),
                'failure_policy': 'revoke_mission_no_linear_fallback',
                'dynamic_obstacle_feed': False,
            },
            'fixed_hook_line_interlock': {
                'enabled': not fixed_hook_line_position_mode,
                'control_mode': (
                    'position_like_time_parameterized_ned'
                    if fixed_hook_line_position_mode
                    else 'measured_progress_interlock_compatibility'
                ),
                'active_states': (
                    []
                    if fixed_hook_line_position_mode
                    else ['GO_FORWARD', 'GO_BACK']
                ),
                'progress_source': (
                    'monotonic_trajectory_time'
                    if fixed_hook_line_position_mode
                    else 'measured_along_track_position'
                ),
                'progress_semantics': (
                    'direct_to_endpoint_no_freeze_no_rewind'
                    if fixed_hook_line_position_mode
                    else 'monotonic_furthest_measured_progress_no_reverse'
                ),
                'yaw_tolerance_deg': fixed_hook_line_yaw_tol_deg,
                'cross_track_tolerance_m': (
                    fixed_hook_line_cross_track_tol_m
                ),
                'release_ratio': (
                    fixed_hook_line_interlock_release_ratio
                ),
                'depth_tolerance_m': fixed_hook_depth_tolerance,
                'maximum_reference_lead_m': (
                    None
                    if fixed_hook_line_position_mode
                    else fixed_hook_line_max_reference_lead_m
                ),
                'maximum_reference_lead_semantics': (
                    'not_used_by_time_parameterized_position_mode'
                    if fixed_hook_line_position_mode
                    else (
                        'new_forward_advancement_only_'
                        'no_retreat_anchor_may_hold'
                    )
                ),
                'velocity_weight_multiplier': (
                    fixed_hook_line_velocity_weight_multiplier
                ),
                'velocity_weight_scope': (
                    'GO_FORWARD_GO_BACK_world_ned_velocity'
                ),
                'velocity_reference_frame': 'ned',
                'depth_velocity_reference_mps': 0.0,
                'interlocked_velocity_weight_multiplier': (
                    None if fixed_hook_line_position_mode else 1.0
                ),
                'interlocked_velocity_weight_semantics': (
                    'not_applicable_interlock_disabled'
                    if fixed_hook_line_position_mode
                    else 'base_weight_allows_cross_track_depth_yaw_recovery'
                ),
                'outside_tolerance_action': (
                    'continue_translation_and_correct_all_axes_with_nmpc'
                    if fixed_hook_line_position_mode
                    else (
                        'freeze_forward_lead_at_furthest_station_and_realign'
                    )
                ),
                'backslide_action': (
                    'time_reference_continues_toward_endpoint_never_rewinds'
                    if fixed_hook_line_position_mode
                    else 'hold_furthest_station_never_rewind_to_phase_start'
                ),
                'cross_track_integral_semantics': (
                    'bounded_line_normal_only_during_transit'
                ),
                'depth_integral_semantics': (
                    'bounded_world_z_updated_during_transit'
                ),
                'along_track_integral_semantics': (
                    'always_zero_during_transit'
                ),
                'reference_depth': 'recorded_hook_depth',
            },
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
            'operator_hook_confirmation': {
                'required': True,
                'input': 'dedicated_terminal_h_key',
                'service': HOOK_CONFIRMATION_SERVICE,
                'accepted_state': 'WAIT_HOOK',
                'next_state': 'GO_BACK',
                'automatic_timeout_s': None,
                'minimum_wait_after_state_entry_s': 0.25,
                'arrival_pose_latched_on_wait_entry': True,
                'reject_if_pose_outside_tolerance': False,
            },
            'final_pose_hold_s_scope': (
                'legacy_final_hold_only_not_dynamic_wait_hook'
            ),
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
            'raw_mocap_pose_topic': raw_pose_topic,
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
        f'depth tolerance={fixed_hook_depth_tolerance:.3f} m; attitude '
        f'reference={prehook_attitude_reference_mode}, pre-hook attitude '
        f'tolerance={prehook_reached_orientation_tol_deg:.1f}deg, '
        'body-X forward-axis tolerance='
        f'{prehook_reached_forward_axis_tol_deg:.1f}deg, '
        f'pre-hook yaw tolerance={prehook_reached_yaw_tol_deg:.1f}deg, '
        f'position/depth wait-exit hysteresis='
        f'{prehook_attitude_wait_exit_hysteresis_ratio:.2f}x. The '
        'controller will hold the common line depth before advancing along '
        'the recorded horizontal nose direction.'
    )
    mission_sequence_summary = (
        'Fixed-hook pose sequence: PLAN_TO_PREHOOK -> TRACK_TO_PREHOOK -> '
        f'PREHOOK_REACHED({prehook_reached_hold_s:.1f}s) -> GO_FORWARD -> '
        'WAIT_HOOK(operator H confirmation, no automatic timeout) -> '
        'straight GO_BACK -> '
        'COMPLETE; speeds: '
        f'pre={pre_approach_speed:.3f}m/s, '
        f'final={final_approach_speed:.3f}m/s, '
        f'retreat={retreat_speed_mps:.3f}m/s.'
    )
    if fixed_hook_line_position_mode:
        line_interlock_summary = (
            'Fixed-hook Position-like translation: GO_FORWARD/GO_BACK use '
            'a monotonic time-parameterized horizontal NED reference directly '
            'to the endpoint. Ordinary yaw/cross-track/depth error does not '
            'freeze, brake, or rewind forward progress; NMPC corrects all '
            'axes concurrently. NED depth and depth-velocity references stay '
            f'constant, with {fixed_hook_line_velocity_weight_multiplier:.1f}x '
            'world-velocity weight throughout the line. Final position, '
            'depth, and attitude acceptance gates remain active.'
        )
    else:
        line_interlock_summary = (
            'Fixed-hook compatibility interlock: GO_FORWARD/GO_BACK progress '
            'is driven by measured along-track position; '
            f'yaw tolerance={fixed_hook_line_yaw_tol_deg:.1f}deg, '
            'cross-track tolerance='
            f'{fixed_hook_line_cross_track_tol_m:.3f}m, '
            f'depth tolerance={fixed_hook_depth_tolerance:.3f}m, '
            'release ratio='
            f'{fixed_hook_line_interlock_release_ratio:.2f}x, maximum lead='
            f'{fixed_hook_line_max_reference_lead_m:.3f}m.'
        )
    obstacle_scope = (
        f'{len(static_obstacles)} configured static real-NED rectangle(s)'
        if static_obstacles
        else 'pool walls only (no internal obstacle geometry configured)'
    )
    planner_summary = (
        f'Real pre-hook A*: {planner_check_rate_hz:.1f}Hz safety/deviation '
        f'checks, resolution={astar_resolution:.2f}m, '
        f'inflation={obstacle_inflation:.2f}m, map={obstacle_scope}; '
        'failure is latched with no straight-line fallback. A* stops '
        'permanently after PREHOOK_REACHED.'
    )
    attitude_alignment_summary = (
        'Pre-hook attitude alignment: no elapsed-time timeout; NMPC keeps '
        'aligning until the '
        f'{prehook_reached_orientation_tol_deg:.1f}-degree full-attitude '
        'safety envelope, '
        f'{prehook_reached_forward_axis_tol_deg:.1f}-degree body-X '
        'forward-axis gate, and '
        f'{prehook_reached_yaw_tol_deg:.1f}-degree yaw gate are '
        'satisfied. Mission/odom/Offboard/solver safety gates remain active.'
        if prehook_attitude_alignment_timeout_s <= 0.0
        else (
            'Pre-hook attitude alignment timeout: '
            f'{prehook_attitude_alignment_timeout_s:.1f}s.'
        )
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
        LogInfo(msg=line_interlock_summary),
        LogInfo(msg=planner_summary),
        LogInfo(msg=attitude_alignment_summary),
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
            default_value='glub',
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
            default_value='unconfigured',
            description=(
                'Required validated glub target recorded directly from '
                '/mocap/glub/pose at the hooked pose. The former glub_fb '
                'recording is retained as provenance and is not relabelled.'
            ),
        ),
        DeclareLaunchArgument(
            'allow_identity_derived_target',
            default_value='false',
            description=(
                'Allow only a hash-locked, exact identity pose transfer from '
                'another raw MoCap rigid body. Disabled by default.'
            ),
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
            'prehook_planner_check_rate_hz',
            default_value='2.0',
            description=(
                'Low-rate path safety/deviation checker; valid range 2-5 Hz.'
            ),
        ),
        DeclareLaunchArgument(
            'prehook_replan_deviation_m',
            default_value='0.30',
            description='Cross-track error that may trigger a new A* path.',
        ),
        DeclareLaunchArgument(
            'prehook_replan_deviation_hold_s',
            default_value='0.50',
            description='Time the cross-track threshold must remain exceeded.',
        ),
        DeclareLaunchArgument(
            'prehook_replan_min_switch_interval_s',
            default_value='1.0',
            description='Replan switching cooldown to prevent path chatter.',
        ),
        DeclareLaunchArgument(
            'prehook_replan_min_improvement_m',
            default_value='0.15',
            description='Absolute path shortening required for optional switch.',
        ),
        DeclareLaunchArgument(
            'prehook_replan_min_improvement_ratio',
            default_value='0.10',
            description='Relative path shortening required for optional switch.',
        ),
        DeclareLaunchArgument(
            'prehook_replan_optimization_period_s',
            default_value='2.0',
            description=(
                'Period for checking whether a materially shorter path exists; '
                '0 disables optimization-only searches.'
            ),
        ),
        DeclareLaunchArgument(
            'prehook_reached_hold_s',
            default_value='1.0',
            description=(
                'Continuous position/depth/attitude dwell before GO_FORWARD.'
            ),
        ),
        DeclareLaunchArgument(
            'prehook_attitude_reference_mode',
            default_value='recorded_hook',
            description=(
                'Pre-hook attitude reference: recorded_hook preserves the '
                'legacy full Hook attitude; capture_start_trim captures the '
                'free-floating start roll/pitch while retaining target yaw.'
            ),
        ),
        DeclareLaunchArgument(
            'prehook_reached_orientation_tol_deg',
            default_value='5.0',
            description=(
                'Full attitude tolerance at PREHOOK_REACHED in degrees '
                '(1-20). Hook and WAIT_HOOK retain the separate 5-degree '
                'goal tolerance.'
            ),
        ),
        DeclareLaunchArgument(
            'prehook_reached_forward_axis_tol_deg',
            default_value='5.0',
            description=(
                'Body-X (nose) forward-axis angular tolerance at '
                'PREHOOK_REACHED in degrees (0.5-20). This remains '
                'independent when the full-attitude safety envelope is '
                'widened for a measured free-floating roll trim.'
            ),
        ),
        DeclareLaunchArgument(
            'prehook_reached_yaw_tol_deg',
            default_value='3.0',
            description=(
                'Independent horizontal-heading tolerance before GO_FORWARD '
                'in degrees (0.5-20). This stays strict even when the full '
                'Splash trim-attitude tolerance is wider.'
            ),
        ),
        DeclareLaunchArgument(
            'fixed_hook_line_position_mode',
            default_value='true',
            description=(
                'Use monotonic time-parameterized NED Position-like '
                'translation for GO_FORWARD/GO_BACK. Ordinary corridor '
                'error is corrected without freezing, braking, or rewinding.'
            ),
        ),
        DeclareLaunchArgument(
            'fixed_hook_line_yaw_tol_deg',
            default_value='3.0',
            description=(
                'Compatibility-interlock heading threshold when '
                'fixed_hook_line_position_mode=false (0.5-20 degrees).'
            ),
        ),
        DeclareLaunchArgument(
            'fixed_hook_line_cross_track_tol_m',
            default_value='0.03',
            description=(
                'Compatibility-interlock horizontal cross-track threshold '
                'when fixed_hook_line_position_mode=false (0.005-0.20 m).'
            ),
        ),
        DeclareLaunchArgument(
            'fixed_hook_line_interlock_release_ratio',
            default_value='0.8',
            description=(
                'Compatibility fraction of the yaw, cross-track, and depth '
                'thresholds that must be recovered before a frozen '
                'GO_FORWARD/GO_BACK interlock releases (0-1, exclusive '
                'of 0).'
            ),
        ),
        DeclareLaunchArgument(
            'fixed_hook_line_max_reference_lead_m',
            default_value='0.02',
            description=(
                'Compatibility maximum along-track advancement beyond measured '
                'progress during GO_FORWARD/GO_BACK. After a backslide, the '
                'no-retreat anchor may remain farther ahead while the '
                'vehicle catches up (0.005-0.20 m).'
            ),
        ),
        DeclareLaunchArgument(
            'fixed_hook_line_velocity_weight_multiplier',
            default_value='20.0',
            description=(
                'Multiplier for the NMPC NED/world linear-velocity error '
                'during GO_FORWARD/GO_BACK (1-50). Position-like mode keeps '
                'this weight constant, including for zero depth velocity.'
            ),
        ),
        DeclareLaunchArgument(
            'prehook_attitude_wait_exit_hysteresis_ratio',
            default_value='1.5',
            description=(
                'Ratio applied to the pre-hook position/depth tolerances '
                'before leaving attitude-only wait, preventing gate chatter '
                'while attitude settles (1-3).'
            ),
        ),
        DeclareLaunchArgument(
            'prehook_attitude_alignment_timeout_s',
            default_value='0.0',
            description=(
                'Optional timeout after the pre-hook position/depth gates '
                'are continuously satisfied but attitude is not. Zero '
                'disables elapsed-time timeout and keeps NMPC aligning; a '
                'positive value latches the guarded current-pose hold after '
                'that duration. Timing starts after the reference finishes.'
            ),
        ),
        DeclareLaunchArgument(
            'prehook_static_obstacles_ned_xyxy',
            default_value='',
            description=(
                'Optional semicolon-separated real-NED no-contact obstacle '
                'rectangles: xmin xmax ymin ymax. They must not intersect '
                'the fixed hook corridor. Empty means pool walls only.'
            ),
        ),
        DeclareLaunchArgument(
            'prehook_astar_resolution_m', default_value='0.10'
        ),
        DeclareLaunchArgument(
            'prehook_robot_radius_m',
            default_value='0.20',
            description=(
                'Horizontal collision radius of the real vehicle plus hook.'
            ),
        ),
        DeclareLaunchArgument(
            'prehook_obstacle_margin_m',
            default_value='0.05',
            description=(
                'Extra XY obstacle clearance; robot radius plus this value '
                'must not exceed pool_safety_margin_m.'
            ),
        ),
        DeclareLaunchArgument(
            'prehook_path_smoothing_iterations', default_value='1'
        ),
        DeclareLaunchArgument(
            'prehook_path_smoothing_corner_fraction', default_value='0.20'
        ),
        DeclareLaunchArgument(
            'prehook_path_smoothing_samples_per_corner', default_value='4'
        ),
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
                'Legacy FINAL_HOLD duration. The real dynamic WAIT_HOOK '
                'stage ignores this timer and requires operator H '
                'confirmation before retreating.'
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
                'Bounded offset-free position gain. Normal trajectories '
                'enable it after the reference finishes; fixed Hook '
                'transit additionally updates its line-normal cross-current '
                'and world-Z depth components while forcing the along-line '
                'component to zero.'
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
