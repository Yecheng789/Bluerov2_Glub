"""Launch the ACK-gated PX4 Offboard enabler for the real vehicle."""

import math

from bluerov2_control.offboard_enable import versioned_px4_topic
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
from px4_msgs.msg import VehicleCommandAck


def _parse_int_argument(context, name, minimum, maximum):
    text = LaunchConfiguration(name).perform(context).strip()
    try:
        value = int(text)
    except ValueError as exc:
        raise RuntimeError('%s must be an integer' % name) from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(
            '%s must be in [%d, %d]' % (name, minimum, maximum))
    return value


def _parse_float_argument(context, name, minimum):
    text = LaunchConfiguration(name).perform(context).strip()
    try:
        value = float(text)
    except ValueError as exc:
        raise RuntimeError('%s must be a number' % name) from exc
    if not math.isfinite(value) or value < minimum:
        raise RuntimeError('%s must be finite and >= %.1f' % (name, minimum))
    return value


def _parse_bool_argument(context, name):
    text = LaunchConfiguration(name).perform(context).strip().lower()
    if text == 'true':
        return True
    if text == 'false':
        return False
    raise RuntimeError('%s must be either true or false' % name)


def _normalise_namespace(context):
    namespace = LaunchConfiguration('robot_namespace').perform(context).strip()
    if not namespace:
        raise RuntimeError('robot_namespace must not be empty')
    components = namespace.strip('/').split('/')
    if not all(component and component.replace('_', '').isalnum()
               for component in components):
        raise RuntimeError(
            'robot_namespace must contain only ROS name components')
    return '/' + '/'.join(components)


def _launch_setup(context):
    namespace = _normalise_namespace(context)
    auto_arm = _parse_bool_argument(context, 'auto_arm')
    require_request_enable = _parse_bool_argument(
        context, 'require_request_enable')
    enforce_control_publisher_exclusivity = _parse_bool_argument(
        context, 'enforce_control_publisher_exclusivity')
    request_enable_topic = LaunchConfiguration(
        'request_enable_topic').perform(context).strip()
    if not request_enable_topic:
        raise RuntimeError('request_enable_topic must not be empty')
    if auto_arm:
        print(
            'WARNING: auto_arm=true was explicitly requested; the node will '
            'ARM only after an ACCEPTED OFFBOARD ACK.'
        )

    parameters = {
        'offboard_mode_topic': namespace + '/fmu/in/offboard_control_mode',
        'vehicle_cmd_topic': namespace + '/fmu/in/vehicle_command',
        'vehicle_cmd_ack_topic': versioned_px4_topic(
            namespace + '/fmu/out/vehicle_command_ack', VehicleCommandAck),
        'vehicle_control_mode_topic': (
            namespace + '/fmu/out/vehicle_control_mode'),
        'request_enable_topic': request_enable_topic,
        'require_request_enable': require_request_enable,
        'enforce_control_publisher_exclusivity': (
            enforce_control_publisher_exclusivity),
        'expected_controller_node_fqn': LaunchConfiguration(
            'expected_controller_node_fqn').perform(context).strip(),
        'exclusive_thrust_sp_topic': (
            namespace + '/fmu/in/vehicle_thrust_setpoint'),
        'exclusive_torque_sp_topic': (
            namespace + '/fmu/in/vehicle_torque_setpoint'),
        'auto_arm': auto_arm,
        'target_system_id': _parse_int_argument(
            context, 'target_system_id', 1, 255),
        'target_component_id': _parse_int_argument(
            context, 'target_component_id', 1, 255),
        'source_system_id': _parse_int_argument(
            context, 'source_system_id', 1, 255),
        'source_component_id': _parse_int_argument(
            context, 'source_component_id', 1, 65535),
        'prestream_duration_sec': _parse_float_argument(
            context, 'prestream_duration_sec', 1.0),
        'command_retry_interval_sec': _parse_float_argument(
            context, 'command_retry_interval_sec', 0.2),
        'control_mode_timeout_sec': _parse_float_argument(
            context, 'control_mode_timeout_sec', 0.1),
    }

    return [
        Node(
            package='bluerov2_control',
            executable='offboard_enable',
            name='offboard_enable_real',
            output='screen',
            emulate_tty=True,
            parameters=[parameters],
        )
    ]


def generate_launch_description():
    """Build the safely parameterized real-vehicle launch description."""
    return LaunchDescription([
        DeclareLaunchArgument(
            'robot_namespace',
            default_value='/glub',
            description='Absolute PX4 ROS namespace.'),
        DeclareLaunchArgument(
            'auto_arm',
            default_value='false',
            description='Explicitly allow ARM after OFFBOARD is accepted.'),
        DeclareLaunchArgument(
            'request_enable_topic',
            default_value='/bluerov2/offboard_request_enable',
            description='Operator gate topic used when the gate is required.'),
        DeclareLaunchArgument(
            'require_request_enable',
            default_value='false',
            description='Require Bool true before issuing OFFBOARD commands.'),
        DeclareLaunchArgument(
            'enforce_control_publisher_exclusivity',
            default_value='false',
            description=(
                'Fail closed unless each wrench-control topic has exactly '
                'its expected publisher.')),
        DeclareLaunchArgument(
            'expected_controller_node_fqn',
            default_value='',
            description=(
                'Absolute expected MPC node name when exclusivity is '
                'enabled.')),
        DeclareLaunchArgument(
            'target_system_id',
            default_value='1',
            description='PX4 MAVLink system ID (non-broadcast).'),
        DeclareLaunchArgument(
            'target_component_id',
            default_value='1',
            description='PX4 MAVLink component ID (non-broadcast).'),
        DeclareLaunchArgument(
            'source_system_id',
            default_value='1',
            description='External command source system ID.'),
        DeclareLaunchArgument(
            'source_component_id',
            default_value='191',
            description='External command source component ID.'),
        DeclareLaunchArgument(
            'prestream_duration_sec',
            default_value='2.0',
            description='Heartbeat prestream time; minimum 1.0 seconds.'),
        DeclareLaunchArgument(
            'command_retry_interval_sec',
            default_value='1.0',
            description='OFFBOARD/ARM retry interval; minimum 0.2 seconds.'),
        DeclareLaunchArgument(
            'control_mode_timeout_sec',
            default_value='1.25',
            description='Maximum age of actual PX4 mode feedback.'),
        OpaqueFunction(function=_launch_setup),
    ])
