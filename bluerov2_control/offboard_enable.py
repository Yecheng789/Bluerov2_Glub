"""Safely request PX4 Offboard mode for wrench control."""

import math
import time

from px4_msgs.msg import OffboardControlMode, VehicleCommand
from px4_msgs.msg import VehicleCommandAck, VehicleControlMode
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool, Empty


def versioned_px4_topic(base_topic, message_type):
    """Append the PX4 message-version suffix required by the built type."""
    version = int(getattr(message_type, 'MESSAGE_VERSION', 0))
    if version < 0:
        raise ValueError('PX4 MESSAGE_VERSION must be non-negative')
    return f'{base_topic}_v{version}' if version > 0 else base_topic


DEFAULT_VEHICLE_COMMAND_ACK_TOPIC = versioned_px4_topic(
    '/fmu/out/vehicle_command_ack', VehicleCommandAck
)
DEFAULT_CONTROL_MODE_TIMEOUT_SEC = 1.25


def _build_ack_result_names(message_type):
    names = {}
    for attribute, label in (
        ('VEHICLE_CMD_RESULT_ACCEPTED', 'ACCEPTED'),
        ('VEHICLE_CMD_RESULT_TEMPORARILY_REJECTED',
         'TEMPORARILY_REJECTED'),
        ('VEHICLE_CMD_RESULT_DENIED', 'DENIED'),
        ('VEHICLE_CMD_RESULT_UNSUPPORTED', 'UNSUPPORTED'),
        ('VEHICLE_CMD_RESULT_FAILED', 'FAILED'),
        ('VEHICLE_CMD_RESULT_IN_PROGRESS', 'IN_PROGRESS'),
        ('VEHICLE_CMD_RESULT_CANCELLED', 'CANCELLED'),
        ('VEHICLE_CMD_RESULT_COMMAND_LONG_ONLY', 'COMMAND_LONG_ONLY'),
        ('VEHICLE_CMD_RESULT_COMMAND_INT_ONLY', 'COMMAND_INT_ONLY'),
        ('VEHICLE_CMD_RESULT_UNSUPPORTED_MAV_FRAME',
         'UNSUPPORTED_MAV_FRAME'),
    ):
        value = getattr(message_type, attribute, None)
        if value is not None:
            names[int(value)] = label
    return names


ACK_RESULT_NAMES = _build_ack_result_names(VehicleCommandAck)


class OffboardEnableNode(Node):
    """Prestream a wrench heartbeat and request Offboard with ACK handling."""

    HEARTBEAT_PERIOD_SEC = 0.1

    def __init__(self):
        """Declare parameters and create the PX4 command state machine."""
        super().__init__('offboard_enable')

        self.declare_parameter(
            'offboard_mode_topic', '/fmu/in/offboard_control_mode')
        self.declare_parameter('vehicle_cmd_topic', '/fmu/in/vehicle_command')
        self.declare_parameter(
            'vehicle_cmd_ack_topic', DEFAULT_VEHICLE_COMMAND_ACK_TOPIC)
        self.declare_parameter(
            'vehicle_control_mode_topic', '/fmu/out/vehicle_control_mode')
        self.declare_parameter(
            'request_enable_topic', '/bluerov2/offboard_request_enable')
        self.declare_parameter('require_request_enable', False)
        self.declare_parameter(
            'controller_heartbeat_topic', '/bluerov2/controller_heartbeat')
        self.declare_parameter('require_controller_heartbeat', False)
        self.declare_parameter(
            'enforce_control_publisher_exclusivity', False)
        self.declare_parameter('expected_controller_node_fqn', '')
        self.declare_parameter(
            'exclusive_thrust_sp_topic', '/fmu/in/vehicle_thrust_setpoint')
        self.declare_parameter(
            'exclusive_torque_sp_topic', '/fmu/in/vehicle_torque_setpoint')
        self.declare_parameter('auto_arm', False)
        self.declare_parameter('target_system_id', 1)
        self.declare_parameter('target_component_id', 1)
        self.declare_parameter('source_system_id', 1)
        self.declare_parameter('source_component_id', 191)
        self.declare_parameter('prestream_duration_sec', 2.0)
        self.declare_parameter('command_retry_interval_sec', 1.0)
        self.declare_parameter(
            'control_mode_timeout_sec', DEFAULT_CONTROL_MODE_TIMEOUT_SEC)
        self.declare_parameter('controller_heartbeat_timeout_sec', 0.3)

        self.offboard_mode_topic = self.get_parameter(
            'offboard_mode_topic').value
        self.vehicle_cmd_topic = self.get_parameter('vehicle_cmd_topic').value
        self.vehicle_cmd_ack_topic = self.get_parameter(
            'vehicle_cmd_ack_topic').value
        self.vehicle_control_mode_topic = self.get_parameter(
            'vehicle_control_mode_topic').value
        self.request_enable_topic = self.get_parameter(
            'request_enable_topic').value
        self.require_request_enable = bool(
            self.get_parameter('require_request_enable').value)
        self.controller_heartbeat_topic = self.get_parameter(
            'controller_heartbeat_topic').value
        self.require_controller_heartbeat = bool(
            self.get_parameter('require_controller_heartbeat').value)
        self.enforce_control_publisher_exclusivity = bool(
            self.get_parameter(
                'enforce_control_publisher_exclusivity').value)
        self.expected_controller_node_fqn = str(
            self.get_parameter('expected_controller_node_fqn').value
        ).strip()
        self.exclusive_thrust_sp_topic = str(
            self.get_parameter('exclusive_thrust_sp_topic').value
        ).strip()
        self.exclusive_torque_sp_topic = str(
            self.get_parameter('exclusive_torque_sp_topic').value
        ).strip()
        self.auto_arm = bool(self.get_parameter('auto_arm').value)

        if self.enforce_control_publisher_exclusivity:
            if not self.require_request_enable:
                raise ValueError(
                    'publisher exclusivity requires '
                    'require_request_enable=true'
                )
            if not self.expected_controller_node_fqn.startswith('/'):
                raise ValueError(
                    'expected_controller_node_fqn must be an absolute '
                    'node name'
                )
            if (
                not self.exclusive_thrust_sp_topic
                or not self.exclusive_torque_sp_topic
            ):
                raise ValueError(
                    'exclusive thrust/torque topics must not be empty'
                )

        self.target_system_id = self._read_id_parameter(
            'target_system_id', 1, 255)
        self.target_component_id = self._read_id_parameter(
            'target_component_id', 1, 255)
        self.source_system_id = self._read_id_parameter(
            'source_system_id', 1, 255)
        self.source_component_id = self._read_id_parameter(
            'source_component_id', 1, 65535)
        self.prestream_duration_sec = self._read_float_parameter(
            'prestream_duration_sec', minimum=1.0)
        self.command_retry_interval_sec = self._read_float_parameter(
            'command_retry_interval_sec', minimum=0.2)
        self.control_mode_timeout_sec = self._read_float_parameter(
            'control_mode_timeout_sec', minimum=0.1)
        self.controller_heartbeat_timeout_sec = self._read_float_parameter(
            'controller_heartbeat_timeout_sec', minimum=0.1)

        self.offboard_pub = self.create_publisher(
            OffboardControlMode,
            self.offboard_mode_topic,
            qos_profile_sensor_data,
        )
        self.cmd_pub = self.create_publisher(
            VehicleCommand,
            self.vehicle_cmd_topic,
            qos_profile_sensor_data,
        )
        self.ack_sub = self.create_subscription(
            VehicleCommandAck,
            self.vehicle_cmd_ack_topic,
            self.vehicle_command_ack_callback,
            qos_profile_sensor_data,
        )
        self.control_mode_sub = self.create_subscription(
            VehicleControlMode,
            self.vehicle_control_mode_topic,
            self.vehicle_control_mode_callback,
            qos_profile_sensor_data,
        )
        self.request_enable_sub = self.create_subscription(
            Bool,
            self.request_enable_topic,
            self.request_enable_callback,
            10,
        )
        self.controller_heartbeat_sub = self.create_subscription(
            Empty,
            self.controller_heartbeat_topic,
            self.controller_heartbeat_callback,
            10,
        )

        self.first_heartbeat_time = None
        self.last_offboard_command_time = None
        self.last_arm_command_time = None
        self.offboard_attempts = 0
        self.arm_attempts = 0
        self.offboard_ack_accepted = False
        self.offboard_active = False
        self.vehicle_armed = False
        self.last_control_mode_time = None
        self.last_controller_heartbeat_time = None
        self.request_enabled = not self.require_request_enable
        self.controller_rearm_required = False
        self.arm_accepted = False
        self.manual_arm_notice_logged = False

        self.timer = self.create_timer(
            self.HEARTBEAT_PERIOD_SEC, self.timer_callback)

        arm_policy = 'enabled' if self.auto_arm else 'disabled (safe default)'
        self.get_logger().info(
            'offboard_enable started: heartbeat=10 Hz, auto_arm=%s, '
            'target=%d/%d, source=%d/%d, ACK=%s, state=%s'
            % (
                arm_policy,
                self.target_system_id,
                self.target_component_id,
                self.source_system_id,
                self.source_component_id,
                self.vehicle_cmd_ack_topic,
                self.vehicle_control_mode_topic,
            )
        )
        if self.require_request_enable:
            self.get_logger().info(
                'OFFBOARD requests are disabled; publish std_msgs/Bool true '
                'on %s after completing preflight checks'
                % self.request_enable_topic
            )
        if self.require_controller_heartbeat:
            self.get_logger().info(
                'PX4 heartbeat is gated by controller liveness on %s '
                '(timeout %.2f s)'
                % (
                    self.controller_heartbeat_topic,
                    self.controller_heartbeat_timeout_sec,
                )
            )
        if self.enforce_control_publisher_exclusivity:
            self.get_logger().info(
                'Control publisher exclusivity is enforced: manager=%s, '
                'controller=%s'
                % (
                    self.get_fully_qualified_name(),
                    self.expected_controller_node_fqn,
                )
            )

    def _read_id_parameter(self, name, minimum, maximum):
        value = self.get_parameter(name).value
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError('%s must be an integer' % name)
        if not minimum <= value <= maximum:
            raise ValueError(
                '%s must be in [%d, %d]' % (name, minimum, maximum))
        return value

    def _read_float_parameter(self, name, minimum):
        value = float(self.get_parameter(name).value)
        if not math.isfinite(value) or value < minimum:
            raise ValueError('%s must be finite and >= %.1f' % (name, minimum))
        return value

    def publish_offboard_control_mode(self):
        """Publish the 10 Hz PX4 wrench-control heartbeat."""
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.thrust_and_torque = True
        msg.direct_actuator = False
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.offboard_pub.publish(msg)

    def publish_vehicle_command(
            self, command, param1=0.0, param2=0.0, confirmation=0):
        """Publish one addressed PX4 vehicle command."""
        msg = VehicleCommand()
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.command = int(command)
        msg.target_system = self.target_system_id
        msg.target_component = self.target_component_id
        msg.source_system = self.source_system_id
        msg.source_component = self.source_component_id
        msg.confirmation = int(confirmation)
        msg.from_external = True
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.cmd_pub.publish(msg)

    def _command_is_due(self, now, last_command_time):
        return (
            last_command_time is None
            or now - last_command_time >= self.command_retry_interval_sec
        )

    def _request_offboard(self, now):
        confirmation = min(self.offboard_attempts, 255)
        self.offboard_attempts += 1
        self.last_offboard_command_time = now
        action = 'Requesting' if self.offboard_attempts == 1 else 'Retrying'
        self.get_logger().info(
            '%s OFFBOARD mode (attempt %d); waiting for matching ACK'
            % (action, self.offboard_attempts)
        )
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,
            param2=6.0,
            confirmation=confirmation,
        )

    def _request_arm(self, now):
        confirmation = min(self.arm_attempts, 255)
        self.arm_attempts += 1
        self.last_arm_command_time = now
        action = 'Requesting' if self.arm_attempts == 1 else 'Retrying'
        self.get_logger().warning(
            '%s ARM (attempt %d); auto_arm was explicitly enabled'
            % (action, self.arm_attempts)
        )
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=1.0,
            confirmation=confirmation,
        )

    def timer_callback(self):
        """Publish heartbeat and advance the ACK-gated command state."""
        now = time.monotonic()

        if self.enforce_control_publisher_exclusivity:
            valid, hard_violation, detail = (
                self._control_publishers_are_exclusive()
            )
            if not valid:
                self._withdraw_heartbeat_for_health_failure(
                    detail, force_rearm=hard_violation
                )
                return

        if not self._controller_is_alive(now):
            self._withdraw_heartbeat_for_health_failure(
                'Controller heartbeat is missing or stale'
            )
            return

        if not self._control_mode_feedback_is_fresh(now):
            self._withdraw_heartbeat_for_health_failure(
                'PX4 VehicleControlMode feedback is missing or stale'
            )
            return

        self.publish_offboard_control_mode()

        if self.first_heartbeat_time is None:
            self.first_heartbeat_time = now
            self.get_logger().info(
                'Prestreaming Offboard heartbeat for %.1f s before requesting '
                'mode change' % self.prestream_duration_sec
            )

        if now - self.first_heartbeat_time < self.prestream_duration_sec:
            return

        if not self.request_enabled:
            return

        if not self.offboard_active:
            if self._command_is_due(now, self.last_offboard_command_time):
                self._request_offboard(now)
            return

        if not self.auto_arm:
            if not self.manual_arm_notice_logged:
                self.get_logger().info(
                    'OFFBOARD is active; auto_arm=false, so no ARM command '
                    'will be sent'
                )
                self.manual_arm_notice_logged = True
            return

        if self.vehicle_armed:
            self.arm_accepted = True
            return

        if (
            self.offboard_ack_accepted
            and not self.arm_accepted
            and self._command_is_due(
                now, self.last_arm_command_time)
        ):
            self._request_arm(now)

    def _control_mode_feedback_is_fresh(self, now):
        if self.last_control_mode_time is None:
            return False
        age_sec = now - self.last_control_mode_time
        return 0.0 <= age_sec <= self.control_mode_timeout_sec

    @staticmethod
    def _endpoint_node_fqn(endpoint):
        namespace = str(endpoint.node_namespace or '/').strip()
        namespace = '/' + namespace.strip('/') if namespace != '/' else ''
        return f'{namespace}/{endpoint.node_name}'

    def _control_publishers_are_exclusive(self):
        expected_by_topic = {
            self.offboard_mode_topic: self.get_fully_qualified_name(),
            self.vehicle_cmd_topic: self.get_fully_qualified_name(),
            self.exclusive_thrust_sp_topic:
                self.expected_controller_node_fqn,
            self.exclusive_torque_sp_topic:
                self.expected_controller_node_fqn,
        }
        missing = []
        violations = []
        for topic, expected_fqn in expected_by_topic.items():
            try:
                endpoints = self.get_publishers_info_by_topic(topic)
            except Exception as exc:  # graph failure must fail closed
                missing.append(f'{topic} (graph error: {exc})')
                continue
            actual_fqns = [
                self._endpoint_node_fqn(endpoint)
                for endpoint in endpoints
            ]
            if not actual_fqns:
                missing.append(topic)
            elif len(actual_fqns) != 1 or actual_fqns[0] != expected_fqn:
                violations.append(
                    f'{topic}: expected [{expected_fqn}], got {actual_fqns}'
                )

        if violations:
            return (
                False,
                True,
                'Control publisher exclusivity violation: '
                + '; '.join(violations),
            )
        if missing:
            return (
                False,
                False,
                'Required control publishers are not discoverable: '
                + ', '.join(missing),
            )
        return True, False, ''

    def _withdraw_heartbeat_for_health_failure(
            self, reason, force_rearm=False):
        rearm_latched = False
        if (
            self.require_request_enable
            and (
                force_rearm
                or self.request_enabled
                or self.offboard_active
            )
        ):
            self.request_enabled = False
            self.controller_rearm_required = True
            self.offboard_ack_accepted = False
            self.arm_accepted = False
            self.manual_arm_notice_logged = False
            rearm_latched = True
        self.first_heartbeat_time = None
        if rearm_latched:
            recovery = (
                'After recovery, publish request false and complete '
                'preflight again before publishing true.'
            )
        else:
            recovery = 'Waiting for healthy preflight feedback.'
        self.get_logger().error(
            f'{reason}; PX4 Offboard heartbeat is withdrawn. {recovery}',
            throttle_duration_sec=1.0,
        )

    def _controller_is_alive(self, now):
        if not self.require_controller_heartbeat:
            return True
        if self.last_controller_heartbeat_time is None:
            return False
        age_sec = now - self.last_controller_heartbeat_time
        return 0.0 <= age_sec <= self.controller_heartbeat_timeout_sec

    def controller_heartbeat_callback(self, _msg):
        """Record a pulse generated by the controller executor."""
        self.last_controller_heartbeat_time = time.monotonic()

    def request_enable_callback(self, msg):
        """Apply the operator-owned gate for issuing mode commands."""
        requested = bool(msg.data)
        if not requested:
            was_enabled = self.request_enabled
            self.request_enabled = False
            if self.controller_rearm_required:
                self.controller_rearm_required = False
                self.get_logger().info(
                    'Controller-health request latch cleared; inspect the '
                    'fault and publish true only after preflight passes'
                )
            elif was_enabled:
                self.get_logger().warning(
                    'Operator disabled further OFFBOARD requests. This does '
                    'not change a mode that is already active; use QGC to '
                    'leave OFFBOARD if required.'
                )
            return

        if self.controller_rearm_required:
            self.request_enabled = False
            self.get_logger().warning(
                'OFFBOARD request rejected by the controller-health latch; '
                'publish false, inspect the fault, then publish true',
                throttle_duration_sec=1.0,
            )
            return
        if self.request_enabled:
            return
        self.request_enabled = True
        self.last_offboard_command_time = None
        self.get_logger().info(
            'Operator enabled OFFBOARD requests; the next request will be '
            'sent only after controller and PX4 feedback preflight passes'
        )

    def vehicle_control_mode_callback(self, msg):
        """Track the actual, fresh PX4 mode instead of trusting ACK alone."""
        was_active = self.offboard_active
        self.last_control_mode_time = time.monotonic()
        self.offboard_active = bool(msg.flag_control_offboard_enabled)
        self.vehicle_armed = bool(msg.flag_armed)

        if self.offboard_active and not was_active:
            self.get_logger().info(
                'PX4 VehicleControlMode confirms OFFBOARD is active'
            )
        elif was_active and not self.offboard_active:
            self.offboard_ack_accepted = False
            self.arm_accepted = False
            self.manual_arm_notice_logged = False
            self.get_logger().warning(
                'PX4 reports OFFBOARD is no longer active; commands will be '
                'retried only while the operator request gate remains enabled'
            )

    def _ack_is_for_this_node(self, msg):
        return (
            int(msg.target_system) == self.source_system_id
            and int(msg.target_component) == self.source_component_id
        )

    def vehicle_command_ack_callback(self, msg):
        """Process ACKs addressed to this command source."""
        command = int(msg.command)
        if command not in (
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM):
            return

        if not self._ack_is_for_this_node(msg):
            self.get_logger().debug(
                'Ignoring command %d ACK addressed to %d/%d (expected %d/%d)'
                % (
                    command,
                    int(msg.target_system),
                    int(msg.target_component),
                    self.source_system_id,
                    self.source_component_id,
                )
            )
            return

        if (command == VehicleCommand.VEHICLE_CMD_DO_SET_MODE
                and self.offboard_attempts == 0):
            return
        if (command == VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
                and self.arm_attempts == 0):
            return

        result = int(msg.result)
        result_name = ACK_RESULT_NAMES.get(result, 'UNKNOWN_%d' % result)
        command_name = (
            'OFFBOARD'
            if command == VehicleCommand.VEHICLE_CMD_DO_SET_MODE
            else 'ARM'
        )
        detail = 'result_param1=%d, result_param2=%d' % (
            int(msg.result_param1), int(msg.result_param2))

        if result == VehicleCommandAck.VEHICLE_CMD_RESULT_ACCEPTED:
            self.get_logger().info(
                '%s ACK: %s (%s)' % (command_name, result_name, detail))
            if command == VehicleCommand.VEHICLE_CMD_DO_SET_MODE:
                self.offboard_ack_accepted = True
            else:
                self.arm_accepted = True
            return

        self.get_logger().warning(
            '%s ACK: %s (%s); command will be retried in %.1f s'
            % (
                command_name,
                result_name,
                detail,
                self.command_retry_interval_sec,
            )
        )
        if command == VehicleCommand.VEHICLE_CMD_DO_SET_MODE:
            self.offboard_ack_accepted = False


def main(args=None):
    """Run the Offboard enabler node."""
    rclpy.init(args=args)
    node = OffboardEnableNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
