import rclpy
from rclpy.node import Node

from px4_msgs.msg import OffboardControlMode, VehicleCommand


class OffboardEnableNode(Node):
    def __init__(self):
        super().__init__('offboard_enable')

        self.declare_parameter('offboard_mode_topic', '/fmu/in/offboard_control_mode')
        self.declare_parameter('vehicle_cmd_topic', '/fmu/in/vehicle_command')
        self.declare_parameter('auto_arm', True)

        self.offboard_mode_topic = self.get_parameter('offboard_mode_topic').value
        self.vehicle_cmd_topic = self.get_parameter('vehicle_cmd_topic').value

        self.offboard_pub = self.create_publisher(
            OffboardControlMode,
            self.offboard_mode_topic,
            10
        )
        self.cmd_pub = self.create_publisher(
            VehicleCommand,
            self.vehicle_cmd_topic,
            10
        )

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.counter = 0
        self.sent_offboard = False
        self.sent_arm = False

        self.get_logger().info('offboard_enable node started')

    def publish_offboard_control_mode(self):
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

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.command = int(command)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.cmd_pub.publish(msg)

    def timer_callback(self):
        self.publish_offboard_control_mode()
        self.counter += 1


        if self.counter == 15 and not self.sent_offboard:
            self.get_logger().info('Sending OFFBOARD mode command')
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                1.0,
                6.0
            )
            self.sent_offboard = True

        if bool(self.get_parameter('auto_arm').value) and self.counter == 25 and not self.sent_arm:
            self.get_logger().info('Sending ARM command')
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                1.0
            )
            self.sent_arm = True


def main(args=None):
    rclpy.init(args=args)
    node = OffboardEnableNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
