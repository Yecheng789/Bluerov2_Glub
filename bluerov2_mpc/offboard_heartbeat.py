#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from px4_msgs.msg import OffboardControlMode

class OffboardHeartbeat(Node):
    def __init__(self):
        super().__init__("offboard_heartbeat")
        self.declare_parameter("topic", "/itrl_rov_1/fmu/in/offboard_control_mode")
        topic = self.get_parameter("topic").get_parameter_value().string_value
        self.pub = self.create_publisher(OffboardControlMode, topic, 10)
        self.timer = self.create_timer(0.05, self.tick)  # 20 Hz

    def tick(self):
        msg = OffboardControlMode()
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        # We use direct_actuator for the PID/MPC
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.thrust_and_torque = False
        msg.direct_actuator = True
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = OffboardHeartbeat()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()