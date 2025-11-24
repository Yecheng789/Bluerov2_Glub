#!/usr/bin/env python3
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock

from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from px4_msgs.msg import ActuatorMotors

class SoftDivePID(Node):
    def __init__(self):
        super().__init__("soft_dive_pid")

        # --- Params ---
        self.declare_parameter("odom_topic", "/mocap/itrl_rov_1/odom")
        self.declare_parameter("motors_topic", "/itrl_rov_1/fmu/in/actuator_motors")
        self.declare_parameter("rate", 20.0)                 # Hz
        self.declare_parameter("z_ref", 0.8)                 # NED (surface=0, down positive)
        self.declare_parameter("engage_band", 0.10)          # m band to switch from dive->hold
        self.declare_parameter("soft_bias", 0.65)            # 0..1 motor cmd (0.5 ~ neutral). >0.5 pushes down
        self.declare_parameter("soft_timeout", 10.0)         # s fallback to HOLD even if band not reached
        self.declare_parameter("kp", 0.20)
        self.declare_parameter("ki", 0.05)
        self.declare_parameter("kd", 0.05)
        self.declare_parameter("u_min", 0.40)                # clamp motors
        self.declare_parameter("u_max", 0.70)
        self.declare_parameter("start_mpc_on_ready", True)
        self.declare_parameter("handoff_seconds", 2.0)

        self.odom_topic  = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.motors_topic= self.get_parameter("motors_topic").get_parameter_value().string_value
        self.rate        = float(self.get_parameter("rate").value)
        self.z_ref       = float(self.get_parameter("z_ref").value)
        self.engage_band = float(self.get_parameter("engage_band").value)
        self.soft_bias   = float(self.get_parameter("soft_bias").value)
        self.soft_timeout= float(self.get_parameter("soft_timeout").value)
        self.kp          = float(self.get_parameter("kp").value)
        self.ki          = float(self.get_parameter("ki").value)
        self.kd          = float(self.get_parameter("kd").value)
        self.u_min       = float(self.get_parameter("u_min").value)
        self.u_max       = float(self.get_parameter("u_max").value)
        self.start_mpc_on_ready = bool(self.get_parameter("start_mpc_on_ready").value)
        self.handoff_seconds = float(self.get_parameter("handoff_seconds").value)

        # --- IO ---
        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.cb_odom, 10)
        self.pub_mot  = self.create_publisher(ActuatorMotors, self.motors_topic, 10)
        self.pub_start= self.create_publisher(Bool, "/bluerov2_mpc/start_mpc", 10)

        # --- State ---
        self.have_odom = False
        self.z = 0.0
        self.last_z = None
        self.last_t = None
        self.phase = "WAIT"  # WAIT -> SOFT_DIVE -> HOLD
        self.t_phase0 = time.time()
        self.pid_i = 0.0
        self.ready_since = None

        self.timer = self.create_timer(1.0/self.rate, self.tick)
        self.get_logger().info("SoftDivePID ready.")

    def cb_odom(self, odom: Odometry):
        self.z = float(odom.pose.pose.position.z)  # NED
        self.have_odom = True

    def motors_msg(self, val_0to1):
        # PX4 expects 12-length; we fill last 4 (indexes 4..7 for motors 5..8? Actually: 0..11)
        # In previous MPC you filled first 8. We keep consistency: fill first 8, last 4 zeros.
        arr = 0.5*np.ones(12, dtype=np.float32)
        # “last 4 motors” in your wiring are motors 5..8 => indices 4,5,6,7
        arr[4] = arr[5] = arr[6] = arr[7] = float(val_0to1)
        msg = ActuatorMotors()
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        msg.control = arr.tolist()
        msg.timestamp_sample = 0
        msg.reversible_flags = 0
        return msg

    def publish_motors(self, u):
        u = float(np.clip(u, self.u_min, self.u_max))
        self.pub_mot.publish(self.motors_msg(u))

    def tick(self):
        now = time.time()

        if self.phase == "WAIT":
            # Send neutral-ish while waiting for odom (slightly down to begin sinking gently)
            self.publish_motors(self.soft_bias)
            if self.have_odom:
                self.phase = "SOFT_DIVE"
                self.t_phase0 = now
                self.get_logger().info("Got odom -> SOFT_DIVE")

        elif self.phase == "SOFT_DIVE":
            # Open-loop gentle downwards
            self.publish_motors(self.soft_bias)
            # Switch to HOLD when close enough or timeout
            if (self.have_odom and (self.z <= self.z_ref + self.engage_band)) or ((now - self.t_phase0) > self.soft_timeout):
                self.phase = "HOLD"
                self.t_phase0 = now
                self.pid_i = 0.0
                self.last_z = self.z
                self.last_t = now
                self.ready_since = None
                self.get_logger().info("Switching to HOLD (PID).")

        elif self.phase == "HOLD":
            # Simple PID on depth (NED): e = z_ref - z (so negative e when above target)
            if not self.have_odom:
                # fall back to gentle bias
                self.publish_motors(self.soft_bias)
                return
            z = self.z
            t = now
            dt = max(1e-3, (t - (self.last_t if self.last_t else t)))
            e = self.z_ref - z
            self.pid_i += e * dt
            d = (z - (self.last_z if self.last_z is not None else z)) / dt
            # controller output around 0.5 neutral
            u = 0.5 + (self.kp * e + self.ki * self.pid_i - self.kd * d)  # minus on d because z increasing is going downwards in NED? keep robust
            self.publish_motors(u)
            self.last_z, self.last_t = z, t

            # Ready to handoff when |e| small for a bit
            if abs(e) < self.engage_band * 0.5:
                if self.ready_since is None:
                    self.ready_since = t
                elif (t - self.ready_since) > self.handoff_seconds and self.start_mpc_on_ready:
                    self.pub_start.publish(Bool(data=True))
                    # only publish once
                    self.start_mpc_on_ready = False
                    self.get_logger().info("Depth stable -> published /bluerov2_mpc/start_mpc=True")

def main():
    rclpy.init()
    node = SoftDivePID()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()