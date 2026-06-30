import math
import shutil
import heapq
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import casadi as ca

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


from px4_msgs.msg import (
    VehicleOdometry,
    VehicleControlMode,
    VehicleThrustSetpoint,
    VehicleTorqueSetpoint,
)

from bluerov2_control.models.fossen_bluerov2_model import build_bluerov2_fossen_model

try:
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
except ImportError as e:
    raise ImportError(
        "acados_template is not installed or not visible in this Python environment. "
        "Install acados + the Python interface first."
    ) from e


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def quat_norm_wxyz(q):
    qw, qx, qy, qz = q
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n > 1e-12:
        return (qw / n, qx / n, qy / n, qz / n)
    return (1.0, 0.0, 0.0, 0.0)


def quat_to_yaw_wxyz(q):
    qw, qx, qy, qz = q
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def euler_to_quat_wxyz(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return quat_norm_wxyz((qw, qx, qy, qz))




def wrap_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


def rotz(yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)

def build_rigid_body_explicit_model(m, Ix, Iy, Iz):
    x = ca.SX.sym("x", 13)
    q = x[3:7]
    v = x[7:10]
    w = x[10:13]

    u = ca.SX.sym("u", 6)
    F = u[0:3]
    tau = u[3:6]

    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    R = ca.SX(3, 3)
    R[0, 0] = 1 - 2 * (qy * qy + qz * qz)
    R[0, 1] = 2 * (qx * qy - qz * qw)
    R[0, 2] = 2 * (qx * qz + qy * qw)
    R[1, 0] = 2 * (qx * qy + qz * qw)
    R[1, 1] = 1 - 2 * (qx * qx + qz * qz)
    R[1, 2] = 2 * (qy * qz - qx * qw)
    R[2, 0] = 2 * (qx * qz - qy * qw)
    R[2, 1] = 2 * (qy * qz + qx * qw)
    R[2, 2] = 1 - 2 * (qx * qx + qy * qy)

    wx, wy, wz = w[0], w[1], w[2]
    qdot = ca.vertcat(
        0.5 * (-qx * wx - qy * wy - qz * wz),
        0.5 * (qw * wx + qy * wz - qz * wy),
        0.5 * (qw * wy - qx * wz + qz * wx),
        0.5 * (qw * wz + qx * wy - qy * wx),
    )

    pdot = R @ v
    vdot = (1.0 / m) * F

    J = ca.diag(ca.vertcat(Ix, Iy, Iz))
    Jinv = ca.diag(ca.vertcat(1.0 / Ix, 1.0 / Iy, 1.0 / Iz))
    Jw = J @ w
    w_cross_Jw = ca.vertcat(
        w[1] * Jw[2] - w[2] * Jw[1],
        w[2] * Jw[0] - w[0] * Jw[2],
        w[0] * Jw[1] - w[1] * Jw[0],
    )
    wdot = Jinv @ (tau - w_cross_Jw)

    xdot = ca.vertcat(pdot, qdot, vdot, wdot)
    return x, u, xdot




def _parse_pose_text(pose_text):
    vals = [float(v) for v in pose_text.strip().split()]
    if len(vals) != 6:
        raise ValueError(f"Expected 6 pose values, got {len(vals)} from: {pose_text!r}")
    return tuple(vals)


def _inflate_rect(rect, inflate_xy):
    xmin, xmax, ymin, ymax = rect
    return (xmin - inflate_xy, xmax + inflate_xy, ymin - inflate_xy, ymax + inflate_xy)


def _rect_contains(rect, pt):
    xmin, xmax, ymin, ymax = rect
    x, y = pt
    return xmin <= x <= xmax and ymin <= y <= ymax


def _pose_rect_to_world(rect_local, pose_xyzrpy):
    tx, ty, _tz, _r, _p, yaw = pose_xyzrpy
    if abs(yaw) > 1e-9:
        raise ValueError("This planner expects axis-aligned tank pose (yaw=0).")
    xmin, xmax, ymin, ymax = rect_local
    return (xmin + tx, xmax + tx, ymin + ty, ymax + ty)


def _effective_bounds(inner_bounds_xy, fallback_bounds, wall_margin, robot_radius):
    xmin, xmax, ymin, ymax = inner_bounds_xy if inner_bounds_xy is not None else fallback_bounds
    m = wall_margin + robot_radius
    return (xmin + m, xmax - m, ymin + m, ymax - m)


def _parse_world_and_tank_geometry(world_sdf_path, tank_model_sdf_path):
    world_root = ET.parse(str(Path(world_sdf_path).expanduser().resolve())).getroot()
    world = world_root.find('world')
    if world is None:
        raise ValueError('No <world> element found in world SDF')

    tank_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    payload_pose = None
    water_surface_z = None

    for inc in world.findall('include'):
        name_el = inc.find('name')
        if name_el is not None and (name_el.text or '').strip() == 'kth_tank':
            pose_el = inc.find('pose')
            if pose_el is not None and pose_el.text:
                tank_pose = _parse_pose_text(pose_el.text)
            break

    for plugin in world.findall('plugin'):
        if plugin.attrib.get('name') == 'gz::sim::systems::Buoyancy':
            graded = plugin.find('graded_buoyancy')
            if graded is not None:
                change = graded.find('density_change')
                if change is not None:
                    above = change.find('above_depth')
                    if above is not None and above.text:
                        water_surface_z = float(above.text.strip())
            break

    for model in world.findall('model'):
        if model.attrib.get('name') == 'payload_box_0':
            pose_el = model.find('pose')
            if pose_el is None or not pose_el.text:
                raise ValueError('payload_box_0 found but has no <pose>')
            payload_pose = _parse_pose_text(pose_el.text)
            break
    if payload_pose is None:
        raise ValueError('payload_box_0 model not found in world SDF')

    tank_root = ET.parse(str(Path(tank_model_sdf_path).expanduser().resolve())).getroot()
    model = tank_root.find('model')
    if model is None:
        raise ValueError('No <model> element found in tank model SDF')
    link = model.find('link')
    if link is None:
        raise ValueError('No <link> element found in tank model SDF')

    inner_bounds_local = None
    floor_z_local = None
    ceiling_z_local = None
    for visual in link.findall('visual'):
        if visual.attrib.get('name') == 'water_volume_visual':
            pose_el = visual.find('pose')
            size_el = visual.find('./geometry/box/size')
            if pose_el is not None and pose_el.text and size_el is not None and size_el.text:
                x, y, z, _r, _p, _yaw = _parse_pose_text(pose_el.text)
                sx, sy, sz = [float(v) for v in size_el.text.strip().split()]
                inner_bounds_local = (x - sx/2.0, x + sx/2.0, y - sy/2.0, y + sy/2.0)
                floor_z_local = z - sz/2.0
                ceiling_z_local = z + sz/2.0
                break

    wall_faces = {'xmin': None, 'xmax': None, 'ymin': None, 'ymax': None}
    floor_top_local = None
    if inner_bounds_local is None:
        for collision in link.findall('collision'):
            name = collision.attrib.get('name', '')
            pose_el = collision.find('pose')
            size_el = collision.find('./geometry/box/size')
            if pose_el is None or not pose_el.text or size_el is None or not size_el.text:
                continue
            x, y, z, _r, _p, _yaw = _parse_pose_text(pose_el.text)
            sx, sy, sz = [float(v) for v in size_el.text.strip().split()]
            xmin, xmax = x - sx/2.0, x + sx/2.0
            ymin, ymax = y - sy/2.0, y + sy/2.0
            zmax = z + sz/2.0
            if name == 'tank_wall_x_min':
                wall_faces['xmin'] = xmax
            elif name == 'tank_wall_x_max':
                wall_faces['xmax'] = xmin
            elif name == 'tank_wall_y_min':
                wall_faces['ymin'] = ymax
            elif name == 'tank_wall_y_max':
                wall_faces['ymax'] = ymin
            elif name == 'tank_floor_box':
                floor_top_local = zmax
        if all(v is not None for v in wall_faces.values()):
            inner_bounds_local = (
                float(wall_faces['xmin']), float(wall_faces['xmax']),
                float(wall_faces['ymin']), float(wall_faces['ymax'])
            )
    if floor_z_local is None and floor_top_local is not None:
        floor_z_local = floor_top_local

    inner_bounds_world = _pose_rect_to_world(inner_bounds_local, tank_pose) if inner_bounds_local is not None else None
    return {
        'tank_pose_xyzrpy': tank_pose,
        'payload_box_pose_xyzrpy': payload_pose,
        'water_surface_z': water_surface_z,
        'tank_inner_bounds_xy': inner_bounds_world,
        'tank_floor_z': None if floor_z_local is None else floor_z_local + tank_pose[2],
        'tank_ceiling_z': None if ceiling_z_local is None else ceiling_z_local + tank_pose[2],
    }


class _OccupancyGrid2D:
    def __init__(self, bounds, resolution, obstacles):
        self.bounds = bounds
        self.resolution = resolution
        self.obstacles = list(obstacles)
        self.xmin, self.xmax, self.ymin, self.ymax = bounds
        self.nx = int(math.floor((self.xmax - self.xmin) / self.resolution)) + 1
        self.ny = int(math.floor((self.ymax - self.ymin) / self.resolution)) + 1
        if self.nx <= 1 or self.ny <= 1:
            raise ValueError('Invalid occupancy grid dimensions; check bounds/resolution')

    def world_to_grid(self, p):
        x, y = p
        i = int(round((x - self.xmin) / self.resolution))
        j = int(round((y - self.ymin) / self.resolution))
        return (max(0, min(self.nx - 1, i)), max(0, min(self.ny - 1, j)))

    def grid_to_world(self, idx):
        i, j = idx
        return (self.xmin + i * self.resolution, self.ymin + j * self.resolution)

    def in_bounds_idx(self, idx):
        i, j = idx
        return 0 <= i < self.nx and 0 <= j < self.ny

    def is_occupied_world(self, p):
        x, y = p
        if x < self.xmin or x > self.xmax or y < self.ymin or y > self.ymax:
            return True
        for rect in self.obstacles:
            if _rect_contains(rect, p):
                return True
        return False

    def is_occupied_idx(self, idx):
        return self.is_occupied_world(self.grid_to_world(idx))

    def line_is_free(self, p0, p1):
        dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        steps = max(1, int(math.ceil(dist / (0.5 * self.resolution))))
        for k in range(steps + 1):
            a = k / steps
            p = ((1.0 - a) * p0[0] + a * p1[0], (1.0 - a) * p0[1] + a * p1[1])
            if self.is_occupied_world(p):
                return False
        return True


def _astar_plan_xy(start_xy, goal_xy, bounds, obstacles, resolution=0.15, diagonal_motion=True):
    grid = _OccupancyGrid2D(bounds, resolution, obstacles)
    if grid.is_occupied_world(start_xy):
        raise ValueError(f'Start point lies in obstacle or outside bounds: {start_xy}')
    if grid.is_occupied_world(goal_xy):
        raise ValueError(f'Goal point lies in obstacle or outside bounds: {goal_xy}')

    start = grid.world_to_grid(start_xy)
    goal = grid.world_to_grid(goal_xy)

    moves = [(1,0),(-1,0),(0,1),(0,-1)]
    if diagonal_motion:
        moves += [(1,1),(1,-1),(-1,1),(-1,-1)]

    def h(a,b):
        dx = a[0]-b[0]
        dy = a[1]-b[1]
        return math.hypot(dx,dy)

    open_heap = []
    heapq.heappush(open_heap, (h(start, goal), 0.0, start))
    came_from = {}
    g_score = {start: 0.0}
    closed = set()

    while open_heap:
        _f, g_cur, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur == goal:
            break
        closed.add(cur)
        for di, dj in moves:
            nxt = (cur[0] + di, cur[1] + dj)
            if not grid.in_bounds_idx(nxt) or grid.is_occupied_idx(nxt):
                continue
            step = math.hypot(di, dj)
            cand = g_cur + step
            if cand < g_score.get(nxt, float('inf')):
                g_score[nxt] = cand
                came_from[nxt] = cur
                heapq.heappush(open_heap, (cand + h(nxt, goal), cand, nxt))

    if goal not in came_from and goal != start:
        raise RuntimeError('A* failed to find a path')

    path_idx = [goal]
    cur = goal
    while cur != start:
        cur = came_from[cur]
        path_idx.append(cur)
    path_idx.reverse()
    path_xy = [grid.grid_to_world(idx) for idx in path_idx]
    if path_xy:
        path_xy[0] = tuple(start_xy)
        path_xy[-1] = tuple(goal_xy)

    def simplify(points):
        if len(points) <= 2:
            return points
        out = [points[0]]
        for i in range(1, len(points)-1):
            a,b,c = out[-1], points[i], points[i+1]
            ab = (b[0]-a[0], b[1]-a[1])
            bc = (c[0]-b[0], c[1]-b[1])
            if abs(ab[0]*bc[1] - ab[1]*bc[0]) > 1e-9:
                out.append(b)
        out.append(points[-1])
        return out

    def shortcut(points):
        if len(points) <= 2:
            return points
        out = [points[0]]
        i = 0
        while i < len(points)-1:
            j = len(points)-1
            while j > i+1:
                if grid.line_is_free(points[i], points[j]):
                    break
                j -= 1
            out.append(points[j])
            i = j
        return out

    path_xy = simplify(path_xy)
    path_xy = shortcut(path_xy)
    return path_xy


class MPCTrackTrajectoryAcados(Node):
    def __init__(self):
        super().__init__("mpc_track_trajectory_acados")

        px4_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.declare_parameter("odom_topic", "/fmu/out/vehicle_odometry")
        self.declare_parameter("control_mode_topic", "/fmu/out/vehicle_control_mode")
        self.declare_parameter("thrust_sp_topic", "/fmu/in/vehicle_thrust_setpoint")
        self.declare_parameter("torque_sp_topic", "/fmu/in/vehicle_torque_setpoint")

        # Final tracking target.
        self.declare_parameter("goal_x", -1.15)
        self.declare_parameter("goal_y", -2.175)
        self.declare_parameter("goal_z", 95.7)
        self.declare_parameter("goal_roll", 0.0)
        self.declare_parameter("goal_pitch", 0.0)
        self.declare_parameter("goal_yaw", 0.0)
        self.declare_parameter("hold_attitude", True)

        # BlueRov2 parameters.
        self.declare_parameter("traj_mode", "linear")
        self.declare_parameter("traj_speed_mps", 0.06)
        self.declare_parameter("forward_pass_speed_mps", 0.04)
        self.declare_parameter("backward_pass_speed_mps", 0.05)
        self.declare_parameter("min_traj_duration_s", 2.0)
        self.declare_parameter("goal_reached_tol_m", 0.02)
        self.declare_parameter("regenerate_on_goal_change", True)

        self.declare_parameter("planner_mode", "astar")
        self.declare_parameter("planner_use_for_align", True)
        self.declare_parameter("planner_use_for_return", False)
        self.declare_parameter("world_sdf_path", "/home/yecheng/PX4-Autopilot/Tools/simulation/gz/worlds/kth_marinarium.sdf")
        self.declare_parameter("tank_model_sdf_path", "/home/yecheng/PX4-Autopilot/Tools/simulation/gz/models/kth_tank/model.sdf")
        self.declare_parameter("astar_resolution", 0.15)
        self.declare_parameter("astar_robot_radius", 0.20)
        self.declare_parameter("astar_obstacle_margin", 0.10)
        self.declare_parameter("astar_box_half_extent_x", 0.18)
        self.declare_parameter("astar_box_half_extent_y", 0.18)
        self.declare_parameter("astar_wall_margin", 0.15)
        self.declare_parameter("astar_diagonal_motion", True)
        self.declare_parameter("planner_fallback_bounds_xmin", -2.6)
        self.declare_parameter("planner_fallback_bounds_xmax", 2.6)
        self.declare_parameter("planner_fallback_bounds_ymin", -2.6)
        self.declare_parameter("planner_fallback_bounds_ymax", 2.6)

        self.declare_parameter("Ts", 0.04)
        self.declare_parameter("N", 25)
        self.declare_parameter("solve_rate_hz", 25.0)

        self.declare_parameter("model_type", "fossen")
        self.declare_parameter("mass", 13.5)
        self.declare_parameter("Ix", 0.26)
        self.declare_parameter("Iy", 0.23)
        self.declare_parameter("Iz", 0.37)

        self.declare_parameter("w_pos", 50.0)
        self.declare_parameter("w_vel", 15.0)
        self.declare_parameter("w_att", 20.0)
        self.declare_parameter("w_omega", 4.0)
        self.declare_parameter("w_u_force", 0.1)
        self.declare_parameter("w_u_torque", 0.05)

        self.declare_parameter("Fx_max_N", 88.0)
        self.declare_parameter("Fy_max_N", 88.0)
        self.declare_parameter("Fz_max_N", 137.0)
        self.declare_parameter("Mx_max_Nm", 30.0)
        self.declare_parameter("My_max_Nm", 16.5)
        self.declare_parameter("Mz_max_Nm", 21.0)

        self.declare_parameter("thrust_sat", 0.08)
        self.declare_parameter("torque_sat", 0.2)
        self.declare_parameter("publish_dt", 0.02)
        self.declare_parameter("odom_timeout_s", 0.3)

        self.declare_parameter("codegen_dir", "/tmp/bluerov2_acados_codegen")
        self.declare_parameter("rebuild_solver", False)

        self.declare_parameter("use_box_recovery_mission", True)

        # Box pose
        self.declare_parameter("box_x", -1.5)
        self.declare_parameter("box_y", -1.5)
        self.declare_parameter("box_z_sdf", -96.5)
        self.declare_parameter("box_roll", 1.57)
        self.declare_parameter("box_pitch", 0.0)
        self.declare_parameter("box_yaw", 1.57)

        # Hook pose
        self.declare_parameter("hook_mount_x", 0.42)
        self.declare_parameter("hook_mount_y", 0.04)
        self.declare_parameter("hook_mount_z_sdf", -0.08)

        self.declare_parameter("hook_tip_extra_x", 0.10)

        # Box handle center
        self.declare_parameter("handle_offset_world_x", 0.1)
        self.declare_parameter("handle_offset_world_y", -0.10)
        self.declare_parameter("handle_offset_world_z_down", -0.025)

        # Mission geometry
        self.declare_parameter("approach_yaw", -1.57)
        self.declare_parameter("approach_clearance", 0.1)
        self.declare_parameter("pass_overshoot", 0.1)
        self.declare_parameter("backward_extra_m", 0.08)
        self.declare_parameter("align_hold_s", 2.0)

        self.declare_parameter("mission_pos_tol_m", 0.08)
        self.declare_parameter("mission_yaw_tol_rad", 0.25)

        self.declare_parameter("forward_contact_enable", True)
        self.declare_parameter("forward_contact_min_time_s", 0.8)
        self.declare_parameter("forward_contact_min_progress_m", 0.03)
        self.declare_parameter("forward_contact_body_speed_eps_mps", 0.03)
        self.declare_parameter("forward_contact_force_cmd_eps_N", 8.0)

        # Return target
        self.declare_parameter("return_to_start", True)
        self.declare_parameter("shore_x", 0.0)
        self.declare_parameter("shore_y", 0.0)
        self.declare_parameter("shore_z", 95.0)
        self.declare_parameter("shore_yaw", 0.0)

        odom_topic = self.get_parameter("odom_topic").value
        cm_topic = self.get_parameter("control_mode_topic").value
        thrust_topic = self.get_parameter("thrust_sp_topic").value
        torque_topic = self.get_parameter("torque_sp_topic").value

        self.sub_odom = self.create_subscription(VehicleOdometry, odom_topic, self.on_odom, px4_qos)
        self.sub_cm = self.create_subscription(VehicleControlMode, cm_topic, self.on_control_mode, px4_qos)
        self.pub_thrust = self.create_publisher(VehicleThrustSetpoint, thrust_topic, px4_qos)
        self.pub_torque = self.create_publisher(VehicleTorqueSetpoint, torque_topic, px4_qos)

        self.have_odom = False
        self.p_w = np.zeros(3)
        self.q_wxyz = (1.0, 0.0, 0.0, 0.0)
        self.v_b = np.zeros(3)
        self.w_b = np.zeros(3)
        self.enabled = False
        self.last_odom_sec = None

        self.q_goal = euler_to_quat_wxyz(
            float(self.get_parameter("goal_roll").value),
            float(self.get_parameter("goal_pitch").value),
            float(self.get_parameter("goal_yaw").value),
        )

        self.ocp_solver = None
        self.u_force_cmd_N = np.zeros(3)
        self.u_tau_cmd_Nm = np.zeros(3)
        self.x_guess = None
        self.u_guess = None
        self.N_horizon = int(self.get_parameter("N").value)

        self.traj_active = False
        self.traj_start_time_sec = 0.0
        self.traj_duration_sec = 0.0
        self.traj_start_pos = np.zeros(3)
        self.traj_goal_pos = np.zeros(3)
        self.path_points = []
        self.path_segment_lengths = []
        self.path_total_length = 0.0
        self.traj_kind = "linear"
        self.last_goal_signature = None
        self._planner_geometry_cache = None

        self.mission_state = "INIT"
        self.state_enter_time_sec = 0.0
        self.home_pos = None
        self.home_yaw = 0.0

        self.active_goal_pos = np.array([
            float(self.get_parameter("goal_x").value),
            float(self.get_parameter("goal_y").value),
            float(self.get_parameter("goal_z").value),
        ], dtype=float)
        self.active_goal_yaw = float(self.get_parameter("goal_yaw").value)

        self.forward_pass_start_pos = None
        self.forward_pass_start_time_sec = 0.0

        self._build_mpc()

        self.get_logger().info(
            f"acados tracking MPC model_type = {str(self.get_parameter('model_type').value)}"
        )

        self.solve_timer = self.create_timer(
            1.0 / float(self.get_parameter("solve_rate_hz").value), self.solve_tick
        )
        self.pub_timer = self.create_timer(float(self.get_parameter("publish_dt").value), self.publish_tick)

    def on_control_mode(self, msg: VehicleControlMode):
        gate = bool(msg.flag_armed) and bool(msg.flag_control_offboard_enabled)
        if gate and not self.enabled:
            self.enabled = True
            self.get_logger().info("acados tracking MPC enabled (armed + offboard).")
        elif (not gate) and self.enabled:
            self.enabled = False
            self.get_logger().info("acados tracking MPC disabled.")
            self.u_force_cmd_N[:] = 0.0
            self.u_tau_cmd_Nm[:] = 0.0
            self.publish_zero()

    def on_odom(self, msg: VehicleOdometry):
        self.last_odom_sec = self._now_sec()
        self.p_w = np.array([float(msg.position[0]), float(msg.position[1]), float(msg.position[2])], dtype=float)
        self.q_wxyz = quat_norm_wxyz((float(msg.q[0]), float(msg.q[1]), float(msg.q[2]), float(msg.q[3])))
        self.v_b = np.array([float(msg.velocity[0]), float(msg.velocity[1]), float(msg.velocity[2])], dtype=float)
        self.w_b = np.array(
            [float(msg.angular_velocity[0]), float(msg.angular_velocity[1]), float(msg.angular_velocity[2])],
            dtype=float,
        )
        self.have_odom = True

        if not hasattr(self, "_logged_frame_once"):
            self._logged_frame_once = True
            yaw = quat_to_yaw_wxyz(self.q_wxyz)
            self.get_logger().info(
                f"ODOM init: p=[{self.p_w[0]:.3f},{self.p_w[1]:.3f},{self.p_w[2]:.3f}] "
                f"v_b=[{self.v_b[0]:.3f},{self.v_b[1]:.3f},{self.v_b[2]:.3f}] yaw={yaw:.3f} rad"
            )

    def _build_mpc(self):
        Ts = float(self.get_parameter("Ts").value)
        N = int(self.get_parameter("N").value)
        self.N_horizon = N
        model_type = str(self.get_parameter("model_type").value).strip().lower()

        if model_type == "fossen":
            x_sym, u_sym, xdot_fun, _ = build_bluerov2_fossen_model(Ts)
            xdot_expr = xdot_fun(x_sym, u_sym)
        else:
            m = float(self.get_parameter("mass").value)
            Ix = float(self.get_parameter("Ix").value)
            Iy = float(self.get_parameter("Iy").value)
            Iz = float(self.get_parameter("Iz").value)
            x_sym, u_sym, xdot_expr = build_rigid_body_explicit_model(m, Ix, Iy, Iz)

        p_sym = ca.SX.sym("p", 8)
        pref = p_sym[0:3]
        qref = p_sym[3:7]
        hold_att_flag = p_sym[7]

        model = AcadosModel()
        model.name = f"bluerov2_{model_type}_track"
        model.x = x_sym
        model.u = u_sym
        model.p = p_sym
        model.xdot = ca.SX.sym("xdot", 13)
        model.f_expl_expr = xdot_expr
        model.f_impl_expr = model.xdot - xdot_expr

        x = model.x
        u = model.u
        q = x[3:7]
        pos = x[0:3]
        vel = x[7:10]
        omega = x[10:13]
        F = u[0:3]
        tau = u[3:6]

        w_pos = float(self.get_parameter("w_pos").value)
        w_vel = float(self.get_parameter("w_vel").value)
        w_att = float(self.get_parameter("w_att").value)
        w_omega = float(self.get_parameter("w_omega").value)
        w_u_force = float(self.get_parameter("w_u_force").value)
        w_u_torque = float(self.get_parameter("w_u_torque").value)

        pos_err = pos - pref

        q1 = qref
        q2 = q
        q_conj = ca.vertcat(q2[0], -q2[1], -q2[2], -q2[3])
        q2_inv = q_conj / ca.norm_2(q2)

        q_w = q1[0] * q2_inv[0] - q1[1] * q2_inv[1] - q1[2] * q2_inv[2] - q1[3] * q2_inv[3]
        q_x = q1[0] * q2_inv[1] + q1[1] * q2_inv[0] + q1[2] * q2_inv[3] - q1[3] * q2_inv[2]
        q_y = q1[0] * q2_inv[2] - q1[1] * q2_inv[3] + q1[2] * q2_inv[0] + q1[3] * q2_inv[1]
        q_z = q1[0] * q2_inv[3] + q1[1] * q2_inv[2] - q1[2] * q2_inv[1] + q1[3] * q2_inv[0]

        q_err = ca.vertcat(q_w, q_x, q_y, q_z)
        q_err = ca.if_else(q_w < 0, -q_err, q_err)
        att_res = hold_att_flag * q_err[1:4]

        y_stage = ca.vertcat(pos_err, att_res, vel, omega, F, tau)
        y_term = ca.vertcat(pos_err, att_res, vel, omega)

        model.cost_y_expr = y_stage
        model.cost_y_expr_e = y_term

        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = N
        ocp.solver_options.tf = N * Ts
        ocp.parameter_values = np.zeros(8)

        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        W = np.diag(np.concatenate([
            w_pos * np.ones(3),
            w_att * np.ones(3),
            w_vel * np.ones(3),
            w_omega * np.ones(3),
            w_u_force * np.ones(3),
            w_u_torque * np.ones(3),
        ]))
        W_e = np.diag(np.concatenate([
            2.0 * w_pos * np.ones(3),
            2.0 * w_att * np.ones(3),
            w_vel * np.ones(3),
            w_omega * np.ones(3),
        ]))

        ocp.cost.W = W
        ocp.cost.W_e = W_e
        ocp.cost.yref = np.zeros((18,))
        ocp.cost.yref_e = np.zeros((12,))

        Fx_max_N = float(self.get_parameter("Fx_max_N").value)
        Fy_max_N = float(self.get_parameter("Fy_max_N").value)
        Fz_max_N = float(self.get_parameter("Fz_max_N").value)
        Mx_max_Nm = float(self.get_parameter("Mx_max_Nm").value)
        My_max_Nm = float(self.get_parameter("My_max_Nm").value)
        Mz_max_Nm = float(self.get_parameter("Mz_max_Nm").value)

        lbu = np.array([-Fx_max_N, -Fy_max_N, -Fz_max_N, -Mx_max_Nm, -My_max_Nm, -Mz_max_Nm], dtype=float)
        ubu = np.array([ Fx_max_N,  Fy_max_N,  Fz_max_N,  Mx_max_Nm,  My_max_Nm,  Mz_max_Nm], dtype=float)
        ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        ocp.constraints.lbu = lbu
        ocp.constraints.ubu = ubu

        x0 = np.zeros(13)
        x0[3] = 1.0
        ocp.constraints.x0 = x0

        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.print_level = 0
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 1
        ocp.solver_options.qp_solver_cond_N = min(10, N)
        ocp.solver_options.tol = 1e-3

        codegen_dir = Path(str(self.get_parameter("codegen_dir").value)).expanduser().resolve()
        if bool(self.get_parameter("rebuild_solver").value) and codegen_dir.exists():
            shutil.rmtree(codegen_dir)
        codegen_dir.mkdir(parents=True, exist_ok=True)
        ocp.code_export_directory = str(codegen_dir / model.name)

        json_file = str(codegen_dir / f"{model.name}_ocp.json")
        self.ocp_solver = AcadosOcpSolver(ocp, json_file=json_file, build=True, generate=True, verbose=False)

        self.x_guess = np.tile(x0.reshape(1, -1), (N + 1, 1))
        self.u_guess = np.zeros((N, 6), dtype=float)

    def _x_meas(self):
        x = np.zeros(13, dtype=float)
        x[0:3] = self.p_w
        x[3:7] = np.array(self.q_wxyz, dtype=float)
        x[7:10] = self.v_b
        x[10:13] = self.w_b
        return x

    def _goal_position_static(self):
        return np.array([
            float(self.get_parameter("goal_x").value),
            float(self.get_parameter("goal_y").value),
            float(self.get_parameter("goal_z").value),
        ], dtype=float)

    def _goal_yaw_static(self):
        return float(self.get_parameter("goal_yaw").value)

    def _goal_position(self):
        return self.active_goal_pos.copy()

    def _goal_quaternion(self):
        self.q_goal = euler_to_quat_wxyz(
            float(self.get_parameter("goal_roll").value),
            float(self.get_parameter("goal_pitch").value),
            float(self.active_goal_yaw),
        )
        return np.array(self.q_goal, dtype=float)


    def _box_center_ctrl(self):
        return np.array([
            float(self.get_parameter("box_x").value),
            float(self.get_parameter("box_y").value),
            -float(self.get_parameter("box_z_sdf").value),
        ], dtype=float)

    def _hook_tip_body_ctrl(self):
        return np.array([
            float(self.get_parameter("hook_mount_x").value) + float(self.get_parameter("hook_tip_extra_x").value),
            float(self.get_parameter("hook_mount_y").value),
            -float(self.get_parameter("hook_mount_z_sdf").value),
        ], dtype=float)

    def _handle_center_world(self):
        box = self._box_center_ctrl()
        offset = np.array([
            float(self.get_parameter("handle_offset_world_x").value),
            float(self.get_parameter("handle_offset_world_y").value),
            float(self.get_parameter("handle_offset_world_z_down").value),
        ], dtype=float)
        return box + offset

    def _mission_targets(self):
        handle_w = self._handle_center_world()

        yaw_align = float(self.get_parameter("approach_yaw").value)
        fwd = np.array([math.cos(yaw_align), math.sin(yaw_align), 0.0], dtype=float)

        r_hook_world = rotz(yaw_align) @ self._hook_tip_body_ctrl()

        approach_clearance = float(self.get_parameter("approach_clearance").value)
        pass_overshoot = float(self.get_parameter("pass_overshoot").value)
        backward_extra = float(self.get_parameter("backward_extra_m").value)

        p_align = handle_w - approach_clearance * fwd - r_hook_world
        p_pass = handle_w + pass_overshoot * fwd - r_hook_world
        p_back = p_align - backward_extra * fwd

        if bool(self.get_parameter("return_to_start").value) and self.home_pos is not None:
            p_home = self.home_pos.copy()
            yaw_home = self.home_yaw
        else:
            p_home = np.array([
                float(self.get_parameter("shore_x").value),
                float(self.get_parameter("shore_y").value),
                float(self.get_parameter("shore_z").value),
            ], dtype=float)
            yaw_home = float(self.get_parameter("shore_yaw").value)

        return p_align, yaw_align, p_pass, yaw_align, p_back, yaw_align, p_home, yaw_home

    def _forward_dir_world(self, yaw_align: float):
        return np.array([math.cos(yaw_align), math.sin(yaw_align), 0.0], dtype=float)

    def _should_abort_forward_due_to_contact(self, yaw_align: float):
        if not bool(self.get_parameter("forward_contact_enable").value):
            return False

        elapsed = self._now_sec() - self.forward_pass_start_time_sec
        if elapsed < float(self.get_parameter("forward_contact_min_time_s").value):
            return False

        if self.forward_pass_start_pos is None:
            return False

        fwd = self._forward_dir_world(yaw_align)
        progress = float(np.dot(self.p_w - self.forward_pass_start_pos, fwd))
        if progress < float(self.get_parameter("forward_contact_min_progress_m").value):
            return False

        body_speed = float(np.linalg.norm(self.v_b))
        force_cmd = float(np.linalg.norm(self.u_force_cmd_N))

        slow = body_speed <= float(self.get_parameter("forward_contact_body_speed_eps_mps").value)
        pushing = force_cmd >= float(self.get_parameter("forward_contact_force_cmd_eps_N").value)
        return slow and pushing

    def _at_active_goal(self):
        pos_tol = float(self.get_parameter("mission_pos_tol_m").value)
        yaw_tol = float(self.get_parameter("mission_yaw_tol_rad").value)

        pos_err = np.linalg.norm(self.p_w - self.active_goal_pos)
        yaw_now = quat_to_yaw_wxyz(self.q_wxyz)
        yaw_err = abs(wrap_pi(yaw_now - self.active_goal_yaw))

        return (pos_err <= pos_tol) and (yaw_err <= yaw_tol)

    def _update_mission(self):
        if not bool(self.get_parameter("use_box_recovery_mission").value):
            self.active_goal_pos = self._goal_position_static()
            self.active_goal_yaw = self._goal_yaw_static()
            return

        if not self.have_odom:
            return

        if self.home_pos is None:
            self.home_pos = self.p_w.copy()
            self.home_yaw = quat_to_yaw_wxyz(self.q_wxyz)
            self.mission_state = "ALIGN"
            self.state_enter_time_sec = self._now_sec()
            self.get_logger().info(
                f"Mission start: home={self.home_pos}, home_yaw={self.home_yaw:.3f} rad"
            )

        p_align, yaw_align, p_pass, yaw_pass, p_back, yaw_back, p_home, yaw_home = self._mission_targets()

        if self.mission_state == "ALIGN":
            self.active_goal_pos = p_align
            self.active_goal_yaw = yaw_align
            if self._at_active_goal():
                self.mission_state = "ALIGN_HOLD"
                self.state_enter_time_sec = self._now_sec()
                self.get_logger().info("Mission -> ALIGN_HOLD")

        elif self.mission_state == "ALIGN_HOLD":
            self.active_goal_pos = p_align
            self.active_goal_yaw = yaw_align
            if not self._at_active_goal():
                self.mission_state = "ALIGN"
                self.state_enter_time_sec = self._now_sec()
                self.get_logger().info("Mission -> ALIGN (drifted during hold)")
            elif (self._now_sec() - self.state_enter_time_sec) >= float(self.get_parameter("align_hold_s").value):
                self.mission_state = "FORWARD_PASS"
                self.state_enter_time_sec = self._now_sec()
                self.forward_pass_start_time_sec = self.state_enter_time_sec
                self.forward_pass_start_pos = self.p_w.copy()
                self.get_logger().info("Mission -> FORWARD_PASS")

        elif self.mission_state == "FORWARD_PASS":
            self.active_goal_pos = p_pass
            self.active_goal_yaw = yaw_pass
            if self._at_active_goal():
                self.mission_state = "BACKWARD_PASS"
                self.state_enter_time_sec = self._now_sec()
                self.get_logger().info("Mission -> BACKWARD_PASS")
            elif self._should_abort_forward_due_to_contact(yaw_pass):
                self.mission_state = "BACKWARD_PASS"
                self.state_enter_time_sec = self._now_sec()
                self.get_logger().info("Mission -> BACKWARD_PASS (contact heuristic)")

        elif self.mission_state == "BACKWARD_PASS":
            self.active_goal_pos = p_back
            self.active_goal_yaw = yaw_back
            if self._at_active_goal():
                self.mission_state = "RETURN_SHORE"
                self.state_enter_time_sec = self._now_sec()
                self.get_logger().info("Mission -> RETURN_SHORE")

        elif self.mission_state == "RETURN_SHORE":
            self.active_goal_pos = p_home
            self.active_goal_yaw = yaw_home
            if self._at_active_goal():
                self.mission_state = "DONE"
                self.state_enter_time_sec = self._now_sec()
                self.get_logger().info("Mission -> DONE")

        else:
            self.active_goal_pos = p_home
            self.active_goal_yaw = yaw_home

    def _goal_signature(self):
        g = self._goal_position()
        q = self._goal_quaternion()
        return tuple(np.round(np.concatenate([g, q]), 6).tolist())

    def _now_sec(self):
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def _odom_fresh(self):
        if not self.have_odom or self.last_odom_sec is None:
            return False
        timeout_s = float(self.get_parameter("odom_timeout_s").value)
        if timeout_s <= 0.0:
            return True
        age_s = self._now_sec() - self.last_odom_sec
        if age_s <= timeout_s:
            return True
        self.get_logger().warn(
            f"Odometry stale for {age_s:.2f}s; publishing zero control.",
            throttle_duration_sec=1.0,
        )
        return False

    def _zero_command_cache(self):
        self.u_force_cmd_N[:] = 0.0
        self.u_tau_cmd_Nm[:] = 0.0

    def _phase_traj_speed(self):
        if self.mission_state == "FORWARD_PASS":
            return float(self.get_parameter("forward_pass_speed_mps").value)
        if self.mission_state == "BACKWARD_PASS":
            return float(self.get_parameter("backward_pass_speed_mps").value)
        return float(self.get_parameter("traj_speed_mps").value)


    def _planner_enabled_for_current_phase(self):
        mode = str(self.get_parameter("planner_mode").value).strip().lower()
        if mode != "astar":
            return False
        if self.mission_state == "ALIGN":
            return bool(self.get_parameter("planner_use_for_align").value)
        if self.mission_state == "RETURN_SHORE":
            return bool(self.get_parameter("planner_use_for_return").value)
        return False

    def _load_planner_geometry(self):
        if self._planner_geometry_cache is not None:
            return self._planner_geometry_cache
        world_sdf_path = str(self.get_parameter("world_sdf_path").value).strip()
        tank_model_sdf_path = str(self.get_parameter("tank_model_sdf_path").value).strip()
        if not world_sdf_path or not tank_model_sdf_path:
            raise ValueError("planner_mode=astar but world_sdf_path/tank_model_sdf_path not set")
        self._planner_geometry_cache = _parse_world_and_tank_geometry(world_sdf_path, tank_model_sdf_path)
        return self._planner_geometry_cache

    def _build_astar_path_xy(self, start_xy, goal_xy):
        geom = self._load_planner_geometry()
        fallback_bounds = (
            float(self.get_parameter("planner_fallback_bounds_xmin").value),
            float(self.get_parameter("planner_fallback_bounds_xmax").value),
            float(self.get_parameter("planner_fallback_bounds_ymin").value),
            float(self.get_parameter("planner_fallback_bounds_ymax").value),
        )
        bounds = _effective_bounds(
            geom.get("tank_inner_bounds_xy"),
            fallback_bounds,
            float(self.get_parameter("astar_wall_margin").value),
            float(self.get_parameter("astar_robot_radius").value),
        )
        bx, by, _bz, _r, _p, _yaw = geom["payload_box_pose_xyzrpy"]
        rect = (
            bx - float(self.get_parameter("astar_box_half_extent_x").value),
            bx + float(self.get_parameter("astar_box_half_extent_x").value),
            by - float(self.get_parameter("astar_box_half_extent_y").value),
            by + float(self.get_parameter("astar_box_half_extent_y").value),
        )
        inflate = float(self.get_parameter("astar_robot_radius").value) + float(self.get_parameter("astar_obstacle_margin").value)
        obstacles = [_inflate_rect(rect, inflate)]
        return _astar_plan_xy(
            start_xy=tuple(start_xy),
            goal_xy=tuple(goal_xy),
            bounds=bounds,
            obstacles=obstacles,
            resolution=float(self.get_parameter("astar_resolution").value),
            diagonal_motion=bool(self.get_parameter("astar_diagonal_motion").value),
        )

    def _set_linear_trajectory(self, goal_pos):
        self.traj_start_pos = self.p_w.copy()
        self.traj_goal_pos = goal_pos.copy()
        self.traj_start_time_sec = self._now_sec()
        dist = float(np.linalg.norm(self.traj_goal_pos - self.traj_start_pos))
        speed = max(self._phase_traj_speed(), 1e-4)
        min_duration = max(float(self.get_parameter("min_traj_duration_s").value), float(self.get_parameter("Ts").value))
        self.traj_duration_sec = max(dist / speed, min_duration)
        self.traj_active = True
        self.traj_kind = "linear"
        self.path_points = []
        self.path_segment_lengths = []
        self.path_total_length = 0.0
        self.last_goal_signature = self._goal_signature()
        self.get_logger().info(
            f"New linear trajectory: start={self.traj_start_pos}, goal={self.traj_goal_pos}, duration={self.traj_duration_sec:.2f}s"
        )

    def _set_path_trajectory(self, path_xy, goal_z):
        self.traj_start_time_sec = self._now_sec()
        self.path_points = [np.array([float(x), float(y), float(goal_z)], dtype=float) for x, y in path_xy]
        self.path_segment_lengths = []
        self.path_total_length = 0.0
        for i in range(len(self.path_points) - 1):
            seg = float(np.linalg.norm(self.path_points[i + 1] - self.path_points[i]))
            self.path_segment_lengths.append(seg)
            self.path_total_length += seg
        self.traj_start_pos = self.path_points[0].copy()
        self.traj_goal_pos = self.path_points[-1].copy()
        speed = max(self._phase_traj_speed(), 1e-4)
        min_duration = max(float(self.get_parameter("min_traj_duration_s").value), float(self.get_parameter("Ts").value))
        self.traj_duration_sec = max(self.path_total_length / speed, min_duration)
        self.traj_active = True
        self.traj_kind = "path"
        self.last_goal_signature = self._goal_signature()
        self.get_logger().info(
            f"New A* path trajectory: npts={len(self.path_points)}, length={self.path_total_length:.2f}m, duration={self.traj_duration_sec:.2f}s"
        )

    def _sample_path_pref(self, alpha):
        if not self.path_points:
            return self._goal_position()
        if len(self.path_points) == 1 or self.path_total_length <= 1e-9:
            return self.path_points[-1].copy()
        s = clamp(alpha, 0.0, 1.0) * self.path_total_length
        acc = 0.0
        for i, seg_len in enumerate(self.path_segment_lengths):
            if acc + seg_len >= s or i == len(self.path_segment_lengths) - 1:
                local = 0.0 if seg_len <= 1e-9 else (s - acc) / seg_len
                return (1.0 - local) * self.path_points[i] + local * self.path_points[i + 1]
            acc += seg_len
        return self.path_points[-1].copy()

    def _reset_trajectory_from_current_pose(self):
        goal_pos = self._goal_position()

        if self._planner_enabled_for_current_phase():
            try:
                path_xy = self._build_astar_path_xy(self.p_w[0:2], goal_pos[0:2])
                if len(path_xy) >= 2:
                    self._set_path_trajectory(path_xy, goal_pos[2])
                    return
                self.get_logger().warn("A* returned fewer than 2 waypoints, fallback to linear trajectory.")
            except Exception as e:
                self.get_logger().warn(f"A* planner failed, fallback to linear trajectory: {e}")

        self._set_linear_trajectory(goal_pos)

    def _maybe_refresh_trajectory(self):
        self._update_mission()

        if not self.have_odom:
            return

        if getattr(self, "mission_state", "") == "DONE":
            self.traj_active = False
            return

        current_goal_sig = self._goal_signature()
        regenerate = bool(self.get_parameter("regenerate_on_goal_change").value)
        goal_changed = (self.last_goal_signature is None) or (current_goal_sig != self.last_goal_signature)

        if (not self.traj_active) or (regenerate and goal_changed):
            self._reset_trajectory_from_current_pose()
            return

        goal_tol = max(float(self.get_parameter("goal_reached_tol_m").value), 1e-4)
        pos_reached = np.linalg.norm(self.p_w - self.traj_goal_pos) <= goal_tol

        if bool(self.get_parameter("use_box_recovery_mission").value):
            if pos_reached and self._at_active_goal():
                self.traj_active = False
        else:
            if pos_reached:
                self.traj_active = False

    def _trajectory_stage_param(self, k: int):
        qref = self._goal_quaternion()
        hold_att_flag = 1.0 if bool(self.get_parameter("hold_attitude").value) else 0.0

        if not self.traj_active:
            pref = self._goal_position()
            return np.concatenate([pref, qref, np.array([hold_att_flag], dtype=float)])

        t_now = self._now_sec()
        Ts = float(self.get_parameter("Ts").value)
        t_stage = (t_now - self.traj_start_time_sec) + k * Ts

        if self.traj_duration_sec <= 1e-6:
            alpha = 1.0
        else:
            alpha = clamp(t_stage / self.traj_duration_sec, 0.0, 1.0)

        if self.traj_kind == "path":
            pref = self._sample_path_pref(alpha)
        else:
            pref = (1.0 - alpha) * self.traj_start_pos + alpha * self.traj_goal_pos

        return np.concatenate([pref, qref, np.array([hold_att_flag], dtype=float)])

    def _forceN_to_thrust_norm(self, F_N):
        Fx_max_N = max(float(self.get_parameter("Fx_max_N").value), 1e-6)
        Fy_max_N = max(float(self.get_parameter("Fy_max_N").value), 1e-6)
        Fz_max_N = max(float(self.get_parameter("Fz_max_N").value), 1e-6)
        return np.array([F_N[0] / Fx_max_N, F_N[1] / Fy_max_N, F_N[2] / Fz_max_N], dtype=float)

    def _torqueNm_to_torque_norm(self, tau_Nm):
        Mx_max_Nm = max(float(self.get_parameter("Mx_max_Nm").value), 1e-6)
        My_max_Nm = max(float(self.get_parameter("My_max_Nm").value), 1e-6)
        Mz_max_Nm = max(float(self.get_parameter("Mz_max_Nm").value), 1e-6)
        return np.array([
            tau_Nm[0] / Mx_max_Nm,
            tau_Nm[1] / My_max_Nm,
            tau_Nm[2] / Mz_max_Nm,
        ], dtype=float)

    def solve_tick(self):
        if not self.enabled or self.ocp_solver is None:
            return
        if not self._odom_fresh():
            self._zero_command_cache()
            return

        self._maybe_refresh_trajectory()
        x0 = self._x_meas()

        qn = np.linalg.norm(x0[3:7])
        if qn > 1e-12:
            x0[3:7] = x0[3:7] / qn
        else:
            x0[3:7] = np.array([1.0, 0.0, 0.0, 0.0])

        for k in range(self.N_horizon):
            p_k = self._trajectory_stage_param(k)
            self.ocp_solver.set(k, "x", self.x_guess[k])
            self.ocp_solver.set(k, "u", self.u_guess[k])
            self.ocp_solver.set(k, "p", p_k)

        p_terminal = self._trajectory_stage_param(self.N_horizon)
        self.ocp_solver.set(self.N_horizon, "x", self.x_guess[self.N_horizon])
        self.ocp_solver.set(self.N_horizon, "p", p_terminal)

        self.ocp_solver.set(0, "lbx", x0)
        self.ocp_solver.set(0, "ubx", x0)

        status = self.ocp_solver.solve()
        if status != 0:
            self.get_logger().warn(f"acados solve failed, status={status}")
            return

        u0 = np.array(self.ocp_solver.get(0, "u"), dtype=float).reshape(-1)
        self.u_force_cmd_N = u0[0:3].copy()
        self.u_tau_cmd_Nm = u0[3:6].copy()

        x_new = np.zeros_like(self.x_guess)
        u_new = np.zeros_like(self.u_guess)
        for k in range(self.N_horizon + 1):
            x_new[k, :] = np.array(self.ocp_solver.get(k, "x"), dtype=float).reshape(-1)
            qk = x_new[k, 3:7]
            nq = np.linalg.norm(qk)
            if nq > 1e-12:
                x_new[k, 3:7] = qk / nq
        for k in range(self.N_horizon):
            u_new[k, :] = np.array(self.ocp_solver.get(k, "u"), dtype=float).reshape(-1)

        self.x_guess[:-1, :] = x_new[1:, :]
        self.x_guess[-1, :] = x_new[-1, :]
        self.u_guess[:-1, :] = u_new[1:, :]
        self.u_guess[-1, :] = u_new[-1, :]

    def publish_zero(self):
        now_us = int(self.get_clock().now().nanoseconds / 1000)

        thr = VehicleThrustSetpoint()
        thr.timestamp = now_us
        thr.timestamp_sample = 0
        thr.xyz = [0.0, 0.0, 0.0]
        self.pub_thrust.publish(thr)

        tor = VehicleTorqueSetpoint()
        tor.timestamp = now_us
        tor.timestamp_sample = 0
        tor.xyz = [0.0, 0.0, 0.0]
        self.pub_torque.publish(tor)

    def publish_tick(self):
        if not self.enabled:
            return
        if not self._odom_fresh():
            self._zero_command_cache()
            self.publish_zero()
            return

        now_us = int(self.get_clock().now().nanoseconds / 1000)

        thr_norm = self._forceN_to_thrust_norm(self.u_force_cmd_N)
        thrust_sat = float(self.get_parameter("thrust_sat").value)
        thr_norm = np.array([
            clamp(thr_norm[0], -thrust_sat, thrust_sat),
            clamp(thr_norm[1], -thrust_sat, thrust_sat),
            clamp(thr_norm[2], -thrust_sat, thrust_sat),
        ], dtype=float)

        thr = VehicleThrustSetpoint()
        thr.timestamp = now_us
        thr.timestamp_sample = 0
        thr.xyz = [float(thr_norm[0]), float(thr_norm[1]), float(thr_norm[2])]
        self.pub_thrust.publish(thr)

        tau_norm = self._torqueNm_to_torque_norm(self.u_tau_cmd_Nm)
        torque_sat = float(self.get_parameter("torque_sat").value)
        tau_norm = np.array([
            clamp(tau_norm[0], -torque_sat, torque_sat),
            clamp(tau_norm[1], -torque_sat, torque_sat),
            clamp(tau_norm[2], -torque_sat, torque_sat),
        ], dtype=float)

        tor = VehicleTorqueSetpoint()
        tor.timestamp = now_us
        tor.timestamp_sample = 0
        tor.xyz = [float(tau_norm[0]), float(tau_norm[1]), float(tau_norm[2])]
        self.pub_torque.publish(tor)


def main():
    rclpy.init()
    node = MPCTrackTrajectoryAcados()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    try:
        if rclpy.ok():
            node.publish_zero()
    except Exception:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
