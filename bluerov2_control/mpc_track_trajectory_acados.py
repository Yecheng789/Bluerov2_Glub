import math
import shutil
import heapq
import time
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
from std_msgs.msg import Bool, Empty

from bluerov2_control.models.fossen_bluerov2_model import (
    build_bluerov2_fossen_model as build_bluerov2_fossen_model_sim,
)
from bluerov2_control.models.fossen_bluerov2_model_real import (
    build_bluerov2_fossen_model as build_bluerov2_fossen_model_real,
)

try:
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
except ImportError as e:
    raise ImportError(
        "acados_template is not installed or not visible in this Python environment. "
        "Install acados + the Python interface first."
    ) from e


DEFAULT_CONTROL_MODE_TIMEOUT_SEC = 1.25


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _validated_operating_bounds(
    xmin,
    xmax,
    ymin,
    ymax,
    zmin,
    zmax,
):
    """Return strict finite NED operating bounds as min/max vectors."""
    values = np.asarray(
        [xmin, xmax, ymin, ymax, zmin, zmax],
        dtype=float,
    )
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("operating bounds must contain six finite values")

    minimum = values[[0, 2, 4]]
    maximum = values[[1, 3, 5]]
    if not np.all(minimum < maximum):
        raise ValueError(
            "operating bounds require xmin < xmax, ymin < ymax, and "
            "zmin < zmax"
        )
    return minimum, maximum


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


def quat_slerp_wxyz(q0, q1, alpha):
    q0 = np.asarray(quat_norm_wxyz(q0), dtype=float)
    q1 = np.asarray(quat_norm_wxyz(q1), dtype=float)
    alpha = clamp(float(alpha), 0.0, 1.0)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = clamp(dot, -1.0, 1.0)

    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
        return np.asarray(quat_norm_wxyz(q), dtype=float)

    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    w0 = math.sin((1.0 - alpha) * theta) / sin_theta
    w1 = math.sin(alpha * theta) / sin_theta
    return np.asarray(quat_norm_wxyz(w0 * q0 + w1 * q1), dtype=float)


def quat_angular_distance_wxyz(q0, q1):
    q0 = np.asarray(quat_norm_wxyz(q0), dtype=float)
    q1 = np.asarray(quat_norm_wxyz(q1), dtype=float)
    dot = clamp(abs(float(np.dot(q0, q1))), 0.0, 1.0)
    return 2.0 * math.acos(dot)


def quat_to_rotation_matrix_wxyz(q):
    """Return the numerical body-to-world rotation for a WXYZ quaternion."""
    qw, qx, qy, qz = quat_norm_wxyz(q)
    return np.array([
        [
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ],
        [
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ],
        [
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
    ], dtype=float)




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

        # A separate operator-owned gate prevents arming + Offboard from
        # immediately starting a trajectory. Set require_mission_enable=false
        # for legacy/simulation launches that intentionally use the old gate.
        self.declare_parameter("mission_enable_topic", "/bluerov2/mission_enable")
        self.declare_parameter("require_mission_enable", True)
        self.declare_parameter(
            "controller_heartbeat_topic",
            "/bluerov2/controller_heartbeat",
        )

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
        # Optional fixed-hook phase overrides.  Zero preserves the legacy
        # shared traj_speed_mps behaviour for simulation and older launches.
        self.declare_parameter("pre_approach_speed_mps", 0.0)
        self.declare_parameter("final_approach_speed_mps", 0.0)
        self.declare_parameter("traj_angular_speed_rad_s", 0.0)
        self.declare_parameter("use_pre_approach_waypoint", False)
        self.declare_parameter("pre_approach_x", 0.0)
        self.declare_parameter("pre_approach_y", 0.0)
        self.declare_parameter("pre_approach_z", 0.0)
        self.declare_parameter("forward_pass_speed_mps", 0.04)
        self.declare_parameter("backward_pass_speed_mps", 0.05)
        self.declare_parameter("final_pose_hold_s", 0.0)
        self.declare_parameter(
            "return_to_pre_approach_after_hold",
            False,
        )
        # A tighter vertical gate is used before the fixed-hook controller
        # advances onto its horizontal in/out line.  Zero preserves legacy
        # behaviour for launches that do not request this extra constraint.
        self.declare_parameter("fixed_hook_depth_tolerance_m", 0.0)
        self.declare_parameter("min_traj_duration_s", 2.0)
        self.declare_parameter("goal_reached_tol_m", 0.02)
        self.declare_parameter(
            "goal_reached_orientation_tol_rad",
            math.radians(10.0),
        )
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
        self.declare_parameter("robot_type", "standard")
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

        # Offset-free outer position loop.  The finite-horizon MPC has no
        # disturbance state, so an unmodelled tether/buoyancy force can leave
        # a persistent waypoint error even while the solver is healthy.  This
        # bounded integrator is enabled only after the nominal reference has
        # finished, and its contribution is combined with (then clipped with)
        # the MPC force.  A zero gain preserves legacy behaviour.
        self.declare_parameter(
            "position_integral_gain_N_per_m_s",
            0.0,
        )
        self.declare_parameter(
            "position_integral_force_limit_fraction",
            0.0,
        )
        self.declare_parameter(
            "position_integral_activation_error_m",
            0.50,
        )
        self.declare_parameter(
            "position_integral_max_dt_s",
            0.10,
        )

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
        # PX4 Commander publishes VehicleControlMode at 2 Hz. A 0.5 s
        # timeout sits exactly on that period and fails on normal DDS/executor
        # jitter. Allow one missed update, consistently with offboard_enable.
        self.declare_parameter(
            "control_mode_timeout_s",
            DEFAULT_CONTROL_MODE_TIMEOUT_SEC,
        )
        self.declare_parameter("command_timeout_s", 0.2)
        self.declare_parameter("max_solve_gap_s", 0.3)
        self.declare_parameter("max_solve_duration_s", 0.2)
        self.declare_parameter("revoke_mission_on_solver_failure", False)
        self.declare_parameter("revoke_mission_on_state_failure", False)
        self.declare_parameter("max_initial_goal_distance_m", 2.0)
        self.declare_parameter(
            "max_initial_goal_orientation_error_rad", 0.0
        )
        self.declare_parameter("max_odom_position_jump_m", 0.5)
        self.declare_parameter("max_odom_orientation_jump_rad", math.radians(45.0))
        self.declare_parameter("max_tilt_rad", 0.0)
        self.declare_parameter("require_expected_odom_frames", False)
        self.declare_parameter("require_increasing_odom_timestamp", False)
        self.declare_parameter("min_odom_quality", 0)

        # Optional axis-aligned operating envelope in the NED odometry frame.
        # It is disabled by default so existing simulation and legacy launches
        # retain their behavior. Bounds are nevertheless validated at startup,
        # preventing a later enable from activating malformed limits.
        self.declare_parameter("operating_bounds_enable", False)
        self.declare_parameter("operating_bounds_xmin_m", -1.0)
        self.declare_parameter("operating_bounds_xmax_m", 1.0)
        self.declare_parameter("operating_bounds_ymin_m", -1.0)
        self.declare_parameter("operating_bounds_ymax_m", 1.0)
        self.declare_parameter("operating_bounds_zmin_m", -1.0)
        self.declare_parameter("operating_bounds_zmax_m", 1.0)

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
        mission_enable_topic = self.get_parameter("mission_enable_topic").value
        controller_heartbeat_topic = self.get_parameter(
            "controller_heartbeat_topic"
        ).value

        self.operating_bounds_enabled = bool(
            self.get_parameter("operating_bounds_enable").value
        )
        (
            self.operating_bounds_min_ned,
            self.operating_bounds_max_ned,
        ) = _validated_operating_bounds(
            self.get_parameter("operating_bounds_xmin_m").value,
            self.get_parameter("operating_bounds_xmax_m").value,
            self.get_parameter("operating_bounds_ymin_m").value,
            self.get_parameter("operating_bounds_ymax_m").value,
            self.get_parameter("operating_bounds_zmin_m").value,
            self.get_parameter("operating_bounds_zmax_m").value,
        )

        self.sub_odom = self.create_subscription(VehicleOdometry, odom_topic, self.on_odom, px4_qos)
        self.sub_cm = self.create_subscription(VehicleControlMode, cm_topic, self.on_control_mode, px4_qos)
        self.sub_mission_enable = self.create_subscription(
            Bool, mission_enable_topic, self.on_mission_enable, 10
        )
        self.pub_thrust = self.create_publisher(VehicleThrustSetpoint, thrust_topic, px4_qos)
        self.pub_torque = self.create_publisher(VehicleTorqueSetpoint, torque_topic, px4_qos)
        self.pub_controller_heartbeat = self.create_publisher(
            Empty, controller_heartbeat_topic, 10
        )

        self.have_odom = False
        self.p_w = np.zeros(3)
        self.q_wxyz = (1.0, 0.0, 0.0, 0.0)
        self.v_b = np.zeros(3)
        self.w_b = np.zeros(3)
        self.enabled = False
        self.mission_enable = False
        self.mission_rearm_required = False
        self.last_control_mode_monotonic = None
        self.last_odom_sec = None
        self.odom_valid = False
        self.last_valid_position = None
        self.last_valid_quat_wxyz = None
        self.last_odom_timestamp_sample = None

        self.q_goal = euler_to_quat_wxyz(
            float(self.get_parameter("goal_roll").value),
            float(self.get_parameter("goal_pitch").value),
            float(self.get_parameter("goal_yaw").value),
        )

        self.ocp_solver = None
        self.u_force_cmd_N = np.zeros(3)
        self.u_tau_cmd_Nm = np.zeros(3)
        self.command_valid = False
        self.last_solution_sec = None
        self.last_solve_tick_monotonic = None
        self.x_guess = None
        self.u_guess = None
        self.N_horizon = int(self.get_parameter("N").value)

        self.traj_active = False
        self.traj_start_time_sec = 0.0
        self.traj_duration_sec = 0.0
        self.traj_start_pos = np.zeros(3)
        self.traj_goal_pos = np.zeros(3)
        self.traj_start_q_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.path_points = []
        self.path_segment_lengths = []
        self.path_total_length = 0.0
        self.traj_kind = "linear"
        self.last_goal_signature = None
        self.terminal_hold_goal_signature = None
        self._trajectory_reset_pending = True
        self.fixed_hook_projected_restart = False
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

        # These values are intentionally fixed when the OCP is built. Using the
        # same values at publish time keeps the optimizer and actuator command
        # limits consistent.
        self.thrust_sat_norm = self._normalized_saturation("thrust_sat")
        self.torque_sat_norm = self._normalized_saturation("torque_sat")
        self.force_axis_max_N = np.array([
            self._positive_limit("Fx_max_N"),
            self._positive_limit("Fy_max_N"),
            self._positive_limit("Fz_max_N"),
        ], dtype=float)
        self.torque_axis_max_Nm = np.array([
            self._positive_limit("Mx_max_Nm"),
            self._positive_limit("My_max_Nm"),
            self._positive_limit("Mz_max_Nm"),
        ], dtype=float)
        self.position_integral_gain = self._finite_nonnegative_parameter(
            "position_integral_gain_N_per_m_s"
        )
        self.position_integral_force_limit_fraction = (
            self._finite_fraction_parameter(
                "position_integral_force_limit_fraction"
            )
        )
        self.position_integral_activation_error_m = (
            self._finite_positive_parameter(
                "position_integral_activation_error_m"
            )
        )
        self.position_integral_max_dt_s = self._finite_positive_parameter(
            "position_integral_max_dt_s"
        )
        self.pre_approach_speed_mps = (
            self._finite_nonnegative_parameter(
                "pre_approach_speed_mps"
            )
        )
        self.final_approach_speed_mps = (
            self._finite_nonnegative_parameter(
                "final_approach_speed_mps"
            )
        )
        self.fixed_hook_depth_tolerance_m = (
            self._finite_nonnegative_parameter(
                "fixed_hook_depth_tolerance_m"
            )
        )
        self.position_integral_error_world = np.zeros(3, dtype=float)
        self.last_position_integral_update_sec = None

        self._build_mpc()

        self.get_logger().info(
            "acados tracking MPC "
            f"model_type={str(self.get_parameter('model_type').value)}, "
            f"robot_type={str(self.get_parameter('robot_type').value)}"
        )
        if bool(self.get_parameter("require_mission_enable").value):
            self.get_logger().info(
                f"Mission disabled; publish std_msgs/Bool true on {mission_enable_topic} to start."
            )
        else:
            self.get_logger().info("Mission-enable gate bypassed (legacy/simulation mode).")
        if self.operating_bounds_enabled:
            self.get_logger().info(
                "NED operating bounds enabled: "
                f"min={self.operating_bounds_min_ned.tolist()}, "
                f"max={self.operating_bounds_max_ned.tolist()}."
            )

        self.solve_timer = self.create_timer(
            1.0 / float(self.get_parameter("solve_rate_hz").value), self.solve_tick
        )
        self.pub_timer = self.create_timer(float(self.get_parameter("publish_dt").value), self.publish_tick)

    def on_control_mode(self, msg: VehicleControlMode):
        # Status freshness is a liveness property. A monotonic clock prevents
        # ROS time resets or /clock jumps from creating a false timeout.
        self.last_control_mode_monotonic = time.monotonic()
        gate = bool(msg.flag_armed) and bool(msg.flag_control_offboard_enabled)
        if gate and not self.enabled:
            self.enabled = True
            self._invalidate_command(reset_trajectory=True, publish_zero=True)
            if self._mission_allowed():
                self.get_logger().info(
                    "acados tracking MPC control gate active "
                    "(armed + offboard)."
                )
            else:
                self.get_logger().info(
                    "Armed + Offboard confirmed; waiting for mission enable "
                    "before publishing nonzero control."
                )
        elif not gate:
            was_enabled = self.enabled
            self.enabled = False
            if self._mission_allowed():
                self._handle_state_failure(
                    "Armed/Offboard control gate is not active."
                )
            else:
                self._invalidate_command(
                    reset_trajectory=True, publish_zero=True
                )
            if was_enabled:
                self.get_logger().info("acados tracking MPC disabled; zero command published.")

    def on_mission_enable(self, msg: Bool):
        was_allowed = self._mission_allowed()
        requested = bool(msg.data)

        if not requested:
            self.mission_enable = False
            if self.mission_rearm_required:
                self.mission_rearm_required = False
                self.get_logger().info(
                    "Mission safety latch cleared; publish true only after checking the cause."
                )
        elif self.mission_rearm_required:
            self.mission_enable = False
            self._invalidate_command(reset_trajectory=True, publish_zero=True)
            self.get_logger().warn(
                "Mission re-enable rejected by safety latch; publish false, inspect, then publish true.",
                throttle_duration_sec=1.0,
            )
            return
        else:
            self.mission_enable = True

        is_allowed = self._mission_allowed()

        if is_allowed and not was_allowed:
            # The actual trajectory is constructed by solve_tick only after
            # armed/Offboard and fresh valid odometry are all present.
            self._invalidate_command(reset_trajectory=True, publish_zero=True)
            self.get_logger().info(
                "Mission enabled; next valid solve will start from the current pose."
            )
        elif was_allowed and not is_allowed:
            self._invalidate_command(reset_trajectory=True, publish_zero=True)
            self.get_logger().info("Mission disabled; zero command published.")

    def on_odom(self, msg: VehicleOdometry):
        if bool(
            self.get_parameter("require_expected_odom_frames").value
        ) and (
            int(msg.pose_frame) != VehicleOdometry.POSE_FRAME_NED
            or int(msg.velocity_frame)
            != VehicleOdometry.VELOCITY_FRAME_BODY_FRD
        ):
            self._reject_odom(
                "Unexpected odometry frames: "
                f"pose_frame={int(msg.pose_frame)}, "
                f"velocity_frame={int(msg.velocity_frame)}."
            )
            return

        min_quality = int(self.get_parameter("min_odom_quality").value)
        if min_quality > 0 and int(msg.quality) < min_quality:
            self._reject_odom(
                f"Odometry quality {int(msg.quality)} is below "
                f"{min_quality}."
            )
            return

        timestamp_sample = int(msg.timestamp_sample)
        if bool(
            self.get_parameter("require_increasing_odom_timestamp").value
        ):
            if timestamp_sample <= 0:
                self._reject_odom(
                    "Odometry timestamp_sample is zero or missing."
                )
                return
            if (
                self.last_odom_timestamp_sample is not None
                and timestamp_sample <= self.last_odom_timestamp_sample
            ):
                self._reject_odom(
                    "Odometry timestamp_sample is duplicate or non-increasing."
                )
                return
        if timestamp_sample > 0:
            self.last_odom_timestamp_sample = timestamp_sample

        p_w = np.array([float(msg.position[0]), float(msg.position[1]), float(msg.position[2])], dtype=float)
        q_wxyz = np.array(
            [float(msg.q[0]), float(msg.q[1]), float(msg.q[2]), float(msg.q[3])],
            dtype=float,
        )
        v_b = np.array([float(msg.velocity[0]), float(msg.velocity[1]), float(msg.velocity[2])], dtype=float)
        w_b = np.array(
            [float(msg.angular_velocity[0]), float(msg.angular_velocity[1]), float(msg.angular_velocity[2])],
            dtype=float,
        )

        q_norm = float(np.linalg.norm(q_wxyz))
        if (
            not np.all(np.isfinite(p_w))
            or not np.all(np.isfinite(q_wxyz))
            or not np.all(np.isfinite(v_b))
            or not np.all(np.isfinite(w_b))
            or not math.isfinite(q_norm)
            or q_norm <= 1e-6
        ):
            self._reject_odom("Odometry contains a non-finite state.")
            return

        q_wxyz /= q_norm
        if self.operating_bounds_enabled and not (
            np.all(p_w >= self.operating_bounds_min_ned)
            and np.all(p_w <= self.operating_bounds_max_ned)
        ):
            self._reject_odom(
                "NED position is outside the configured operating bounds: "
                f"position={p_w.tolist()}, "
                f"min={self.operating_bounds_min_ned.tolist()}, "
                f"max={self.operating_bounds_max_ned.tolist()}.",
                latch_mission=True,
            )
            return

        max_tilt_rad = float(self.get_parameter("max_tilt_rad").value)
        if max_tilt_rad > 0.0:
            body_down_dot_world_down = 1.0 - 2.0 * (
                q_wxyz[1] * q_wxyz[1]
                + q_wxyz[2] * q_wxyz[2]
            )
            tilt_rad = math.acos(
                clamp(float(body_down_dot_world_down), -1.0, 1.0)
            )
            if tilt_rad > max_tilt_rad:
                self._reject_odom(
                    f"Odometry tilt {tilt_rad:.3f}rad exceeds "
                    f"{max_tilt_rad:.3f}rad."
                )
                return
        position_jump_m = 0.0
        orientation_jump_rad = 0.0
        jump_detected = False
        if (
            self.last_valid_position is not None
            and self.last_valid_quat_wxyz is not None
            and bool(self.get_parameter("require_mission_enable").value)
            and self.mission_enable
        ):
            position_jump_m = float(
                np.linalg.norm(p_w - self.last_valid_position)
            )
            orientation_jump_rad = quat_angular_distance_wxyz(
                q_wxyz, self.last_valid_quat_wxyz
            )
            max_position_jump_m = float(
                self.get_parameter("max_odom_position_jump_m").value
            )
            max_orientation_jump_rad = float(
                self.get_parameter("max_odom_orientation_jump_rad").value
            )
            jump_detected = (
                max_position_jump_m > 0.0
                and position_jump_m > max_position_jump_m
            ) or (
                max_orientation_jump_rad > 0.0
                and orientation_jump_rad > max_orientation_jump_rad
            )

        self.last_odom_sec = self._now_sec()
        self.p_w = p_w
        self.q_wxyz = tuple(q_wxyz.tolist())
        self.v_b = v_b
        self.w_b = w_b
        self.have_odom = True
        self.odom_valid = True
        self.last_valid_position = p_w.copy()
        self.last_valid_quat_wxyz = tuple(q_wxyz.tolist())

        if jump_detected:
            self._revoke_mission_enable(
                "Odometry jump detected "
                f"(position={position_jump_m:.3f}m, "
                f"orientation={orientation_jump_rad:.3f}rad)."
            )
            return

        if not hasattr(self, "_logged_frame_once"):
            self._logged_frame_once = True
            yaw = quat_to_yaw_wxyz(self.q_wxyz)
            self.get_logger().info(
                f"ODOM init: p=[{self.p_w[0]:.3f},{self.p_w[1]:.3f},{self.p_w[2]:.3f}] "
                f"v_b=[{self.v_b[0]:.3f},{self.v_b[1]:.3f},{self.v_b[2]:.3f}] yaw={yaw:.3f} rad"
            )

    def _reject_odom(self, reason, *, latch_mission=False):
        self.have_odom = False
        self.odom_valid = False
        if (
            latch_mission
            and bool(self.get_parameter("require_mission_enable").value)
        ):
            # A geometric boundary breach is a hard state failure. Latch even
            # during preflight so a later true message cannot start motion
            # without an explicit false -> inspect -> true recovery sequence.
            self._revoke_mission_enable(reason)
        else:
            self._handle_state_failure(reason)
        self.get_logger().error(
            f"Rejected odometry: {reason} Zero command published.",
            throttle_duration_sec=1.0,
        )

    def _build_mpc(self):
        Ts = float(self.get_parameter("Ts").value)
        N = int(self.get_parameter("N").value)
        self.N_horizon = N
        model_type = str(self.get_parameter("model_type").value).strip().lower()
        robot_type = str(self.get_parameter("robot_type").value).strip().lower()

        if model_type == "fossen":
            x_sym, u_sym, xdot_fun, _ = build_bluerov2_fossen_model_sim(Ts)
            model_name = "bluerov2_fossen_track"
            xdot_expr = xdot_fun(x_sym, u_sym)
        elif model_type == "fossen_real":
            x_sym, u_sym, xdot_fun, _ = build_bluerov2_fossen_model_real(
                Ts, robot_type=robot_type
            )
            model_name = f"bluerov2_fossen_real_{robot_type}_track"
            xdot_expr = xdot_fun(x_sym, u_sym)
        elif model_type in ("rigid", "rigid_body"):
            m = float(self.get_parameter("mass").value)
            Ix = float(self.get_parameter("Ix").value)
            Iy = float(self.get_parameter("Iy").value)
            Iz = float(self.get_parameter("Iz").value)
            x_sym, u_sym, xdot_expr = build_rigid_body_explicit_model(m, Ix, Iy, Iz)
            model_name = "bluerov2_rigid_body_track"
        else:
            raise ValueError(
                f"Unknown model_type {model_type!r}; expected "
                "'fossen', 'fossen_real', 'rigid', or 'rigid_body'."
            )

        p_sym = ca.SX.sym("p", 8)
        pref = p_sym[0:3]
        qref = p_sym[3:7]
        hold_att_flag = p_sym[7]

        model = AcadosModel()
        model.name = model_name
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
        ocp.solver_options.N_horizon = N
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

        force_limits_N = self.thrust_sat_norm * self.force_axis_max_N
        torque_limits_Nm = self.torque_sat_norm * self.torque_axis_max_Nm
        command_limits = np.concatenate([force_limits_N, torque_limits_Nm])

        # The OCP and PX4 publisher must represent the same actuator envelope.
        # Previously the OCP optimized against 100% physical authority while
        # publish_tick silently clipped the result to thrust_sat/torque_sat.
        lbu = -command_limits
        ubu = command_limits
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

        self.get_logger().info(
            "OCP command limits match publisher saturation: "
            f"force={force_limits_N.tolist()} N, torque={torque_limits_Nm.tolist()} Nm"
        )

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

    def _pre_approach_waypoint_enabled(self):
        return bool(
            self.get_parameter("use_pre_approach_waypoint").value
        )

    def _pre_approach_position(self):
        return np.array([
            float(self.get_parameter("pre_approach_x").value),
            float(self.get_parameter("pre_approach_y").value),
            float(self.get_parameter("pre_approach_z").value),
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
            if (
                self._pre_approach_waypoint_enabled()
                and self.mission_state
                in ("PRE_APPROACH", "RETREAT", "COMPLETE")
            ):
                self.active_goal_pos = self._pre_approach_position()
            else:
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
                self.active_goal_pos = p_pass
                self.active_goal_yaw = yaw_pass
                self.state_enter_time_sec = self._now_sec()
                self.forward_pass_start_time_sec = self.state_enter_time_sec
                self.forward_pass_start_pos = self.p_w.copy()
                self.get_logger().info("Mission -> FORWARD_PASS")

        elif self.mission_state == "FORWARD_PASS":
            self.active_goal_pos = p_pass
            self.active_goal_yaw = yaw_pass
            if self._at_active_goal():
                self.mission_state = "BACKWARD_PASS"
                self.active_goal_pos = p_back
                self.active_goal_yaw = yaw_back
                self.state_enter_time_sec = self._now_sec()
                self.get_logger().info("Mission -> BACKWARD_PASS")
            elif self._should_abort_forward_due_to_contact(yaw_pass):
                self.mission_state = "BACKWARD_PASS"
                self.active_goal_pos = p_back
                self.active_goal_yaw = yaw_back
                self.state_enter_time_sec = self._now_sec()
                self.get_logger().info("Mission -> BACKWARD_PASS (contact heuristic)")

        elif self.mission_state == "BACKWARD_PASS":
            self.active_goal_pos = p_back
            self.active_goal_yaw = yaw_back
            if self._at_active_goal():
                self.mission_state = "RETURN_SHORE"
                self.active_goal_pos = p_home
                self.active_goal_yaw = yaw_home
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

    def _normalized_saturation(self, parameter_name):
        value = float(self.get_parameter(parameter_name).value)
        if not math.isfinite(value):
            raise ValueError(f"{parameter_name} must be finite, got {value!r}")
        return clamp(abs(value), 0.0, 1.0)

    def _positive_limit(self, parameter_name):
        value = float(self.get_parameter(parameter_name).value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{parameter_name} must be finite and > 0, got {value!r}")
        return value

    def _finite_nonnegative_parameter(self, parameter_name):
        value = float(self.get_parameter(parameter_name).value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"{parameter_name} must be finite and >= 0, got {value!r}"
            )
        return value

    def _finite_positive_parameter(self, parameter_name):
        value = float(self.get_parameter(parameter_name).value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{parameter_name} must be finite and > 0, got {value!r}"
            )
        return value

    def _finite_fraction_parameter(self, parameter_name):
        value = self._finite_nonnegative_parameter(parameter_name)
        if value > 1.0:
            raise ValueError(
                f"{parameter_name} must be <= 1, got {value!r}"
            )
        return value

    def _mission_allowed(self):
        if not bool(self.get_parameter("require_mission_enable").value):
            return True
        return self.mission_enable and not self.mission_rearm_required

    def _revoke_mission_enable(self, reason):
        self.mission_enable = False
        self.mission_rearm_required = True
        self._invalidate_command(reset_trajectory=True, publish_zero=True)
        self.get_logger().error(
            f"{reason} Mission enable revoked; publish false, inspect, then publish true."
        )

    def _control_mode_feedback_fresh(self, now_monotonic=None):
        if self.last_control_mode_monotonic is None:
            return False
        timeout_s = float(self.get_parameter("control_mode_timeout_s").value)
        if timeout_s <= 0.0:
            return True
        if now_monotonic is None:
            now_monotonic = time.monotonic()
        age_s = float(now_monotonic) - self.last_control_mode_monotonic
        if 0.0 <= age_s <= timeout_s:
            return True
        self.get_logger().warn(
            f"VehicleControlMode stale for {age_s:.2f}s; publishing zero control.",
            throttle_duration_sec=1.0,
        )
        return False

    def _control_gate_active(self):
        return self.enabled and self._control_mode_feedback_fresh()

    def _controller_heartbeat_ready(self):
        """Return true only while the external Offboard manager may stream."""
        if self.ocp_solver is None or self.mission_rearm_required:
            return False
        if not self._control_mode_feedback_fresh():
            return False
        if not self._odom_fresh() or not self._state_valid(self._x_meas()):
            return False

        require_gate = bool(
            self.get_parameter("require_mission_enable").value
        )
        if require_gate and not self.mission_enable:
            # Preflight: state estimation and PX4 feedback are healthy, but the
            # controller is intentionally not allowed to move yet.
            return True
        return (
            self._mission_allowed()
            and self.enabled
            and self._command_fresh()
        )

    def _handle_state_failure(self, reason):
        if (
            bool(
                self.get_parameter(
                    "revoke_mission_on_state_failure"
                ).value
            )
            and bool(self.get_parameter("require_mission_enable").value)
            and self._mission_allowed()
        ):
            self._revoke_mission_enable(reason)
            return
        self._invalidate_command(reset_trajectory=True, publish_zero=True)

    def _state_valid(self, state):
        if state.shape != (13,) or not np.all(np.isfinite(state)):
            return False
        q_norm = float(np.linalg.norm(state[3:7]))
        return math.isfinite(q_norm) and q_norm > 1e-6

    def _command_fresh(self):
        if not self.command_valid or self.last_solution_sec is None:
            return False
        timeout_s = float(self.get_parameter("command_timeout_s").value)
        if timeout_s <= 0.0:
            return True
        age_s = self._now_sec() - self.last_solution_sec
        if 0.0 <= age_s <= timeout_s:
            return True
        self.get_logger().warn(
            f"MPC command stale for {age_s:.2f}s; publishing zero control.",
            throttle_duration_sec=1.0,
        )
        return False

    def _odom_fresh(self):
        if not self.have_odom or not self.odom_valid or self.last_odom_sec is None:
            return False
        timeout_s = float(self.get_parameter("odom_timeout_s").value)
        if timeout_s <= 0.0:
            return True
        age_s = self._now_sec() - self.last_odom_sec
        if 0.0 <= age_s <= timeout_s:
            return True
        self.get_logger().warn(
            f"Odometry stale for {age_s:.2f}s; publishing zero control.",
            throttle_duration_sec=1.0,
        )
        return False

    def _zero_command_cache(self):
        self.u_force_cmd_N[:] = 0.0
        self.u_tau_cmd_Nm[:] = 0.0

    def _reset_position_integral(self):
        self.position_integral_error_world[:] = 0.0
        self.last_position_integral_update_sec = None

    def _position_integral_reference_finished(self, now_sec):
        if not self.traj_active:
            return True
        return now_sec >= (
            self.traj_start_time_sec + self.traj_duration_sec
        )

    def _force_with_position_integral(self, mpc_force_body, now_sec):
        """Add bounded offset-free force without bypassing actuator limits."""
        total_limits = self.thrust_sat_norm * self.force_axis_max_N
        base_force = np.clip(
            np.asarray(mpc_force_body, dtype=float),
            -total_limits,
            total_limits,
        )

        gain = self.position_integral_gain
        bias_fraction = self.position_integral_force_limit_fraction
        if gain <= 0.0 or bias_fraction <= 0.0:
            self._reset_position_integral()
            return base_force

        reference_finished = self._position_integral_reference_finished(
            now_sec
        )
        fixed_hook_transit = (
            not reference_finished
            and self._fixed_hook_transit_active()
        )
        if not reference_finished and not fixed_hook_transit:
            # Do not integrate normal lag behind a moving reference.  The
            # fixed-hook horizontal segment is the sole exception below: it
            # freezes and reuses only the previously learned world-Z bias.
            self._reset_position_integral()
            return base_force

        if fixed_hook_transit:
            # The required heave compensation should not disappear merely
            # because the XY reference starts moving.  Freeze it (do not
            # integrate endpoint error during transit), and keep XY bias zero.
            self.position_integral_error_world[0:2] = 0.0
            self.last_position_integral_update_sec = None
        else:
            position_error_world = self._goal_position() - self.p_w
            error_norm = float(np.linalg.norm(position_error_world))
            if (
                not np.all(np.isfinite(position_error_world))
                or error_norm > self.position_integral_activation_error_m
            ):
                # A large error is more likely a bad reference, obstruction,
                # or frame problem.  Do not let the integral mask it.
                self._reset_position_integral()
                return base_force

            if self.last_position_integral_update_sec is None:
                self.last_position_integral_update_sec = float(now_sec)
            else:
                dt = float(now_sec) - self.last_position_integral_update_sec
                self.last_position_integral_update_sec = float(now_sec)
                if math.isfinite(dt) and dt > 0.0:
                    dt = min(dt, self.position_integral_max_dt_s)
                    self.position_integral_error_world += (
                        position_error_world * dt
                    )

        rotation_body_to_world = quat_to_rotation_matrix_wxyz(
            self.q_wxyz
        )
        bias_body = rotation_body_to_world.T @ (
            gain * self.position_integral_error_world
        )
        bias_limits = bias_fraction * self.force_axis_max_N
        bias_body = np.clip(bias_body, -bias_limits, bias_limits)

        # The bias and nominal MPC force share one physical envelope.  This
        # second clip is essential: the integrator must never add authority on
        # top of thrust_sat.  Back-calculation stores only the bias that can
        # actually be delivered, preventing wind-up at the total-force limit.
        combined_force = np.clip(
            base_force + bias_body,
            -total_limits,
            total_limits,
        )
        delivered_bias_body = combined_force - base_force
        self.position_integral_error_world[:] = (
            rotation_body_to_world @ delivered_bias_body
        ) / gain
        return combined_force

    def _invalidate_command(self, reset_trajectory, publish_zero):
        self._zero_command_cache()
        self.command_valid = False
        self.last_solution_sec = None
        if reset_trajectory:
            self.traj_active = False
            self.last_goal_signature = None
            self.terminal_hold_goal_signature = None
            self._trajectory_reset_pending = True
            self.fixed_hook_projected_restart = False
            self._reset_position_integral()
        if publish_zero:
            self.publish_zero()

    def _restart_trajectory_from_current_state(self, x0):
        self.traj_active = False
        self.last_goal_signature = None
        self.terminal_hold_goal_signature = None
        self.forward_pass_start_pos = None
        self.forward_pass_start_time_sec = 0.0
        self.fixed_hook_projected_restart = False

        if bool(self.get_parameter("use_box_recovery_mission").value):
            self.mission_state = "INIT"
            self.state_enter_time_sec = 0.0
            self.home_pos = None
            self.home_yaw = 0.0
        else:
            if self._pre_approach_waypoint_enabled():
                self.mission_state = "PRE_APPROACH"
                self.active_goal_pos = self._pre_approach_position()
            else:
                self.active_goal_pos = self._goal_position_static()
            self.active_goal_yaw = self._goal_yaw_static()

        self._update_mission()
        initial_goal = self._goal_position()
        if not np.all(np.isfinite(initial_goal)):
            if bool(self.get_parameter("require_mission_enable").value):
                self._revoke_mission_enable("Initial goal is non-finite.")
                return False
            raise ValueError("initial goal is non-finite")

        initial_distance_m = float(np.linalg.norm(initial_goal - self.p_w))
        max_initial_distance_m = float(
            self.get_parameter("max_initial_goal_distance_m").value
        )
        if (
            bool(self.get_parameter("require_mission_enable").value)
            and max_initial_distance_m > 0.0
            and initial_distance_m > max_initial_distance_m
        ):
            self._revoke_mission_enable(
                f"Initial goal distance {initial_distance_m:.3f}m exceeds "
                f"max_initial_goal_distance_m={max_initial_distance_m:.3f}m."
            )
            return False

        initial_orientation_error_rad = quat_angular_distance_wxyz(
            self.q_wxyz,
            self._goal_quaternion(),
        )
        max_initial_orientation_error_rad = float(
            self.get_parameter(
                "max_initial_goal_orientation_error_rad"
            ).value
        )
        if (
            bool(self.get_parameter("require_mission_enable").value)
            and max_initial_orientation_error_rad > 0.0
            and initial_orientation_error_rad
            > max_initial_orientation_error_rad
        ):
            self._revoke_mission_enable(
                "Initial goal orientation error "
                f"{initial_orientation_error_rad:.3f}rad exceeds "
                "max_initial_goal_orientation_error_rad="
                f"{max_initial_orientation_error_rad:.3f}rad."
            )
            return False

        self._reset_trajectory_from_current_pose()

        # Do not let a warm start from a previous enable period influence the
        # first solve of a new operator-approved run.
        self.x_guess[:, :] = x0.reshape(1, -1)
        self.u_guess[:, :] = 0.0
        self._trajectory_reset_pending = False
        return True

    def _phase_traj_speed(self):
        if (
            self.mission_state == "PRE_APPROACH"
            and self.pre_approach_speed_mps > 0.0
        ):
            return self.pre_approach_speed_mps
        if (
            self.mission_state == "FINAL_APPROACH"
            and self.final_approach_speed_mps > 0.0
        ):
            return self.final_approach_speed_mps
        if self.mission_state == "FORWARD_PASS":
            return float(self.get_parameter("forward_pass_speed_mps").value)
        if self.mission_state in ("BACKWARD_PASS", "RETREAT"):
            return float(self.get_parameter("backward_pass_speed_mps").value)
        return float(self.get_parameter("traj_speed_mps").value)

    def _fixed_hook_transit_active(self):
        return (
            getattr(self, "mission_state", "")
            in ("FINAL_APPROACH", "RETREAT")
            and not bool(
                self.get_parameter("use_box_recovery_mission").value
            )
            and self._pre_approach_waypoint_enabled()
        )

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
        fixed_hook_line = self._fixed_hook_line_segment()
        if fixed_hook_line is None:
            self._reset_position_integral()
            self.fixed_hook_projected_restart = False
            self.traj_start_pos = self.p_w.copy()
            self.traj_goal_pos = goal_pos.copy()
            self.traj_start_q_wxyz = np.asarray(
                self.q_wxyz,
                dtype=float,
            ).copy()
        else:
            line_start, line_goal = fixed_hook_line
            if not np.allclose(goal_pos, line_goal, atol=1e-6, rtol=0.0):
                raise ValueError(
                    "fixed-hook state goal does not match its horizontal "
                    "line endpoint"
                )
            line_delta = line_goal - line_start
            line_length_sq = float(np.dot(line_delta, line_delta))
            if line_length_sq <= 1e-12:
                raise ValueError(
                    "fixed-hook pre-approach and hook poses must be distinct"
                )
            if self.fixed_hook_projected_restart:
                # If FINAL_HOLD drifted just outside tolerance, recover from
                # the closest point instead of replaying the entire line.
                progress = clamp(
                    float(np.dot(self.p_w - line_start, line_delta))
                    / line_length_sq,
                    0.0,
                    1.0,
                )
                self.traj_start_pos = line_start + progress * line_delta
            else:
                # Normal forward and reverse stages use the exact same pair
                # of configured endpoints, in opposite directions.
                self.traj_start_pos = line_start.copy()
            self.fixed_hook_projected_restart = False
            self.traj_goal_pos = line_goal.copy()
            # PRE_APPROACH already requires the recorded attitude.  Keep that
            # attitude fixed for both the forward and reverse straight line.
            self.traj_start_q_wxyz = np.asarray(
                self._goal_quaternion(),
                dtype=float,
            ).copy()
            # Carry only the learned world-Z compensation into horizontal
            # transit.  Cross-track XY bias belongs to the previous waypoint
            # and must not steer the nominal straight line.
            self.position_integral_error_world[0:2] = 0.0
            self.last_position_integral_update_sec = None
        self.traj_start_time_sec = self._now_sec()
        dist = float(np.linalg.norm(self.traj_goal_pos - self.traj_start_pos))
        speed = max(self._phase_traj_speed(), 1e-4)
        angular_speed = float(
            self.get_parameter("traj_angular_speed_rad_s").value
        )
        if not math.isfinite(angular_speed) or angular_speed < 0.0:
            raise ValueError(
                "traj_angular_speed_rad_s must be finite and >= 0"
            )
        angular_duration = 0.0
        if angular_speed > 0.0:
            angular_distance = quat_angular_distance_wxyz(
                self.traj_start_q_wxyz,
                self._goal_quaternion(),
            )
            angular_duration = angular_distance / angular_speed
        min_duration = max(float(self.get_parameter("min_traj_duration_s").value), float(self.get_parameter("Ts").value))
        self.traj_duration_sec = max(
            dist / speed,
            angular_duration,
            min_duration,
        )
        self.traj_active = True
        self.traj_kind = "linear"
        self.path_points = []
        self.path_segment_lengths = []
        self.path_total_length = 0.0
        self.terminal_hold_goal_signature = None
        self.last_goal_signature = self._goal_signature()
        self.get_logger().info(
            f"New linear trajectory: start={self.traj_start_pos}, goal={self.traj_goal_pos}, duration={self.traj_duration_sec:.2f}s"
        )

    def _fixed_hook_line_segment(self):
        """Return the nominal level segment for final approach or retreat."""
        if not self._fixed_hook_transit_active():
            return None

        hook = self._goal_position_static()
        pre_approach = self._pre_approach_position()
        if not np.all(np.isfinite(hook)) or not np.all(
            np.isfinite(pre_approach)
        ):
            raise ValueError("fixed-hook line endpoints must be finite")
        if abs(float(pre_approach[2] - hook[2])) > 1e-6:
            raise ValueError(
                "fixed-hook pre-approach and hook poses must have the same "
                "NED depth"
            )

        if self.mission_state == "FINAL_APPROACH":
            return pre_approach, hook
        return hook, pre_approach

    def _set_path_trajectory(self, path_xy, goal_z):
        if len(path_xy) < 2:
            raise ValueError("A* path trajectory requires at least two points")

        self._reset_position_integral()

        self.traj_start_time_sec = self._now_sec()
        self.traj_start_q_wxyz = np.asarray(self.q_wxyz, dtype=float).copy()

        self.path_points = [
            np.array([float(x), float(y), float(goal_z)], dtype=float)
            for x, y in path_xy
        ]
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
        self.traj_duration_sec = max(
            self.path_total_length / speed,
            min_duration,
        )
        self.traj_active = True
        self.traj_kind = "path"
        self.terminal_hold_goal_signature = None
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

        if self._update_fixed_hook_hold_phase():
            return

        if getattr(self, "mission_state", "") == "DONE":
            self.traj_active = False
            return

        current_goal_sig = self._goal_signature()
        regenerate = bool(self.get_parameter("regenerate_on_goal_change").value)
        goal_changed = (self.last_goal_signature is None) or (current_goal_sig != self.last_goal_signature)

        # Once a goal is reached, keep publishing that complete terminal goal.
        # A changed goal must still start a smooth trajectory, including when
        # regenerate_on_goal_change is false: that flag only controls changes
        # while an existing trajectory is active.
        if self.terminal_hold_goal_signature is not None:
            if current_goal_sig == self.terminal_hold_goal_signature:
                self.traj_active = False
                return
            self.terminal_hold_goal_signature = None
            self._reset_trajectory_from_current_pose()
            return

        if regenerate and goal_changed:
            self._reset_trajectory_from_current_pose()
            return

        if not self.traj_active:
            self._reset_trajectory_from_current_pose()
            return

        if self._trajectory_completion_reached():
            if (
                getattr(self, "mission_state", "") == "PRE_APPROACH"
                and self._advance_pre_approach_phase()
            ):
                self.terminal_hold_goal_signature = None
                self._reset_trajectory_from_current_pose()
                return
            if self._begin_fixed_hook_final_hold(current_goal_sig):
                return
            if self._complete_fixed_hook_retreat(current_goal_sig):
                return
            self._enter_terminal_hold(current_goal_sig)

    def _begin_fixed_hook_final_hold(self, goal_signature):
        if (
            bool(self.get_parameter("use_box_recovery_mission").value)
            or self.mission_state != "FINAL_APPROACH"
        ):
            return False

        hold_s = float(self.get_parameter("final_pose_hold_s").value)
        retreat_enabled = bool(
            self.get_parameter(
                "return_to_pre_approach_after_hold"
            ).value
        )
        if not math.isfinite(hold_s) or hold_s < 0.0:
            raise ValueError("final_pose_hold_s must be finite and >= 0")
        if hold_s <= 0.0 and not retreat_enabled:
            return False

        self.mission_state = "FINAL_HOLD"
        self.state_enter_time_sec = self._now_sec()
        self._enter_terminal_hold(goal_signature)
        self.get_logger().info(
            "Recorded hook pose reached; mission -> FINAL_HOLD "
            f"for {hold_s:.2f}s."
        )
        return True

    def _update_fixed_hook_hold_phase(self):
        if (
            bool(self.get_parameter("use_box_recovery_mission").value)
            or self.mission_state != "FINAL_HOLD"
        ):
            return False

        if not self._trajectory_completion_reached():
            self.mission_state = "FINAL_APPROACH"
            self.state_enter_time_sec = self._now_sec()
            self.terminal_hold_goal_signature = None
            self.last_goal_signature = None
            self.fixed_hook_projected_restart = True
            self._reset_trajectory_from_current_pose()
            self.get_logger().warning(
                "Drifted outside the recorded hook-pose tolerance; "
                "hold timer reset and mission -> FINAL_APPROACH."
            )
            return True

        hold_s = float(self.get_parameter("final_pose_hold_s").value)
        if not math.isfinite(hold_s) or hold_s < 0.0:
            raise ValueError("final_pose_hold_s must be finite and >= 0")
        if self._now_sec() - self.state_enter_time_sec < hold_s:
            return True

        retreat_enabled = bool(
            self.get_parameter(
                "return_to_pre_approach_after_hold"
            ).value
        )
        if retreat_enabled:
            if not self._pre_approach_waypoint_enabled():
                raise ValueError(
                    "return_to_pre_approach_after_hold requires "
                    "use_pre_approach_waypoint=true"
                )
            self.mission_state = "RETREAT"
            self.state_enter_time_sec = self._now_sec()
            self.terminal_hold_goal_signature = None
            self.last_goal_signature = None
            self._update_mission()
            self._reset_trajectory_from_current_pose()
            self.get_logger().info(
                "Final-pose hold complete; mission -> RETREAT along the "
                "straight line to the pre-approach pose."
            )
        else:
            self.mission_state = "FINAL_COMPLETE"
            self.state_enter_time_sec = self._now_sec()
            self.get_logger().info(
                "Final-pose hold complete; mission -> FINAL_COMPLETE"
            )
        return True

    def _complete_fixed_hook_retreat(self, goal_signature):
        if (
            bool(self.get_parameter("use_box_recovery_mission").value)
            or self.mission_state != "RETREAT"
        ):
            return False
        self.mission_state = "COMPLETE"
        self.state_enter_time_sec = self._now_sec()
        self._enter_terminal_hold(goal_signature)
        self.get_logger().info(
            "Pre-approach pose reached after retreat; mission -> COMPLETE."
        )
        return True

    def _advance_pre_approach_phase(self):
        if (
            bool(self.get_parameter("use_box_recovery_mission").value)
            or not self._pre_approach_waypoint_enabled()
            or self.mission_state != "PRE_APPROACH"
        ):
            return False

        self.mission_state = "FINAL_APPROACH"
        self.state_enter_time_sec = self._now_sec()
        self.active_goal_pos = self._goal_position_static()
        self.active_goal_yaw = self._goal_yaw_static()
        self.get_logger().info(
            "Pre-approach pose reached; mission -> FINAL_APPROACH"
        )
        return True

    def _trajectory_completion_reached(self):
        goal_tol = max(
            float(self.get_parameter("goal_reached_tol_m").value),
            1e-4,
        )
        if np.linalg.norm(self.p_w - self.traj_goal_pos) > goal_tol:
            return False

        fixed_hook_depth_states = (
            "PRE_APPROACH",
            "FINAL_APPROACH",
            "FINAL_HOLD",
            "RETREAT",
            "COMPLETE",
        )
        if (
            not bool(
                self.get_parameter("use_box_recovery_mission").value
            )
            and getattr(self, "mission_state", "")
            in fixed_hook_depth_states
            and getattr(self, "fixed_hook_depth_tolerance_m", 0.0) > 0.0
            and abs(float(self.p_w[2] - self.traj_goal_pos[2]))
            > getattr(self, "fixed_hook_depth_tolerance_m", 0.0)
        ):
            return False

        pre_approach_requires_attitude = (
            getattr(self, "mission_state", "") == "PRE_APPROACH"
            and not bool(
                self.get_parameter("use_box_recovery_mission").value
            )
            and self._pre_approach_waypoint_enabled()
        )
        if (
            bool(self.get_parameter("hold_attitude").value)
            or pre_approach_requires_attitude
        ):
            orientation_tol = max(
                float(
                    self.get_parameter(
                        "goal_reached_orientation_tol_rad"
                    ).value
                ),
                1e-4,
            )
            orientation_error = quat_angular_distance_wxyz(
                self.q_wxyz,
                self._goal_quaternion(),
            )
            if orientation_error > orientation_tol:
                return False

        if bool(self.get_parameter("use_box_recovery_mission").value):
            return self._at_active_goal()
        return True

    def _enter_terminal_hold(self, goal_signature):
        self.traj_active = False
        self.last_goal_signature = goal_signature
        self.terminal_hold_goal_signature = goal_signature
        self.get_logger().info(
            "Trajectory goal reached; entering terminal hold."
        )

    def _trajectory_stage_param(self, k: int):
        q_goal = self._goal_quaternion()
        hold_att_flag = 1.0 if bool(self.get_parameter("hold_attitude").value) else 0.0

        if not self.traj_active:
            pref = self._goal_position()
            return np.concatenate([pref, q_goal, np.array([hold_att_flag], dtype=float)])

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

        qref = quat_slerp_wxyz(
            self.traj_start_q_wxyz,
            q_goal,
            alpha,
        )
        return np.concatenate([pref, qref, np.array([hold_att_flag], dtype=float)])

    def _forceN_to_thrust_norm(self, F_N):
        return np.asarray(F_N, dtype=float) / self.force_axis_max_N

    def _torqueNm_to_torque_norm(self, tau_Nm):
        return np.asarray(tau_Nm, dtype=float) / self.torque_axis_max_Nm

    def solve_tick(self):
        solve_started_monotonic = time.monotonic()
        max_solve_gap_s = float(
            self.get_parameter("max_solve_gap_s").value
        )
        if (
            self.last_solve_tick_monotonic is not None
            and max_solve_gap_s > 0.0
            and solve_started_monotonic - self.last_solve_tick_monotonic
            > max_solve_gap_s
            and self._mission_allowed()
        ):
            gap_s = (
                solve_started_monotonic - self.last_solve_tick_monotonic
            )
            self.last_solve_tick_monotonic = solve_started_monotonic
            self._handle_solver_failure(
                f"Controller solve callback gap {gap_s:.3f}s exceeds "
                f"{max_solve_gap_s:.3f}s."
            )
            return
        self.last_solve_tick_monotonic = solve_started_monotonic

        if self.ocp_solver is None:
            self._invalidate_command(reset_trajectory=True, publish_zero=True)
            return
        if not self._control_gate_active():
            self._handle_state_failure(
                "Armed/Offboard control mode is absent or stale."
            )
            return
        if not self._mission_allowed():
            self._invalidate_command(reset_trajectory=True, publish_zero=True)
            return
        if not self._odom_fresh():
            self._handle_state_failure("Odometry is absent or stale.")
            return

        try:
            x0 = self._x_meas()
            if not self._state_valid(x0):
                self._handle_state_failure(
                    "Measured state is non-finite or has an invalid quaternion."
                )
                return
            x0[3:7] /= np.linalg.norm(x0[3:7])

            if self._trajectory_reset_pending:
                if not self._restart_trajectory_from_current_state(x0):
                    return
                self.get_logger().info(
                    f"Trajectory restarted from current pose {self.p_w.tolist()}."
                )
            else:
                self._maybe_refresh_trajectory()

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
            solve_duration_s = time.monotonic() - solve_started_monotonic
            max_solve_duration_s = float(
                self.get_parameter("max_solve_duration_s").value
            )
            if (
                max_solve_duration_s > 0.0
                and solve_duration_s > max_solve_duration_s
            ):
                self._handle_solver_failure(
                    f"acados solve duration {solve_duration_s:.3f}s exceeds "
                    f"{max_solve_duration_s:.3f}s."
                )
                return
            if status != 0:
                self.get_logger().warn(f"acados solve failed, status={status}")
                self._handle_solver_failure(
                    f"acados solve failed with status={status}."
                )
                return

            # Read and validate the complete solution before committing either
            # the command or warm start. A partial/NaN solution must never leave
            # an earlier nonzero command in the publisher cache.
            x_new = np.zeros_like(self.x_guess)
            u_new = np.zeros_like(self.u_guess)
            for k in range(self.N_horizon + 1):
                xk = np.array(self.ocp_solver.get(k, "x"), dtype=float).reshape(-1)
                if xk.shape != (13,) or not np.all(np.isfinite(xk)):
                    raise ValueError(f"invalid acados state at stage {k}")
                q_norm = float(np.linalg.norm(xk[3:7]))
                if not math.isfinite(q_norm) or q_norm <= 1e-6:
                    raise ValueError(f"invalid acados quaternion at stage {k}")
                xk[3:7] /= q_norm
                x_new[k, :] = xk
            for k in range(self.N_horizon):
                uk = np.array(self.ocp_solver.get(k, "u"), dtype=float).reshape(-1)
                if uk.shape != (6,) or not np.all(np.isfinite(uk)):
                    raise ValueError(f"invalid acados control at stage {k}")
                u_new[k, :] = uk

            u0 = u_new[0, :]
            self.x_guess[:-1, :] = x_new[1:, :]
            self.x_guess[-1, :] = x_new[-1, :]
            self.u_guess[:-1, :] = u_new[1:, :]
            self.u_guess[-1, :] = u_new[-1, :]
            self.u_force_cmd_N[:] = self._force_with_position_integral(
                u0[0:3],
                self._now_sec(),
            )
            self.u_tau_cmd_Nm[:] = u0[3:6]
            self.last_solution_sec = self._now_sec()
            self.command_valid = True
        except Exception as exc:
            self.get_logger().error(
                f"MPC solve exception: {type(exc).__name__}: {exc}; zero command published.",
                throttle_duration_sec=1.0,
            )
            self._handle_solver_failure(
                f"MPC solve exception: {type(exc).__name__}: {exc}."
            )

    def _handle_solver_failure(self, reason):
        if (
            bool(
                self.get_parameter(
                    "revoke_mission_on_solver_failure"
                ).value
            )
            and bool(self.get_parameter("require_mission_enable").value)
        ):
            self._revoke_mission_enable(reason)
            return
        self._invalidate_command(reset_trajectory=True, publish_zero=True)

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
        # This pulse proves both executor liveness and state-chain readiness.
        # It stops on stale PX4 feedback, stale/invalid odometry, a safety
        # latch, stale command during a mission, or a blocked native solve.
        if self._controller_heartbeat_ready():
            self.pub_controller_heartbeat.publish(Empty())
        if not self._control_gate_active():
            self._handle_state_failure(
                "Armed/Offboard control mode is absent or stale."
            )
            return
        if not self._mission_allowed():
            self._invalidate_command(reset_trajectory=True, publish_zero=True)
            return
        if not self._odom_fresh():
            self._handle_state_failure("Odometry is absent or stale.")
            return
        if not self._state_valid(self._x_meas()):
            self.get_logger().error(
                "Invalid measured state at publish time; zero command published.",
                throttle_duration_sec=1.0,
            )
            self._handle_state_failure("Measured state became invalid.")
            return
        if not self._command_fresh():
            self._invalidate_command(reset_trajectory=True, publish_zero=True)
            return

        try:
            if (
                not np.all(np.isfinite(self.u_force_cmd_N))
                or not np.all(np.isfinite(self.u_tau_cmd_Nm))
            ):
                raise ValueError("cached wrench is non-finite")

            now_us = int(self.get_clock().now().nanoseconds / 1000)

            thr_norm = self._forceN_to_thrust_norm(self.u_force_cmd_N)
            thr_norm = np.array([
                clamp(thr_norm[0], -self.thrust_sat_norm, self.thrust_sat_norm),
                clamp(thr_norm[1], -self.thrust_sat_norm, self.thrust_sat_norm),
                clamp(thr_norm[2], -self.thrust_sat_norm, self.thrust_sat_norm),
            ], dtype=float)

            thr = VehicleThrustSetpoint()
            thr.timestamp = now_us
            thr.timestamp_sample = 0
            thr.xyz = [float(thr_norm[0]), float(thr_norm[1]), float(thr_norm[2])]
            self.pub_thrust.publish(thr)

            tau_norm = self._torqueNm_to_torque_norm(self.u_tau_cmd_Nm)
            tau_norm = np.array([
                clamp(tau_norm[0], -self.torque_sat_norm, self.torque_sat_norm),
                clamp(tau_norm[1], -self.torque_sat_norm, self.torque_sat_norm),
                clamp(tau_norm[2], -self.torque_sat_norm, self.torque_sat_norm),
            ], dtype=float)

            tor = VehicleTorqueSetpoint()
            tor.timestamp = now_us
            tor.timestamp_sample = 0
            tor.xyz = [float(tau_norm[0]), float(tau_norm[1]), float(tau_norm[2])]
            self.pub_torque.publish(tor)
        except Exception as exc:
            self.get_logger().error(
                f"Command publish exception: {type(exc).__name__}: {exc}; zero command published.",
                throttle_duration_sec=1.0,
            )
            self._invalidate_command(reset_trajectory=True, publish_zero=True)


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
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
