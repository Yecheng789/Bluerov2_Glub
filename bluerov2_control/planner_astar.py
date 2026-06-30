from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

Point2 = Tuple[float, float]
GridIdx = Tuple[int, int]
Rect = Tuple[float, float, float, float]  # xmin, xmax, ymin, ymax


@dataclass(frozen=True)
class WorldInfo:
    world_sdf_path: str
    tank_model_sdf_path: Optional[str]
    tank_pose_xyzrpy: Tuple[float, float, float, float, float, float]
    payload_box_pose_xyzrpy: Tuple[float, float, float, float, float, float]
    water_surface_z: Optional[float]
    tank_inner_bounds_xy: Optional[Rect]
    tank_floor_z: Optional[float]
    tank_ceiling_z: Optional[float]


@dataclass(frozen=True)
class PlannerConfig:
    fallback_bounds: Rect = (-2.6, 2.6, -2.6, 2.6)

    resolution: float = 0.05

    robot_radius: float = 0.20
    obstacle_margin: float = 0.10

    payload_box_half_extent_x: float = 0.18
    payload_box_half_extent_y: float = 0.18

    wall_margin: float = 0.15

    simplify_path: bool = True
    line_of_sight_shortcut: bool = True

    diagonal_motion: bool = True


def _parse_pose_text(pose_text: str) -> Tuple[float, float, float, float, float, float]:
    vals = [float(v) for v in pose_text.strip().split()]
    if len(vals) != 6:
        raise ValueError(f"Expected 6 pose values, got {len(vals)} from: {pose_text!r}")
    return tuple(vals)


def _pose_rect_to_world(rect_local: Rect, pose_xyzrpy: Tuple[float, float, float, float, float, float]) -> Rect:
    """Apply only XY translation. Tank is axis-aligned in the uploaded files."""
    tx, ty, _tz, _r, _p, yaw = pose_xyzrpy
    if abs(yaw) > 1e-9:
        raise ValueError("This planner expects axis-aligned tank pose (yaw=0).")
    xmin, xmax, ymin, ymax = rect_local
    return (xmin + tx, xmax + tx, ymin + ty, ymax + ty)


def parse_kth_marinarium_world_sdf(sdf_path: str | Path) -> Dict[str, object]:
    sdf_path = str(Path(sdf_path).expanduser().resolve())
    root = ET.parse(sdf_path).getroot()
    world = root.find("world")
    if world is None:
        raise ValueError("No <world> element found in world SDF")

    tank_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    payload_pose = None
    water_surface_z: Optional[float] = None

    for inc in world.findall("include"):
        name_el = inc.find("name")
        if name_el is not None and (name_el.text or "").strip() == "kth_tank":
            pose_el = inc.find("pose")
            if pose_el is not None and pose_el.text:
                tank_pose = _parse_pose_text(pose_el.text)
            break

    for plugin in world.findall("plugin"):
        if plugin.attrib.get("name") == "gz::sim::systems::Buoyancy":
            graded = plugin.find("graded_buoyancy")
            if graded is not None:
                change = graded.find("density_change")
                if change is not None:
                    above = change.find("above_depth")
                    if above is not None and above.text:
                        water_surface_z = float(above.text.strip())
            break

    for model in world.findall("model"):
        if model.attrib.get("name") == "payload_box_0":
            pose_el = model.find("pose")
            if pose_el is None or not pose_el.text:
                raise ValueError("payload_box_0 found but has no <pose>")
            payload_pose = _parse_pose_text(pose_el.text)
            break

    if payload_pose is None:
        raise ValueError("payload_box_0 model not found in world SDF")

    return {
        "world_sdf_path": sdf_path,
        "tank_pose_xyzrpy": tank_pose,
        "payload_box_pose_xyzrpy": payload_pose,
        "water_surface_z": water_surface_z,
    }


def parse_kth_tank_model_sdf(model_sdf_path: str | Path) -> Dict[str, object]:
    model_sdf_path = str(Path(model_sdf_path).expanduser().resolve())
    root = ET.parse(model_sdf_path).getroot()
    model = root.find("model")
    if model is None:
        raise ValueError("No <model> element found in tank model SDF")
    link = model.find("link")
    if link is None:
        raise ValueError("No <link> element found in tank model SDF")

    water_bounds_local: Optional[Rect] = None
    tank_floor_z_local: Optional[float] = None
    tank_ceiling_z_local: Optional[float] = None

    for visual in link.findall("visual"):
        if visual.attrib.get("name") == "water_volume_visual":
            pose_el = visual.find("pose")
            size_el = visual.find("./geometry/box/size")
            if pose_el is not None and pose_el.text and size_el is not None and size_el.text:
                x, y, z, _r, _p, _yaw = _parse_pose_text(pose_el.text)
                sx, sy, sz = [float(v) for v in size_el.text.strip().split()]
                water_bounds_local = (x - sx / 2.0, x + sx / 2.0, y - sy / 2.0, y + sy / 2.0)
                tank_floor_z_local = z - sz / 2.0
                tank_ceiling_z_local = z + sz / 2.0
                break

    wall_faces = {"xmin": None, "xmax": None, "ymin": None, "ymax": None}
    floor_top_local: Optional[float] = None

    for collision in link.findall("collision"):
        name = collision.attrib.get("name", "")
        pose_el = collision.find("pose")
        size_el = collision.find("./geometry/box/size")
        if pose_el is None or not pose_el.text or size_el is None or not size_el.text:
            continue
        x, y, z, _r, _p, _yaw = _parse_pose_text(pose_el.text)
        sx, sy, sz = [float(v) for v in size_el.text.strip().split()]
        xmin, xmax = x - sx / 2.0, x + sx / 2.0
        ymin, ymax = y - sy / 2.0, y + sy / 2.0
        zmin, zmax = z - sz / 2.0, z + sz / 2.0

        if name == "tank_wall_x_min":
            wall_faces["xmin"] = xmax
        elif name == "tank_wall_x_max":
            wall_faces["xmax"] = xmin
        elif name == "tank_wall_y_min":
            wall_faces["ymin"] = ymax
        elif name == "tank_wall_y_max":
            wall_faces["ymax"] = ymin
        elif name == "tank_floor_box":
            floor_top_local = zmax

    if water_bounds_local is None:
        if all(v is not None for v in wall_faces.values()):
            water_bounds_local = (
                float(wall_faces["xmin"]),
                float(wall_faces["xmax"]),
                float(wall_faces["ymin"]),
                float(wall_faces["ymax"]),
            )

    if tank_floor_z_local is None and floor_top_local is not None:
        tank_floor_z_local = floor_top_local

    return {
        "tank_model_sdf_path": model_sdf_path,
        "tank_inner_bounds_local_xy": water_bounds_local,
        "tank_floor_z_local": tank_floor_z_local,
        "tank_ceiling_z_local": tank_ceiling_z_local,
    }


def inflate_rect(rect: Rect, inflate_xy: float) -> Rect:
    xmin, xmax, ymin, ymax = rect
    return (xmin - inflate_xy, xmax + inflate_xy, ymin - inflate_xy, ymax + inflate_xy)


def rect_contains(rect: Rect, pt: Point2) -> bool:
    xmin, xmax, ymin, ymax = rect
    x, y = pt
    return xmin <= x <= xmax and ymin <= y <= ymax


def make_box_obstacle_from_sdf(world_info: WorldInfo, cfg: PlannerConfig) -> Rect:
    x, y, _z, _r, _p, _yaw = world_info.payload_box_pose_xyzrpy
    rect = (
        x - cfg.payload_box_half_extent_x,
        x + cfg.payload_box_half_extent_x,
        y - cfg.payload_box_half_extent_y,
        y + cfg.payload_box_half_extent_y,
    )
    return inflate_rect(rect, cfg.robot_radius + cfg.obstacle_margin)


def effective_bounds(cfg: PlannerConfig, inner_bounds_xy: Optional[Rect]) -> Rect:
    xmin, xmax, ymin, ymax = inner_bounds_xy if inner_bounds_xy is not None else cfg.fallback_bounds
    m = cfg.wall_margin + cfg.robot_radius
    return (xmin + m, xmax - m, ymin + m, ymax - m)


class OccupancyGrid2D:
    def __init__(self, bounds: Rect, resolution: float, obstacles: Sequence[Rect]):
        self.bounds = bounds
        self.resolution = resolution
        self.obstacles = list(obstacles)
        self.xmin, self.xmax, self.ymin, self.ymax = bounds
        self.nx = int(math.floor((self.xmax - self.xmin) / self.resolution)) + 1
        self.ny = int(math.floor((self.ymax - self.ymin) / self.resolution)) + 1
        if self.nx <= 1 or self.ny <= 1:
            raise ValueError("Invalid occupancy grid dimensions; check bounds/resolution")

    def world_to_grid(self, p: Point2) -> GridIdx:
        x, y = p
        i = int(round((x - self.xmin) / self.resolution))
        j = int(round((y - self.ymin) / self.resolution))
        return (max(0, min(self.nx - 1, i)), max(0, min(self.ny - 1, j)))

    def grid_to_world(self, idx: GridIdx) -> Point2:
        i, j = idx
        x = self.xmin + i * self.resolution
        y = self.ymin + j * self.resolution
        return (x, y)

    def in_bounds_idx(self, idx: GridIdx) -> bool:
        i, j = idx
        return 0 <= i < self.nx and 0 <= j < self.ny

    def in_bounds_world(self, p: Point2) -> bool:
        x, y = p
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    def is_blocked_world(self, p: Point2) -> bool:
        if not self.in_bounds_world(p):
            return True
        return any(rect_contains(rect, p) for rect in self.obstacles)

    def is_blocked_idx(self, idx: GridIdx) -> bool:
        return self.is_blocked_world(self.grid_to_world(idx))

    def neighbors(self, idx: GridIdx, diagonal_motion: bool = True) -> Iterable[Tuple[GridIdx, float]]:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if diagonal_motion:
            steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for di, dj in steps:
            nxt = (idx[0] + di, idx[1] + dj)
            if not self.in_bounds_idx(nxt):
                continue
            if self.is_blocked_idx(nxt):
                continue
            if di != 0 and dj != 0:
                if self.is_blocked_idx((idx[0] + di, idx[1])) or self.is_blocked_idx((idx[0], idx[1] + dj)):
                    continue
            yield nxt, math.hypot(di, dj)


def heuristic(a: GridIdx, b: GridIdx) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def astar_search(grid: OccupancyGrid2D, start_xy: Point2, goal_xy: Point2, diagonal_motion: bool = True) -> List[Point2]:
    if grid.is_blocked_world(start_xy):
        raise ValueError(f"Start lies inside obstacle or outside bounds: {start_xy}")
    if grid.is_blocked_world(goal_xy):
        raise ValueError(f"Goal lies inside obstacle or outside bounds: {goal_xy}")

    start = grid.world_to_grid(start_xy)
    goal = grid.world_to_grid(goal_xy)

    frontier: List[Tuple[float, int, GridIdx]] = []
    counter = 0
    heapq.heappush(frontier, (0.0, counter, start))

    came_from: Dict[GridIdx, Optional[GridIdx]] = {start: None}
    g_cost: Dict[GridIdx, float] = {start: 0.0}

    while frontier:
        _priority, _count, current = heapq.heappop(frontier)
        if current == goal:
            break

        for nxt, step_cost in grid.neighbors(current, diagonal_motion=diagonal_motion):
            new_cost = g_cost[current] + step_cost
            if nxt not in g_cost or new_cost < g_cost[nxt]:
                g_cost[nxt] = new_cost
                counter += 1
                priority = new_cost + heuristic(nxt, goal)
                heapq.heappush(frontier, (priority, counter, nxt))
                came_from[nxt] = current

    if goal not in came_from:
        raise RuntimeError("A* failed: no path found")

    path_idx: List[GridIdx] = []
    cur: Optional[GridIdx] = goal
    while cur is not None:
        path_idx.append(cur)
        cur = came_from[cur]
    path_idx.reverse()

    path_xy = [grid.grid_to_world(idx) for idx in path_idx]
    path_xy[0] = start_xy
    path_xy[-1] = goal_xy
    return path_xy


def _are_collinear(a: Point2, b: Point2, c: Point2, eps: float = 1e-9) -> bool:
    area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    return area2 <= eps


def remove_collinear_points(path_xy: Sequence[Point2]) -> List[Point2]:
    if len(path_xy) <= 2:
        return list(path_xy)
    out = [path_xy[0]]
    for i in range(1, len(path_xy) - 1):
        if not _are_collinear(path_xy[i - 1], path_xy[i], path_xy[i + 1]):
            out.append(path_xy[i])
    out.append(path_xy[-1])
    return out


def line_segment_is_free(grid: OccupancyGrid2D, a: Point2, b: Point2) -> bool:
    dist = math.hypot(b[0] - a[0], b[1] - a[1])
    n = max(2, int(math.ceil(dist / (0.5 * grid.resolution))))
    for k in range(n + 1):
        t = k / n
        p = (a[0] * (1.0 - t) + b[0] * t, a[1] * (1.0 - t) + b[1] * t)
        if grid.is_blocked_world(p):
            return False
    return True


def shortcut_path(path_xy: Sequence[Point2], grid: OccupancyGrid2D) -> List[Point2]:
    if len(path_xy) <= 2:
        return list(path_xy)
    out = [path_xy[0]]
    i = 0
    while i < len(path_xy) - 1:
        j = len(path_xy) - 1
        while j > i + 1:
            if line_segment_is_free(grid, path_xy[i], path_xy[j]):
                break
            j -= 1
        out.append(path_xy[j])
        i = j
    return out


def build_world_info(
    world_sdf_path: str | Path,
    tank_model_sdf_path: Optional[str | Path],
) -> WorldInfo:
    world_data = parse_kth_marinarium_world_sdf(world_sdf_path)
    tank_inner_bounds_xy: Optional[Rect] = None
    tank_floor_z: Optional[float] = None
    tank_ceiling_z: Optional[float] = None
    tank_model_sdf_path_str: Optional[str] = None

    if tank_model_sdf_path is not None:
        tank_data = parse_kth_tank_model_sdf(tank_model_sdf_path)
        tank_model_sdf_path_str = str(Path(tank_model_sdf_path).expanduser().resolve())
        local_bounds = tank_data["tank_inner_bounds_local_xy"]
        if local_bounds is not None:
            tank_inner_bounds_xy = _pose_rect_to_world(local_bounds, world_data["tank_pose_xyzrpy"])
        floor_local = tank_data["tank_floor_z_local"]
        if floor_local is not None:
            tank_floor_z = float(world_data["tank_pose_xyzrpy"][2]) + float(floor_local)
        ceil_local = tank_data["tank_ceiling_z_local"]
        if ceil_local is not None:
            tank_ceiling_z = float(world_data["tank_pose_xyzrpy"][2]) + float(ceil_local)

    return WorldInfo(
        world_sdf_path=str(world_data["world_sdf_path"]),
        tank_model_sdf_path=tank_model_sdf_path_str,
        tank_pose_xyzrpy=world_data["tank_pose_xyzrpy"],
        payload_box_pose_xyzrpy=world_data["payload_box_pose_xyzrpy"],
        water_surface_z=world_data["water_surface_z"],
        tank_inner_bounds_xy=tank_inner_bounds_xy,
        tank_floor_z=tank_floor_z,
        tank_ceiling_z=tank_ceiling_z,
    )


def plan_xy_path_from_sdf(
    world_sdf_path: str | Path,
    start_xy: Point2,
    goal_xy: Point2,
    cfg: PlannerConfig = PlannerConfig(),
    tank_model_sdf_path: Optional[str | Path] = None,
    extra_rect_obstacles: Optional[Sequence[Rect]] = None,
) -> Tuple[List[Point2], WorldInfo, List[Rect], Rect]:
    world_info = build_world_info(world_sdf_path, tank_model_sdf_path)
    bounds = effective_bounds(cfg, world_info.tank_inner_bounds_xy)

    obstacles: List[Rect] = [make_box_obstacle_from_sdf(world_info, cfg)]
    if extra_rect_obstacles:
        for rect in extra_rect_obstacles:
            obstacles.append(inflate_rect(rect, cfg.robot_radius + cfg.obstacle_margin))

    grid = OccupancyGrid2D(bounds=bounds, resolution=cfg.resolution, obstacles=obstacles)
    path = astar_search(grid, start_xy, goal_xy, diagonal_motion=cfg.diagonal_motion)

    if cfg.simplify_path:
        path = remove_collinear_points(path)
    if cfg.line_of_sight_shortcut:
        path = shortcut_path(path, grid)
        if cfg.simplify_path:
            path = remove_collinear_points(path)

    return path, world_info, obstacles, bounds


def path_xy_to_xyz(path_xy: Sequence[Point2], z_ctrl: float) -> List[Tuple[float, float, float]]:
    return [(x, y, z_ctrl) for (x, y) in path_xy]


def path_length(path_xy: Sequence[Point2]) -> float:
    if len(path_xy) <= 1:
        return 0.0
    return sum(
        math.hypot(path_xy[i + 1][0] - path_xy[i][0], path_xy[i + 1][1] - path_xy[i][1])
        for i in range(len(path_xy) - 1)
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Standalone A* planner for KTH marinarium")
    parser.add_argument("--world-sdf", required=True, help="Path to kth_marinarium.sdf")
    parser.add_argument("--tank-sdf", default=None, help="Path to kth_tank model.sdf")
    parser.add_argument("--start-x", type=float, required=True)
    parser.add_argument("--start-y", type=float, required=True)
    parser.add_argument("--goal-x", type=float, required=True)
    parser.add_argument("--goal-y", type=float, required=True)
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--fallback-bounds", type=float, nargs=4,
                        default=[-2.6, 2.6, -2.6, 2.6],
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    parser.add_argument("--box-half-x", type=float, default=0.18)
    parser.add_argument("--box-half-y", type=float, default=0.18)
    parser.add_argument("--robot-radius", type=float, default=0.20)
    parser.add_argument("--obstacle-margin", type=float, default=0.10)
    parser.add_argument("--wall-margin", type=float, default=0.15)
    args = parser.parse_args()

    cfg = PlannerConfig(
        fallback_bounds=tuple(args.fallback_bounds),
        resolution=args.resolution,
        robot_radius=args.robot_radius,
        obstacle_margin=args.obstacle_margin,
        payload_box_half_extent_x=args.box_half_x,
        payload_box_half_extent_y=args.box_half_y,
        wall_margin=args.wall_margin,
    )

    path, world_info, obstacles, bounds = plan_xy_path_from_sdf(
        world_sdf_path=args.world_sdf,
        tank_model_sdf_path=args.tank_sdf,
        start_xy=(args.start_x, args.start_y),
        goal_xy=(args.goal_x, args.goal_y),
        cfg=cfg,
    )

    print(json.dumps({
        "world_sdf_path": world_info.world_sdf_path,
        "tank_model_sdf_path": world_info.tank_model_sdf_path,
        "tank_pose_xyzrpy": world_info.tank_pose_xyzrpy,
        "tank_inner_bounds_xy": world_info.tank_inner_bounds_xy,
        "tank_floor_z": world_info.tank_floor_z,
        "tank_ceiling_z": world_info.tank_ceiling_z,
        "payload_box_pose_xyzrpy": world_info.payload_box_pose_xyzrpy,
        "water_surface_z": world_info.water_surface_z,
        "effective_bounds": bounds,
        "obstacles": obstacles,
        "path_xy": path,
        "path_length_m": path_length(path),
    }, indent=2))
