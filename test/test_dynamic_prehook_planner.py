"""Pure-function safety tests for the dynamic pre-hook A* planner."""

from bluerov2_control.planner_astar import (
    OccupancyGrid2D,
    astar_search,
    line_segment_is_free,
    path_is_free,
    path_length,
    plan_xy_path,
    project_point_to_path,
    remaining_path_from_projection,
    smooth_path_collision_free,
)

import pytest


def _plan_around_central_obstacle():
    start = (0.5, 2.0)
    goal = (3.5, 2.0)
    obstacle = (1.5, 2.5, 1.0, 3.0)
    path, grid = plan_xy_path(
        start_xy=start,
        goal_xy=goal,
        bounds=(0.0, 4.0, 0.0, 4.0),
        obstacles=[obstacle],
        resolution=0.25,
        diagonal_motion=True,
        smoothing_iterations=1,
    )
    return start, goal, path, grid


def test_astar_detours_around_a_blocking_obstacle():
    """A blocking rectangle must force a longer, non-direct route."""
    start, goal, path, grid = _plan_around_central_obstacle()

    assert path[0] == start
    assert path[-1] == goal
    assert not line_segment_is_free(grid, start, goal)
    assert path_length(path) > path_length([start, goal])
    assert path_is_free(path, grid)


def test_diagonal_motion_cannot_cut_a_blocked_corner():
    """A diagonal is forbidden when either adjacent cardinal cell blocks."""
    grid = OccupancyGrid2D(
        bounds=(0.0, 2.0, 0.0, 2.0),
        resolution=1.0,
        obstacles=[
            (0.75, 1.25, -0.25, 0.25),
            (-0.25, 0.25, 0.75, 1.25),
        ],
    )

    neighbor_indices = {
        index for index, _cost in grid.neighbors((0, 0), True)
    }
    assert (1, 1) not in neighbor_indices
    with pytest.raises(RuntimeError, match='no path found'):
        astar_search(
            grid,
            start_xy=(0.0, 0.0),
            goal_xy=(1.0, 1.0),
            diagonal_motion=True,
        )


def test_neighbor_edge_cannot_cross_a_narrow_between_node_obstacle():
    """Free endpoints do not make an intersecting grid edge collision-free."""
    grid = OccupancyGrid2D(
        bounds=(0.0, 2.0, 0.0, 2.0),
        resolution=1.0,
        obstacles=[(0.49, 0.51, 0.0, 0.1)],
    )

    assert not grid.is_blocked_idx((0, 0))
    assert not grid.is_blocked_idx((1, 0))
    assert (1, 0) not in {
        index for index, _cost in grid.neighbors((0, 0), False)
    }


def test_astar_routes_around_an_obstacle_between_grid_nodes():
    """A* must detour when a thin rectangle intersects only a grid edge."""
    grid = OccupancyGrid2D(
        bounds=(0.0, 2.0, 0.0, 2.0),
        resolution=1.0,
        obstacles=[(0.49, 0.51, 0.0, 0.1)],
    )

    path = astar_search(
        grid,
        start_xy=(0.0, 0.0),
        goal_xy=(1.0, 0.0),
        diagonal_motion=True,
    )

    assert path[0] == (0.0, 0.0)
    assert path[-1] == (1.0, 0.0)
    assert path_length(path) > 1.0
    assert path_is_free(path, grid)


def test_arbitrary_start_uses_a_safe_alternative_grid_connector():
    """A non-grid start must not be substituted onto an unsafe first edge."""
    grid = OccupancyGrid2D(
        bounds=(0.0, 4.0, 0.0, 3.0),
        resolution=1.0,
        obstacles=[
            (0.24, 0.26, 0.0, 0.75),
            (1.4, 1.6, 0.0, 1.5),
        ],
    )
    start = (0.49, 0.0)
    goal = (3.0, 0.0)

    path = astar_search(
        grid,
        start_xy=start,
        goal_xy=goal,
        diagonal_motion=True,
    )

    assert path[0] == start
    assert path[-1] == goal
    assert path_is_free(path, grid)
    assert path[1][0] >= 1.0


def test_final_postprocessed_path_passes_complete_collision_check():
    """Every final waypoint and segment must pass path_is_free."""
    start, goal, path, grid = _plan_around_central_obstacle()

    assert not path_is_free([start, goal], grid)
    assert path_is_free(path, grid)
    for index in range(len(path) - 1):
        assert line_segment_is_free(grid, path[index], path[index + 1])


def test_continuous_check_detects_a_shallow_obstacle_corner_clip():
    """A short corner crossing between grid samples must be a collision."""
    grid = OccupancyGrid2D(
        bounds=(0.0, 10.0, 0.0, 10.0),
        resolution=1.0,
        obstacles=[(4.0, 5.0, 4.0, 5.0)],
    )

    # y = -x + 8.01 enters only 0.01 m into the obstacle at (4, 4).
    assert not line_segment_is_free(grid, (0.0, 8.01), (8.01, 0.0))


def test_continuous_check_treats_obstacle_tangency_as_collision():
    """Touching one point of a closed obstacle is not collision-free."""
    grid = OccupancyGrid2D(
        bounds=(0.0, 10.0, 0.0, 10.0),
        resolution=1.0,
        obstacles=[(4.0, 5.0, 4.0, 5.0)],
    )

    # y = -x + 8 touches the obstacle only at its lower-left corner.
    assert not line_segment_is_free(grid, (0.0, 8.0), (8.0, 0.0))


def test_continuous_check_accepts_a_nearby_nonintersecting_segment():
    """A segment with a positive gap from the obstacle remains free."""
    grid = OccupancyGrid2D(
        bounds=(0.0, 10.0, 0.0, 10.0),
        resolution=1.0,
        obstacles=[(4.0, 5.0, 4.0, 5.0)],
    )

    assert line_segment_is_free(grid, (0.0, 7.99), (7.99, 0.0))


def test_continuous_check_enforces_pool_bounds_for_the_whole_segment():
    """The closed operating bounds are checked analytically, not sampled."""
    grid = OccupancyGrid2D(
        bounds=(0.0, 10.0, 0.0, 10.0),
        resolution=1.0,
        obstacles=[],
    )

    assert line_segment_is_free(grid, (0.0, 0.0), (10.0, 10.0))
    assert not line_segment_is_free(
        grid,
        (0.0, 0.0),
        (10.0 + 1e-9, 10.0),
    )


def test_safe_corner_smoothing_remains_collision_free():
    """Accepted Bezier corner samples must remain in free space."""
    grid = OccupancyGrid2D(
        bounds=(0.0, 3.0, 0.0, 3.0),
        resolution=0.05,
        obstacles=[],
    )
    polyline = [(0.25, 0.25), (0.25, 2.0), (2.0, 2.0)]

    smoothed = smooth_path_collision_free(
        polyline,
        grid,
        iterations=1,
        corner_fraction=0.20,
        samples_per_corner=4,
    )

    assert smoothed != polyline
    assert smoothed[0] == polyline[0]
    assert smoothed[-1] == polyline[-1]
    assert path_is_free(smoothed, grid)


def test_unsafe_corner_smoothing_falls_back_to_safe_polyline():
    """Smoothing must be rejected when its rounded corner hits an obstacle."""
    grid = OccupancyGrid2D(
        bounds=(0.0, 3.0, 0.0, 3.0),
        resolution=0.02,
        obstacles=[(1.0, 2.0, 1.0, 2.0)],
    )
    safe_polyline = [(0.99, 1.5), (0.99, 0.99), (1.5, 0.99)]
    assert path_is_free(safe_polyline, grid)

    smoothed = smooth_path_collision_free(
        safe_polyline,
        grid,
        iterations=1,
        corner_fraction=0.49,
        samples_per_corner=8,
    )

    assert smoothed == safe_polyline
    assert path_is_free(smoothed, grid)


def test_projection_progress_never_regresses_and_remainder_is_consistent():
    """Minimum progress clamps projection and preserves remaining length."""
    path = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)]
    forward = project_point_to_path((2.1, 0.75), path)

    assert forward.point_xy == pytest.approx((2.0, 0.75))
    assert forward.cross_track_m == pytest.approx(0.1)
    assert forward.progress_m == pytest.approx(2.75)
    assert forward.remaining_length_m == pytest.approx(1.25)

    attempted_regression = project_point_to_path(
        (0.5, 0.05),
        path,
        minimum_progress_m=forward.progress_m,
    )
    remainder = remaining_path_from_projection(path, attempted_regression)

    assert attempted_regression.progress_m >= forward.progress_m
    assert attempted_regression.progress_m == pytest.approx(2.75)
    assert attempted_regression.remaining_length_m == pytest.approx(1.25)
    assert remainder[0] == pytest.approx(attempted_regression.point_xy)
    assert path_length(remainder) == pytest.approx(
        attempted_regression.remaining_length_m
    )
