import logging
import math

from shapely.affinity import rotate as _shapely_rotate
from shapely.geometry import MultiPolygon, Point, Polygon

from ..angle_finders.altitude_optimizer import find_optimal_angle
from . import utils
from .visibility_graph import VisibilityGraph, _transition_waypoints

logger = logging.getLogger(__name__)

_NAN = (float("nan"), float("nan"))


def _polygon_for_optimal_angle(geom: Polygon | MultiPolygon) -> Polygon:
    if geom.is_empty:
        return Polygon()
    if isinstance(geom, MultiPolygon):
        parts = [g for g in geom.geoms if not g.is_empty and g.area > 0]
        return max(parts, key=lambda p: p.area) if parts else Polygon()
    return geom


def _solve_order_ortools(
    segs: list,
    start_pt: tuple[float, float],
    time_limit_sec: int = 10,
) -> list[tuple[int, bool]] | None:
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except ImportError:
        return None

    n = len(segs)
    if n == 0:
        return []
    if n == 1:
        coords = list(segs[0].coords)
        d0 = math.hypot(coords[0][0] - start_pt[0], coords[0][1] - start_pt[1])
        d1 = math.hypot(coords[-1][0] - start_pt[0], coords[-1][1] - start_pt[1])
        return [(0, d1 < d0)]

    ep: list[tuple[tuple, tuple]] = [
        (list(segs[i].coords)[0], list(segs[i].coords)[-1]) for i in range(n)
    ]

    def _d(a, b) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    depot = 2 * n
    n_nodes = 2 * n + 1
    _SCALE = 1000
    big = int(1e9)

    def entry_pt(node: int) -> tuple:
        if node == depot:
            return start_pt
        i, fwd = node // 2, node % 2 == 0
        return ep[i][0] if fwd else ep[i][1]

    def exit_pt(node: int) -> tuple:
        if node == depot:
            return start_pt
        i, fwd = node // 2, node % 2 == 0
        return ep[i][1] if fwd else ep[i][0]

    dist_matrix: list[list[int]] = [[0] * n_nodes for _ in range(n_nodes)]
    for u in range(n_nodes):
        for v in range(n_nodes):
            if u == v:
                dist_matrix[u][v] = 0
            elif v == depot:
                dist_matrix[u][v] = 0
            elif u != depot and u // 2 == v // 2:
                dist_matrix[u][v] = big
            else:
                dist_matrix[u][v] = int(_d(exit_pt(u), entry_pt(v)) * _SCALE)

    manager = pywrapcp.RoutingIndexManager(n_nodes, 1, depot)
    routing = pywrapcp.RoutingModel(manager)

    def _callback(from_idx: int, to_idx: int) -> int:
        return dist_matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]

    cb = routing.RegisterTransitCallback(_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(cb)

    for i in range(n):
        routing.AddDisjunction(
            [manager.NodeToIndex(2 * i), manager.NodeToIndex(2 * i + 1)],
            big,
            1,
        )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.seconds = time_limit_sec

    solution = routing.SolveWithParameters(params)
    if not solution:
        return None

    order_flags: list[tuple[int, bool]] = []
    idx = routing.Start(0)
    while not routing.IsEnd(idx):
        node = manager.IndexToNode(idx)
        if node != depot:
            order_flags.append((node // 2, node % 2 == 1))
        idx = solution.Value(routing.NextVar(idx))

    if len(order_flags) != n:
        return None

    return order_flags


def _order_swaths(
    segs: list,
    start_pt: tuple[float, float],
    exact: bool = False,
) -> list[tuple[int, bool]]:
    if exact:
        result = _solve_order_ortools(segs, start_pt)
        if result is not None:
            return result

    if len(segs) <= 1:
        if not segs:
            return []
        coords = list(segs[0].coords)
        d0 = math.hypot(coords[0][0] - start_pt[0], coords[0][1] - start_pt[1])
        d1 = math.hypot(coords[-1][0] - start_pt[0], coords[-1][1] - start_pt[1])
        return [(0, d1 < d0)]

    def segment_distance_and_flip(
        seg_idx: int, pt: tuple[float, float]
    ) -> tuple[float, bool]:
        seg = segs[seg_idx]
        coords = list(seg.coords)
        d_start = math.hypot(coords[0][0] - pt[0], coords[0][1] - pt[1])
        d_end = math.hypot(coords[-1][0] - pt[0], coords[-1][1] - pt[1])
        return (d_start, False) if d_start <= d_end else (d_end, True)

    nearest_idx = 0
    nearest_dist, nearest_flip = segment_distance_and_flip(0, start_pt)
    for i in range(1, len(segs)):
        dist_i, flip_i = segment_distance_and_flip(i, start_pt)
        if dist_i < nearest_dist:
            nearest_dist = dist_i
            nearest_idx = i
            nearest_flip = flip_i

    base_y = start_pt[1]

    first_side_is_top = segs[nearest_idx].centroid.y >= base_y

    top_side = []  # centroid.y >= base_y
    bottom_side = []  # centroid.y < base_y

    for i, seg in enumerate(segs):
        if seg.centroid.y >= base_y:
            top_side.append(i)
        else:
            bottom_side.append(i)

    def sort_by_y_distance(indices):
        return sorted(indices, key=lambda i: abs(segs[i].centroid.y - base_y))

    top_sorted = sort_by_y_distance(top_side)
    bottom_sorted = sort_by_y_distance(bottom_side)

    order_flags: list[tuple[int, bool]] = []

    def add_side_ordered(side_indices, is_first_side):
        if not side_indices:
            return

        current_flip = nearest_flip if is_first_side else None

        for idx in side_indices:
            coords = list(segs[idx].coords)

            if current_flip is None:
                last_seg_idx, last_flip = order_flags[-1]
                last_coords = list(segs[last_seg_idx].coords)
                last_exit = last_coords[0] if last_flip else last_coords[-1]

                d_start = math.hypot(
                    coords[0][0] - last_exit[0], coords[0][1] - last_exit[1]
                )
                d_end = math.hypot(
                    coords[-1][0] - last_exit[0], coords[-1][1] - last_exit[1]
                )
                current_flip = d_end < d_start

            order_flags.append((idx, current_flip))
            current_flip = not current_flip

    if first_side_is_top:
        add_side_ordered(top_sorted, is_first_side=True)
        add_side_ordered(bottom_sorted, is_first_side=False)
    else:
        add_side_ordered(bottom_sorted, is_first_side=True)
        add_side_ordered(top_sorted, is_first_side=False)

    return order_flags


def _remove_hole_intersections(
    corridor: Polygon | MultiPolygon, zone: Polygon | MultiPolygon
) -> Polygon | MultiPolygon:
    if corridor.is_empty or zone.is_empty:
        return corridor

    if zone.geom_type == "Polygon":
        holes = zone.interiors
    elif zone.geom_type == "MultiPolygon":
        holes = []
        for part in zone.geoms:
            if hasattr(part, "interiors"):
                holes.extend(part.interiors)
    else:
        holes = []

    if not holes:
        return corridor

    hole_polygons = [Polygon(hole) for hole in holes]

    def clip_single(poly: Polygon) -> Polygon:
        if not poly.interiors:
            return poly
        new_interiors = []
        for ring in poly.interiors:
            interior_poly = Polygon(ring)
            best_inter = None
            max_area = 0.0
            for hole_poly in hole_polygons:
                inter = interior_poly.intersection(hole_poly)
                if not inter.is_empty and inter.area > max_area:
                    max_area = inter.area
                    best_inter = inter
            if best_inter is not None and not best_inter.is_empty:
                if best_inter.geom_type == "Polygon":
                    new_ring = best_inter.exterior
                elif best_inter.geom_type == "MultiPolygon":
                    largest = max(best_inter.geoms, key=lambda g: g.area)
                    new_ring = largest.exterior
                else:
                    continue
                new_interiors.append(new_ring)
        return Polygon(poly.exterior, new_interiors)

    if isinstance(corridor, MultiPolygon):
        new_geoms = []
        for geom in corridor.geoms:
            if geom.geom_type == "Polygon":
                new_geoms.append(clip_single(geom))
            else:
                new_geoms.append(geom)
        new_geoms = [g for g in new_geoms if not g.is_empty and g.area > 1e-12]
        if len(new_geoms) == 0:
            return Polygon()
        elif len(new_geoms) == 1:
            return new_geoms[0]
        else:
            return MultiPolygon(new_geoms)
    else:
        return clip_single(corridor)


def _build_spray_zone(
    zone: Polygon | MultiPolygon, swath_width: float
) -> Polygon | MultiPolygon:
    if zone.is_empty or swath_width <= 0:
        return Polygon()

    if zone.geom_type == "MultiPolygon":
        parts = [
            _build_spray_zone(p, swath_width) for p in zone.geoms if not p.is_empty
        ]
        parts = [p for p in parts if not p.is_empty]
        if not parts:
            return Polygon()
        if len(parts) == 1:
            return parts[0]
        from shapely.ops import unary_union

        return unary_union(parts)

    def _build_inset_polygon(poly: Polygon) -> Polygon:
        inset_y = swath_width / 2.0
        inset = poly.buffer(-inset_y, cap_style=2, join_style=2)
        if inset.is_empty or not inset.is_valid:
            return Polygon()
        if inset.geom_type == "MultiPolygon":
            parts = [g for g in inset.geoms if not g.is_empty and g.area > 1e-9]
            if not parts:
                return Polygon()
            return max(parts, key=lambda p: p.area)
        return inset

    inset_poly = _build_inset_polygon(zone)

    if not inset_poly.is_valid or inset_poly.is_empty:
        min_x, min_y, max_x, max_y = zone.bounds
        inset_y = swath_width / 2.0
        clip_box = Polygon(
            [
                (min_x - 1.0, min_y + inset_y),
                (max_x + 1.0, min_y + inset_y),
                (max_x + 1.0, max_y - inset_y),
                (min_x - 1.0, max_y - inset_y),
            ]
        )
        corridor = zone.intersection(clip_box)
    else:
        corridor = zone.intersection(inset_poly)

    if corridor.is_empty:
        return Polygon()
    if corridor.geom_type == "GeometryCollection":
        polygons = [
            g
            for g in corridor.geoms
            if g.geom_type == "Polygon" and not g.is_empty and g.area > 0
        ]
        if not polygons:
            return Polygon()
        if len(polygons) == 1:
            corridor = polygons[0]
        else:
            from shapely.ops import unary_union

            union = unary_union(polygons)
            if union.is_empty:
                return Polygon()
            corridor = union

    return _remove_hole_intersections(corridor, zone)


def _normalize_spray_zone(
    corridor: Polygon | MultiPolygon,
    swath_width: float,
) -> Polygon | MultiPolygon:
    if corridor.is_empty or not isinstance(corridor, MultiPolygon):
        return corridor

    parts = [p for p in corridor.geoms if not p.is_empty and p.area > 1e-12]
    if len(parts) <= 1:
        return corridor

    keep: list[Polygon] = []
    area_tol = max((swath_width**2) * 0.5, 1e-6)
    cover_gap_tol = max(swath_width * 0.5, 1e-6)

    for i, part in enumerate(parts):
        nearest_other = min(
            (part.distance(other) for j, other in enumerate(parts) if j != i),
            default=math.inf,
        )
        drop_as_coverable_island = (
            part.area <= area_tol and nearest_other <= cover_gap_tol
        )
        if not drop_as_coverable_island:
            keep.append(part)

    if not keep:
        keep = [max(parts, key=lambda p: p.area)]
    if len(keep) == 1:
        return keep[0]
    return MultiPolygon(keep)


def _last_real_xy(path: list[tuple[float, float]]) -> tuple[float, float] | None:
    for p in reversed(path):
        if not math.isnan(p[0]):
            return (p[0], p[1])
    return None


def _stitch_swath_segments(
    path_rot: list[tuple[float, float]],
    all_segs: list,
    order_flags: list[tuple[int, bool]],
    vis: VisibilityGraph,
) -> None:
    logger.debug(
        "_stitch_swath_segments called with %d segments, order_flags %s",
        len(all_segs),
        order_flags,
    )

    for seg_idx, flipped in order_flags:
        coords = list(all_segs[seg_idx].coords)
        if flipped:
            coords = list(reversed(coords))
        near = coords[0]
        if not path_rot:
            start = coords[0]
            end = coords[-1]
            if vis._visible(start, end):
                path_rot.extend(coords)
            else:
                detour = vis.shortest_path(start, end)
                if len(detour) == 2:
                    detour = _transition_waypoints(start, end, vis._original_polygon)
                path_rot.extend(detour)
        else:
            last = path_rot[-1]
            if math.hypot(last[0] - near[0], last[1] - near[1]) > 1e-9:
                transition = vis.shortest_path(last, near)
                if len(transition) == 2 and not vis._visible(last, near):
                    transition = _transition_waypoints(
                        last, near, vis._original_polygon
                    )
                path_rot.extend(transition[1:])
            start = coords[0]
            end = coords[-1]
            if vis._visible(start, end):
                path_rot.extend(coords[1:])
            else:
                detour = vis.shortest_path(start, end)
                if len(detour) == 2:
                    detour = _transition_waypoints(start, end, vis._original_polygon)
                path_rot.extend(detour[1:])

    path_rot[:] = utils.ensure_path_avoids_holes(path_rot, vis._original_polygon, vis)


def _cell_coverage_path(
    cell_poly: Polygon,
    swath_width: float,
    start_position: tuple[float, float],
    *,
    exact: bool = False,
) -> list[tuple[float, float]]:
    if cell_poly.is_empty or cell_poly.area <= 1e-12:
        return []

    cell_angle = find_optimal_angle(cell_poly)
    cell_angle_deg = math.degrees(cell_angle)
    centroid = cell_poly.centroid
    rotated_cell = _shapely_rotate(
        cell_poly, -cell_angle_deg, origin=centroid, use_radians=False
    )
    rot_start_pt = _shapely_rotate(
        Point(start_position), -cell_angle_deg, origin=centroid, use_radians=False
    )
    rot_start = (rot_start_pt.x, rot_start_pt.y)

    all_segs = utils.horizontal_swath_segments(rotated_cell, swath_width)
    if not all_segs:
        return []

    order_flags = _order_swaths(all_segs, rot_start, exact=exact)
    vis = VisibilityGraph(rotated_cell)
    path_rot: list[tuple[float, float]] = []
    _stitch_swath_segments(path_rot, all_segs, order_flags, vis)
    if not path_rot:
        return []
    return utils.rotate_path_points(path_rot, centroid, cell_angle_deg)


def _rotated_to_world(
    path_rot: list[tuple[float, float]],
    centroid: Point,
    angle_deg: float,
) -> list[tuple[float, float]]:
    return utils.rotate_path_points(path_rot, centroid, angle_deg)


def _bcd_snake_path(
    polygon: Polygon | MultiPolygon,
    swath_width: float,
    angle_rad: float,
    start_position: tuple[float, float],
    exact: bool = False,
) -> list[tuple[float, float]]:
    from .bcd import (
        _traversal_order,
        _two_opt_order,
        bcd_slice_decompose,
    )

    centroid = polygon.centroid

    candidate_angles = [
        math.degrees(angle_rad),
        math.degrees(angle_rad) + 90.0,
    ]

    best_path: list[tuple[float, float]] = []
    best_cells_count = 0
    best_angle_deg = candidate_angles[0]

    for angle_deg in candidate_angles:
        angle_deg = angle_deg % 180.0

        rotated_poly = _shapely_rotate(
            polygon, -angle_deg, origin=centroid, use_radians=False
        )
        rot_start_pt = _shapely_rotate(
            Point(start_position), -angle_deg, origin=centroid, use_radians=False
        )
        rot_start = (rot_start_pt.x, rot_start_pt.y)

        corridor = _normalize_spray_zone(
            _build_spray_zone(rotated_poly, swath_width),
            swath_width,
        )
        if corridor.is_empty:
            continue

        cells = bcd_slice_decompose(
            corridor,
            swath=swath_width,
        )
        if not cells:
            continue

        cells_count = len(cells)

        visit_order = _two_opt_order(cells, _traversal_order(cells, rot_start))
        vis = VisibilityGraph(corridor)
        path_rot: list[tuple[float, float]] = []

        for ci in visit_order:
            cell_poly = cells[ci].poly
            cursor = (
                rot_start if not path_rot else (_last_real_xy(path_rot) or rot_start)
            )
            cell_path = _cell_coverage_path(
                cell_poly,
                swath_width,
                cursor,
                exact=exact,
            )
            if not cell_path:
                continue
            if not path_rot:
                path_rot.extend(cell_path)
                continue
            last = _last_real_xy(path_rot)
            first = cell_path[0]
            if (
                last is not None
                and math.hypot(last[0] - first[0], last[1] - first[1]) > 1e-9
            ):
                transition = vis.shortest_path(last, first)
                path_rot.extend(transition[1:])
            path_rot.extend(cell_path[1:])

        if not path_rot:
            continue

        path_rot[:] = utils.ensure_path_avoids_holes(
            path_rot, vis._original_polygon, vis
        )
        world_path = _rotated_to_world(path_rot, centroid, angle_deg)

        if cells_count > best_cells_count or (
            cells_count == best_cells_count
            and best_path
            and utils.path_length(world_path) < utils.path_length(best_path)
        ):
            best_path = world_path
            best_cells_count = cells_count

    return best_path


def build_zone_snake(
    zone_polygon: Polygon | MultiPolygon,
    swath_width: float,
    angle_rad: float = None,
    start_position=None,
    exact: bool = False,
    field_polygon: Polygon | MultiPolygon | None = None,
) -> tuple[list[tuple[float, float]], float, int]:
    if zone_polygon.is_empty:
        return [], 0.0, 0

    if angle_rad is None:
        angle_rad = find_optimal_angle(_polygon_for_optimal_angle(zone_polygon))

    if start_position is None:
        start_position = (zone_polygon.centroid.x, zone_polygon.centroid.y)

    path = _bcd_snake_path(
        zone_polygon,
        swath_width,
        angle_rad,
        start_position,
        exact=exact,
    )

    polygon_for_holes = field_polygon if field_polygon is not None else zone_polygon
    path = utils.ensure_path_avoids_holes(path, polygon_for_holes)

    num_turns = utils.count_turns(path)
    return path, angle_rad, num_turns
