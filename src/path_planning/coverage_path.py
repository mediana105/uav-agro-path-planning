import math

from shapely.affinity import rotate as _shapely_rotate
from shapely.geometry import LineString, Point, Polygon

from ..angle_finders.altitude_optimizer import find_optimal_angle
from ..path_planning.boustrophedon import _extract_linestring
from .visibility_graph import VisibilityGraph


def _greedy_order_subpolygons(
    parts: list[Polygon],
    start_xy: tuple[float, float],
) -> list[Polygon]:
    remaining = [p for p in parts if not p.is_empty and p.area > 0]
    ordered: list[Polygon] = []
    cx, cy = start_xy
    while remaining:
        best_k = 0
        best_d = math.inf
        for k, poly in enumerate(remaining):
            g = poly.centroid
            d = math.hypot(g.x - cx, g.y - cy)
            if d < best_d:
                best_d = d
                best_k = k
        nxt = remaining.pop(best_k)
        ordered.append(nxt)
        c = nxt.centroid
        cx, cy = c.x, c.y
    return ordered


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

    remaining: set[int] = set(range(len(segs)))
    order_flags: list[tuple[int, bool]] = []
    cx, cy = start_pt
    while remaining:
        best_i, best_d, best_flip = -1, math.inf, False
        for i in remaining:
            coords = list(segs[i].coords)
            d0 = math.hypot(coords[0][0] - cx, coords[0][1] - cy)
            d1 = math.hypot(coords[-1][0] - cx, coords[-1][1] - cy)
            d, flip = (d0, False) if d0 <= d1 else (d1, True)
            if d < best_d:
                best_d, best_i, best_flip = d, i, flip
        remaining.remove(best_i)
        order_flags.append((best_i, best_flip))
        coords = list(segs[best_i].coords)
        cx, cy = coords[0] if best_flip else coords[-1]
    return _two_opt_swaths(segs, order_flags)


def _two_opt_swaths(
    segs: list,
    order_flags: list[tuple[int, bool]],
) -> list[tuple[int, bool]]:
    def entry(idx: int, flipped: bool) -> tuple[float, float]:
        coords = list(segs[idx].coords)
        return coords[-1] if flipped else coords[0]

    def exit_(idx: int, flipped: bool) -> tuple[float, float]:
        coords = list(segs[idx].coords)
        return coords[0] if flipped else coords[-1]

    def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    best = list(order_flags)
    n = len(best)
    improved = True
    while improved:
        improved = False
        for i in range(1, n):
            for j in range(i + 1, n):
                exit_prev = exit_(*best[i - 1])
                entry_i = entry(*best[i])
                exit_j = exit_(*best[j])

                before = dist(exit_prev, entry_i)
                if j + 1 < n:
                    before += dist(exit_j, entry(*best[j + 1]))

                after = dist(exit_prev, exit_j)
                if j + 1 < n:
                    after += dist(entry_i, entry(*best[j + 1]))

                if after < before - 1e-9:
                    segment = best[i : j + 1]
                    segment.reverse()
                    best[i : j + 1] = [(idx, not flip) for idx, flip in segment]
                    improved = True
                    break
            if improved:
                break
    return best


def _bcd_snake_path(
    polygon: Polygon,
    swath_width: float,
    angle_rad: float,
    start_position: tuple[float, float],
    exact: bool = False,
) -> list[tuple[float, float]]:
    from ..path_planning.bcd import bcd_slice_decompose

    centroid = polygon.centroid
    angle_deg = math.degrees(angle_rad)

    rotated_poly = _shapely_rotate(
        polygon, -angle_deg, origin=centroid, use_radians=False
    )
    rotated_poly = rotated_poly.buffer(-swath_width / 2.0)
    if rotated_poly.is_empty:
        return []
    rot_start_pt = _shapely_rotate(
        Point(start_position), -angle_deg, origin=centroid, use_radians=False
    )
    rot_start = (rot_start_pt.x, rot_start_pt.y)

    cells = bcd_slice_decompose(rotated_poly, swath_width)
    if not cells:
        return _greedy_safe_path(
            polygon, swath_width, angle_rad, start_position, exact=exact
        )

    minx, miny, maxx, maxy = rotated_poly.bounds
    margin = (maxx - minx) + 1.0
    all_segs: list = []

    y = miny + swath_width / 2.0
    while y < maxy:
        sweep_line = LineString([(minx - margin, y), (maxx + margin, y)])
        for cell in cells:
            cb = cell.poly.bounds
            if not (cb[1] <= y <= cb[3]):
                continue
            try:
                inter = cell.poly.intersection(sweep_line)
            except Exception:
                continue
            for seg in _extract_linestring(inter):
                if seg.length > 1e-9:
                    all_segs.append(seg)
        y += swath_width

    if not all_segs:
        return []

    order_flags = _order_swaths(all_segs, rot_start, exact=exact)

    vis = VisibilityGraph(rotated_poly)
    path_rot: list[tuple[float, float]] = []

    for seg_idx, flipped in order_flags:
        coords = list(all_segs[seg_idx].coords)
        if flipped:
            coords = list(reversed(coords))
        if not path_rot:
            path_rot.extend(coords)
        else:
            last = path_rot[-1]
            near = coords[0]
            if math.hypot(last[0] - near[0], last[1] - near[1]) > 1e-9:
                transition = vis.shortest_path(last, near)
                path_rot.extend(transition[1:])
            path_rot.extend(coords[1:])

    if not path_rot:
        return []

    cx_c, cy_c = centroid.x, centroid.y
    cos_a = math.cos(math.radians(angle_deg))
    sin_a = math.sin(math.radians(angle_deg))
    result: list[tuple[float, float]] = []
    for px, py in path_rot:
        if math.isnan(px):
            result.append((px, py))
            continue
        dx, dy = px - cx_c, py - cy_c
        result.append((cx_c + cos_a * dx - sin_a * dy, cy_c + sin_a * dx + cos_a * dy))
    return result


def _greedy_safe_path(
    polygon: Polygon,
    swath_width: float,
    angle_rad: float,
    start_position: tuple[float, float],
    exact: bool = False,
) -> list[tuple[float, float]]:
    centroid = polygon.centroid
    angle_deg = math.degrees(angle_rad)

    rotated_poly = _shapely_rotate(
        polygon, -angle_deg, origin=centroid, use_radians=False
    )
    rotated_poly = rotated_poly.buffer(-swath_width / 2.0)
    if rotated_poly.is_empty:
        return []
    rot_start_pt = _shapely_rotate(
        Point(start_position), -angle_deg, origin=centroid, use_radians=False
    )
    rot_start = (rot_start_pt.x, rot_start_pt.y)

    min_x, min_y, max_x, max_y = rotated_poly.bounds
    if max_y - min_y < 1e-9:
        return []

    margin = (max_x - min_x) + 1.0
    all_segs: list = []
    y = min_y + swath_width / 2.0
    while y < max_y:
        sweep_line = LineString([(min_x - margin, y), (max_x + margin, y)])
        try:
            inter = rotated_poly.intersection(sweep_line)
        except Exception:
            y += swath_width
            continue
        for seg in _extract_linestring(inter):
            if seg.length > 1e-9:
                all_segs.append(seg)
        y += swath_width

    if not all_segs:
        return []

    order_flags = _order_swaths(all_segs, rot_start, exact=exact)

    vis = VisibilityGraph(rotated_poly)
    path_rot: list[tuple[float, float]] = []

    for seg_idx, flipped in order_flags:
        coords = list(all_segs[seg_idx].coords)
        if flipped:
            coords = list(reversed(coords))

        near = coords[0]

        if not path_rot:
            path_rot.extend(coords)
        else:
            last = path_rot[-1]
            if math.hypot(last[0] - near[0], last[1] - near[1]) > 1e-9:
                transition = vis.shortest_path(last, near)
                path_rot.extend(transition[1:])
            path_rot.extend(coords[1:])

    if not path_rot:
        return []

    cx, cy = centroid.x, centroid.y
    cos_a = math.cos(math.radians(angle_deg))
    sin_a = math.sin(math.radians(angle_deg))
    result: list[tuple[float, float]] = []
    for px, py in path_rot:
        if math.isnan(px):
            result.append((px, py))
            continue
        dx, dy = px - cx, py - cy
        result.append((cx + cos_a * dx - sin_a * dy, cy + sin_a * dx + cos_a * dy))
    return result


def build_zone_snake(
    zone_polygon: Polygon,
    swath_width: float,
    angle_rad: float = None,
    start_position=None,
    strategy: str = "greedy_safe",
    exact: bool = False,
) -> tuple[list[tuple[float, float]], float, int]:
    if zone_polygon.is_empty:
        return [], 0.0, 0

    if angle_rad is None:
        angle_rad = find_optimal_angle(zone_polygon)

    if start_position is None:
        start_position = (zone_polygon.centroid.x, zone_polygon.centroid.y)

    if strategy == "bcd":
        path = _bcd_snake_path(
            zone_polygon, swath_width, angle_rad, start_position, exact=exact
        )
    else:
        path = _greedy_safe_path(
            zone_polygon, swath_width, angle_rad, start_position, exact=exact
        )

    num_turns = count_turns(path)
    return path, angle_rad, num_turns


def count_turns(path: list, angle_threshold_deg: float = 10.0) -> int:
    real_points = [p for p in path if not math.isnan(p[0])]

    if len(real_points) < 3:
        return 0

    turns = 0
    threshold = math.radians(angle_threshold_deg)

    for i in range(1, len(real_points) - 1):
        x0, y0 = real_points[i - 1]
        x1, y1 = real_points[i]
        x2, y2 = real_points[i + 1]

        dx1, dy1 = x1 - x0, y1 - y0
        dx2, dy2 = x2 - x1, y2 - y1

        len1 = math.hypot(dx1, dy1)
        len2 = math.hypot(dx2, dy2)
        if len1 < 1e-10 or len2 < 1e-10:
            continue

        cos_a = (dx1 * dx2 + dy1 * dy2) / (len1 * len2)
        cos_a = max(-1.0, min(1.0, cos_a))
        angle = math.acos(cos_a)

        if angle > threshold:
            turns += 1

    return turns
