import logging
import math
from typing import Callable, List, Tuple

from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from .visibility_graph import VisibilityGraph, _transition_waypoints

logger = logging.getLogger(__name__)


def extract_linestring(geom) -> list[LineString]:
    if geom is None or geom.is_empty:
        return []
    gtype = geom.geom_type
    if gtype == "LineString":
        return [geom]
    if gtype in ("MultiLineString", "GeometryCollection"):
        result: list[LineString] = []
        for part in geom.geoms:
            result.extend(extract_linestring(part))
        return result
    return []


def rotate_point_back(
    px: float,
    py: float,
    cx: float,
    cy: float,
    cos_a: float,
    sin_a: float,
) -> tuple[float, float]:
    dx, dy = px - cx, py - cy
    return cx + cos_a * dx - sin_a * dy, cy + sin_a * dx + cos_a * dy


def rotate_path_points(
    pts: list[tuple[float, float]],
    centroid: Point,
    angle_deg: float,
) -> list[tuple[float, float]]:
    cx, cy = centroid.x, centroid.y
    cos_a = math.cos(math.radians(angle_deg))
    sin_a = math.sin(math.radians(angle_deg))
    out: list[tuple[float, float]] = []
    for px, py in pts:
        if math.isnan(px):
            out.append((px, py))
            continue
        dx, dy = px - cx, py - cy
        out.append((cx + cos_a * dx - sin_a * dy, cy + sin_a * dx + cos_a * dy))
    return out


def angle_between(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> float:
    dx1, dy1 = p1[0] - p0[0], p1[1] - p0[1]
    dx2, dy2 = p2[0] - p1[0], p2[1] - p1[1]
    len1 = math.hypot(dx1, dy1)
    len2 = math.hypot(dx2, dy2)
    if len1 < 1e-10 or len2 < 1e-10:
        return 0.0
    cos_a = (dx1 * dx2 + dy1 * dy2) / (len1 * len2)
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.acos(cos_a)


def count_turns(path: list, angle_threshold_deg: float = 10.0) -> int:
    real_points = [p for p in path if not math.isnan(p[0])]
    if len(real_points) < 3:
        return 0
    turns = 0
    threshold = math.radians(angle_threshold_deg)
    for i in range(1, len(real_points) - 1):
        angle = angle_between(real_points[i - 1], real_points[i], real_points[i + 1])
        if angle > threshold:
            turns += 1
    return turns


def path_length(path: list[tuple[float, float]]) -> float:
    total = 0.0
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        if math.isnan(x1) or math.isnan(x2):
            continue
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def two_opt(
    items: list,
    cost: Callable[[int, int], float],
    swap_fn: Callable[[list, int, int], list] | None = None,
) -> list:
    if len(items) < 4:
        return list(items)

    def default_swap(arr, i, j):
        segment = arr[i : j + 1]
        segment.reverse()
        return arr[:i] + segment + arr[j + 1 :]

    _swap = swap_fn or default_swap

    best = list(items)
    n = len(best)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                before = cost(best[i - 1], best[i])
                if j + 1 < n:
                    before += cost(best[j], best[j + 1])
                after = cost(best[i - 1], best[j])
                if j + 1 < n:
                    after += cost(best[i], best[j + 1])
                if after < before - 1e-9:
                    best = _swap(best, i, j)
                    improved = True
                    break
            if improved:
                break
    return best


def horizontal_swath_segments(
    safe_poly: Polygon, swath_width: float
) -> list[LineString]:
    min_x, min_y, max_x, max_y = safe_poly.bounds
    if max_y - min_y < 1e-9:
        return []
    margin = (max_x - min_x) + 1.0
    eps = min(swath_width * 0.01, 1e-3 + swath_width * 1e-6)
    ys: list[float] = []
    y = min_y + eps
    while y <= max_y - eps + 1e-9:
        ys.append(y)
        y += swath_width
    top_y = max_y - eps
    if not ys or ys[-1] + 1e-9 < top_y:
        ys.append(top_y)

    all_segs: list[LineString] = []
    for y in ys:
        sweep_line = LineString([(min_x - margin, y), (max_x + margin, y)])
        try:
            inter = safe_poly.intersection(sweep_line)
        except Exception:
            continue
        for seg in extract_linestring(inter):
            if seg.length > 1e-9:
                all_segs.append(seg)
    return all_segs


def _collect_holes(polygon: Polygon | MultiPolygon) -> list[Polygon]:
    holes: list[Polygon] = []
    polys = list(polygon.geoms) if isinstance(polygon, MultiPolygon) else [polygon]
    for poly in polys:
        if hasattr(poly, "interiors"):
            for interior in poly.interiors:
                holes.append(Polygon(interior.coords))
    return holes


def ensure_path_avoids_holes(
    path: list[tuple[float, float]],
    polygon: Polygon | MultiPolygon,
    vis: VisibilityGraph | None = None,
) -> list[tuple[float, float]]:
    holes = _collect_holes(polygon)
    if not holes:
        return path

    if vis is None:
        vis = VisibilityGraph(polygon)

    logger.debug(
        "Starting hole-avoidance post-processing, path length=%d, holes=%d",
        len(path),
        len(holes),
    )

    i = 0
    while i < len(path) - 1:
        p1 = path[i]
        p2 = path[i + 1]
        if math.isnan(p1[0]) or math.isnan(p2[0]):
            i += 1
            continue

        seg = LineString([p1, p2])
        visible = vis._visible(p1, p2)
        intersects_hole = _segment_intersects_hole(seg, holes)

        if not visible or intersects_hole:
            logger.debug(
                "Segment %s -> %s visible=%s intersects_hole=%s, replacing",
                p1,
                p2,
                visible,
                intersects_hole,
            )
            detour = vis.shortest_path(p1, p2)
            if len(detour) == 2:
                detour = _transition_waypoints(p1, p2, vis._original_polygon)

            if _detour_unchanged(detour, p1, p2):
                logger.debug("Detour identical to original, skipping")
                i += 1
                continue

            path[i + 1 : i + 2] = detour[1:]
            continue
        i += 1

    logger.debug("Post-processing finished, path length=%d", len(path))
    return path


def _segment_intersects_hole(seg: LineString, holes: list[Polygon]) -> bool:
    for hole_poly in holes:
        if hole_poly.intersects(seg):
            if not seg.touches(hole_poly):
                return True
    return False


def _detour_unchanged(
    detour: list[tuple[float, float]],
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> bool:
    return (
        len(detour) == 2
        and abs(detour[0][0] - p1[0]) < 1e-9
        and abs(detour[0][1] - p1[1]) < 1e-9
        and abs(detour[1][0] - p2[0]) < 1e-9
        and abs(detour[1][1] - p2[1]) < 1e-9
    )


def split_at_turns(
    path: list[tuple[float, float]],
    angle_threshold: float = math.pi / 4,
) -> list[list[tuple[float, float]]]:
    if not path:
        return []
    segments = []
    current = []
    for i, pt in enumerate(path):
        if math.isnan(pt[0]):
            if current:
                segments.append(current)
                current = []
            segments.append([pt])
            continue
        current.append(pt)
        if len(current) < 2:
            continue
        next_real = None
        for j in range(i + 1, len(path)):
            if not math.isnan(path[j][0]):
                next_real = path[j]
                break
        if next_real is None:
            continue
        angle = angle_between(current[-2], current[-1], next_real)
        if angle > angle_threshold:
            segments.append(current)
            current = [current[-1]]
    if current:
        segments.append(current)
    return segments


def pts_eq(a: tuple[float, float], b: tuple[float, float], eps: float = 1e-6) -> bool:
    return math.hypot(a[0] - b[0], a[1] - b[1]) < eps


_NAN = (float("nan"), float("nan"))


def apply_resource_limits(
    path: List[Tuple[float, float]],
    start_position: Tuple[float, float],
    speed: float,
    substance_rate: float = 0.0,
    tank_volume: float = float("inf"),
    max_flight_time: float = float("inf"),
    turn_time: float = 1.0,
    obstacles_polygon=None,
) -> Tuple[List[Tuple[float, float]], int]:
    no_substance = substance_rate <= 0 or not math.isfinite(tank_volume)
    no_time = not math.isfinite(max_flight_time)
    if no_substance and no_time:
        return list(path), 0

    _vis = None
    if obstacles_polygon is not None and not obstacles_polygon.is_empty:
        from .visibility_graph import VisibilityGraph

        _vis = VisibilityGraph(obstacles_polygon)

    def _rth_waypoints(from_pt: Tuple[float, float]) -> List[Tuple[float, float]]:
        if _vis is not None:
            return _vis.shortest_path(from_pt, start_position)
        return [from_pt, start_position]

    def _rth_path_length(from_pt: Tuple[float, float]) -> float:
        wps = _rth_waypoints(from_pt)
        return sum(
            math.hypot(wps[i + 1][0] - wps[i][0], wps[i + 1][1] - wps[i][1])
            for i in range(len(wps) - 1)
        )

    segments = split_at_turns(path)
    result = []
    substance_used = 0.0
    flight_time_used = 0.0
    rth_count = 0
    current_pos = start_position
    after_rth = True

    for seg in segments:
        if len(seg) == 1 and math.isnan(seg[0][0]):
            result.append(seg[0])
            after_rth = True
            continue
        seg_length = path_length(seg)
        if not no_substance and speed > 0:
            seg_time_min = (seg_length / speed) / 60.0  # minutes
            seg_substance = substance_rate * seg_time_min
        else:
            seg_substance = 0.0
        seg_time = (seg_length / speed if speed > 0 else 0.0) + (
            0.0 if after_rth else turn_time
        )
        end_pt = seg[-1]
        rth_dist_end = _rth_path_length(end_pt)
        rth_time_end = rth_dist_end / speed if speed > 0 else 0.0

        substance_ok = no_substance or (substance_used + seg_substance <= tank_volume)
        time_ok = no_time or (
            flight_time_used + seg_time + rth_time_end <= max_flight_time
        )

        if not substance_ok or not time_ok:
            home_path = _rth_waypoints(current_pos)
            home_dist = sum(
                math.hypot(
                    home_path[i + 1][0] - home_path[i][0],
                    home_path[i + 1][1] - home_path[i][1],
                )
                for i in range(len(home_path) - 1)
            )
            home_time = home_dist / speed if speed > 0 else 0.0
            can_reach_home = no_time or (
                flight_time_used + home_time <= max_flight_time
            )
            if can_reach_home and not after_rth:
                if not result or not math.isnan(result[-1][0]):
                    result.append(_NAN)
                result.extend(home_path[1:])
                result.append(_NAN)
                return_path = list(reversed(home_path))
                result.extend(return_path[1:])

                rth_count += 1
                substance_used = 0.0
                flight_time_used = home_dist / speed if speed > 0 else 0.0
                seg_time = seg_length / speed if speed > 0 else 0.0

        last_real = next((p for p in reversed(result) if not math.isnan(p[0])), None)
        if last_real is not None and pts_eq(last_real, seg[0]):
            result.extend(seg[1:])
        else:
            result.extend(seg)
        substance_used += seg_substance
        flight_time_used += seg_time
        current_pos = seg[-1]
        after_rth = False

    return result, rth_count
