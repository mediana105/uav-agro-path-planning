from __future__ import annotations

import math
from typing import List, Optional, Tuple

from shapely.affinity import rotate
from shapely.geometry import (
    LineString,
    Point,
    Polygon as ShapelyPolygon,
)


def _extract_linestring(geom) -> List[LineString]:
    if geom is None or geom.is_empty:
        return []
    gtype = geom.geom_type
    if gtype == "LineString":
        return [geom]
    if gtype in ("MultiLineString", "GeometryCollection"):
        result: List[LineString] = []
        for part in geom.geoms:
            result.extend(_extract_linestring(part))
        return result
    return []


def _rotate_point_back(
    px: float,
    py: float,
    cx: float,
    cy: float,
    cos_a: float,
    sin_a: float,
) -> Tuple[float, float]:
    dx, dy = px - cx, py - cy
    return cx + cos_a * dx - sin_a * dy, cy + sin_a * dx + cos_a * dy


def generate_boustrophedon_coverage(
    polygon: ShapelyPolygon,
    swath: float,
    angle: float = 0.0,  # radians
    start_position: Optional[Tuple[float, float]] = None,
) -> List[Tuple[float, float]]:
    if polygon.is_empty or swath <= 0:
        return []

    centroid = polygon.centroid
    angle_deg = math.degrees(angle)

    rotated = rotate(polygon, -angle_deg, origin=centroid, use_radians=False)
    min_x, min_y, max_x, max_y = rotated.bounds
    if max_y - min_y < 1e-9:
        return []

    margin = (max_x - min_x) + 1.0

    ys: List[float] = []
    y = min_y + swath / 2.0
    while y < max_y:
        ys.append(y)
        y += swath
    if not ys:
        ys = [(min_y + max_y) / 2.0]

    rows: List[List[LineString]] = []
    for y_val in ys:
        sweep_line = LineString([(min_x - margin, y_val), (max_x + margin, y_val)])
        try:
            inter = rotated.intersection(sweep_line)
        except Exception:
            continue
        segments = [s for s in _extract_linestring(inter) if s.length > 1e-9]
        if segments:
            segments.sort(key=lambda s: s.centroid.x)
            rows.append(segments)

    if not rows:
        return []

    if start_position is not None:
        rot_pt = rotate(
            Point(start_position), -angle_deg, origin=centroid, use_radians=False
        )
        sx, sy = rot_pt.x, rot_pt.y
        if abs(sy - rows[-1][0].centroid.y) < abs(sy - rows[0][0].centroid.y):
            rows.reverse()
        row0_xs = [s.centroid.x for s in rows[0]]
        left_to_right = sx <= (min(row0_xs) + max(row0_xs)) / 2.0
    else:
        left_to_right = True

    waypoints_rot: List[Tuple[float, float]] = []

    for segments in rows:
        ordered = list(segments)
        if not left_to_right:
            ordered = [
                LineString(list(reversed(list(s.coords)))) for s in reversed(ordered)
            ]
        for seg in ordered:
            cords: List[Tuple[float, float]] = [(x, y) for x, y in seg.coords]
            if not waypoints_rot:
                waypoints_rot.extend(cords)
            else:
                last = waypoints_rot[-1]
                if math.hypot(last[0] - cords[0][0], last[1] - cords[0][1]) > 1e-9:
                    waypoints_rot.append(cords[0])
                waypoints_rot.extend(cords[1:])
        left_to_right = not left_to_right

    if not waypoints_rot:
        return []

    cos_a = math.cos(math.radians(angle_deg))
    sin_a = math.sin(math.radians(angle_deg))
    cx, cy = centroid.x, centroid.y

    return [
        _rotate_point_back(px, py, cx, cy, cos_a, sin_a) for px, py in waypoints_rot
    ]


def generate_greedy_coverage(
    polygon: ShapelyPolygon,
    swath: float,
    angle: float = 0.0,  # radians
    start_position: Optional[Tuple[float, float]] = None,
) -> List[Tuple[float, float]]:
    if polygon.is_empty or swath <= 0:
        return []

    centroid = polygon.centroid
    angle_deg = math.degrees(angle)

    rotated = rotate(polygon, -angle_deg, origin=centroid, use_radians=False)
    min_x, min_y, max_x, max_y = rotated.bounds
    if max_y - min_y < 1e-9:
        return []

    margin = (max_x - min_x) + 1.0

    all_segments: List[LineString] = []
    y = min_y + swath / 2.0
    while y < max_y:
        sweep_line = LineString([(min_x - margin, y), (max_x + margin, y)])
        try:
            inter = rotated.intersection(sweep_line)
        except Exception:
            y += swath
            continue
        for seg in _extract_linestring(inter):
            if seg.length > 1e-9:
                all_segments.append(seg)
        y += swath

    if not all_segments:
        return []

    if start_position is not None:
        rot_pt = rotate(
            Point(start_position), -angle_deg, origin=centroid, use_radians=False
        )
        cur_x, cur_y = rot_pt.x, rot_pt.y
    else:
        cur_x = (min_x + max_x) / 2.0
        cur_y = min_y + swath / 2.0

    remaining = list(all_segments)
    waypoints_rot: List[Tuple[float, float]] = []

    while remaining:
        best_i: int = 0
        best_dist: float = math.inf
        best_flip: bool = False

        for i, seg in enumerate(remaining):
            cords: List[Tuple[float, float]] = [(x, y) for x, y in seg.coords]
            d0 = math.hypot(cords[0][0] - cur_x, cords[0][1] - cur_y)
            d1 = math.hypot(cords[-1][0] - cur_x, cords[-1][1] - cur_y)
            if d0 <= d1:
                d, flip = d0, False
            else:
                d, flip = d1, True
            if d < best_dist:
                best_dist, best_i, best_flip = d, i, flip

        seg = remaining.pop(best_i)
        cords: List[Tuple[float, float]] = [(x, y) for x, y in seg.coords]
        if best_flip:
            cords = list(reversed(cords))

        near = cords[0]

        if not waypoints_rot:
            waypoints_rot.extend(cords)
        else:
            last = waypoints_rot[-1]
            if math.hypot(last[0] - near[0], last[1] - near[1]) > 1e-9:
                waypoints_rot.append(near)
            waypoints_rot.extend(cords[1:])

        cur_x, cur_y = cords[-1]

    if not waypoints_rot:
        return []

    cos_a = math.cos(math.radians(angle_deg))
    sin_a = math.sin(math.radians(angle_deg))
    cx, cy = centroid.x, centroid.y

    return [
        _rotate_point_back(px, py, cx, cy, cos_a, sin_a) for px, py in waypoints_rot
    ]


def path_length(path: List[Tuple[float, float]]) -> float:
    total = 0.0
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        if math.isnan(x1) or math.isnan(x2):
            continue
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def decompose_field(
    polygon: ShapelyPolygon,
    swath: float,
    angle: float = 0.0,  # degrees
) -> Tuple[object, List, ShapelyPolygon]:
    from .bcd import bcd_slice_decompose, Cell as BcdCell

    if polygon.is_empty:
        return None, [], polygon

    centroid = polygon.centroid
    rotated = rotate(polygon, -angle, origin=centroid, use_radians=False)
    raw_cells = bcd_slice_decompose(rotated, swath)

    result_cells = []
    for cell in raw_cells:
        poly_back = rotate(cell.poly, angle, origin=centroid, use_radians=False)
        result_cells.append(
            BcdCell(
                idx=cell.idx,
                poly=poly_back,
                x_min=poly_back.bounds[0],
                x_max=poly_back.bounds[2],
                neighbours=cell.neighbours,
            )
        )

    return None, result_cells, polygon


_NAN = (float("nan"), float("nan"))


def _pts_eq(a, b, eps=1e-6):
    return math.hypot(a[0] - b[0], a[1] - b[1]) < eps


def _split_at_turns(path, angle_threshold=math.pi / 4):
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
        p0, p1, p2 = current[-2], current[-1], next_real
        d1 = (p1[0] - p0[0], p1[1] - p0[1])
        d2 = (p2[0] - p1[0], p2[1] - p1[1])
        l1, l2 = math.hypot(*d1), math.hypot(*d2)
        if l1 < 1e-9 or l2 < 1e-9:
            continue
        cos_a = max(-1.0, min(1.0, (d1[0] * d2[0] + d1[1] * d2[1]) / (l1 * l2)))
        if math.acos(cos_a) > angle_threshold:
            segments.append(current)
            current = [current[-1]]
    if current:
        segments.append(current)
    return segments


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

    segments = _split_at_turns(path)
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
        seg_substance = seg_length * substance_rate if not no_substance else 0.0
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
                result.extend(return_path[1:])  # skip start_position duplicate

                rth_count += 1
                substance_used = 0.0
                flight_time_used = home_dist / speed if speed > 0 else 0.0
                seg_time = seg_length / speed if speed > 0 else 0.0

        last_real = next((p for p in reversed(result) if not math.isnan(p[0])), None)
        if last_real is not None and _pts_eq(last_real, seg[0]):
            result.extend(seg[1:])
        else:
            result.extend(seg)
        substance_used += seg_substance
        flight_time_used += seg_time
        current_pos = seg[-1]
        after_rth = False

    return result, rth_count
