import math
from dataclasses import dataclass, field
from typing import cast

from shapely.affinity import rotate
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    box,
)
from shapely.geometry import (
    Polygon as ShapelyPolygon,
)
from shapely.ops import unary_union

_NAN = (float("nan"), float("nan"))


def _pts_eq(a: tuple[float, ...], b: tuple[float, ...], eps: float = 1e-9) -> bool:
    return math.hypot(a[0] - b[0], a[1] - b[1]) < eps


@dataclass
class Cell:
    idx: int
    poly: ShapelyPolygon
    x_min: float
    x_max: float
    neighbours: set[int] = field(default_factory=set)


@dataclass
class Swath:
    segments: list[list[tuple[float, float]]]


def _build_adjacency(cells: list[Cell], eps: float = 1e-6) -> None:
    n = len(cells)
    for i in range(n):
        for j in range(i + 1, n):
            x_adjacent = (abs(cells[i].x_max - cells[j].x_min) < eps or
                          abs(cells[i].x_min - cells[j].x_max) < eps)

            if x_adjacent and cells[i].poly.distance(cells[j].poly) < eps:
                cells[i].neighbours.add(j)
                cells[j].neighbours.add(i)


def _traversal_order(cells: list[Cell], start_pt: tuple[float, float]) -> list[int]:
    if not cells:
        return []

    start_idx = 0
    min_dist = math.inf
    pt = Point(start_pt)

    for i, cell in enumerate(cells):
        if cell.poly.contains(pt):
            start_idx = i
            break
        d = cell.poly.distance(pt)
        if d < min_dist:
            min_dist = d
            start_idx = i

    visited: set[int] = set()
    order: list[int] = []

    def dfs(u: int):
        visited.add(u)
        order.append(u)
        for v in sorted(cells[u].neighbours):
            if v not in visited:
                dfs(v)

    dfs(start_idx)

    for i in range(len(cells)):
        if i not in visited:
            dfs(i)

    return order


def _segment_inside_area(
        p1: tuple[float, float],
        p2: tuple[float, float],
        area: ShapelyPolygon,
) -> bool:
    if _pts_eq(p1, p2):
        return True
    return area.buffer(1e-8).covers(LineString([p1, p2]))


def _cover_swath_list(
        swaths: list[Swath],
        go_up: bool,
        check_transitions: bool = False,
        safe_area: ShapelyPolygon | None = None,
) -> tuple[list[tuple[float, float]], bool]:
    path: list[tuple[float, float]] = []

    for swath_idx, swath in enumerate(swaths):
        segments_sorted = sorted(swath.segments, key=lambda s: min(pt[1] for pt in s))
        if not go_up:
            segments_sorted = list(reversed(segments_sorted))

        if swath_idx > 0 and path and not math.isnan(path[-1][0]):
            first_point = segments_sorted[0][0] if go_up else segments_sorted[0][-1]
            if not _pts_eq(path[-1], first_point, eps=1e-6):
                if check_transitions and safe_area is not None:
                    if not _segment_inside_area(path[-1], first_point, safe_area):
                        path.append(_NAN)
                    else:
                        path.append(first_point)
                else:
                    path.append(first_point)

        for segment in segments_sorted:
            if not go_up:
                segment = list(reversed(segment))
            if (path
                    and not math.isnan(path[-1][0])
                    and not _pts_eq(path[-1], segment[0], eps=1e-6)):
                if check_transitions and safe_area is not None:
                    if not _segment_inside_area(path[-1], segment[0], safe_area):
                        path.append(_NAN)
                    else:
                        path.append(segment[0])
                else:
                    path.append(segment[0])

            path.extend(segment)

        go_up = not go_up

    return path, go_up


def _simple_boustrophedon(
        poly: ShapelyPolygon,
        safe_area: ShapelyPolygon,
        swath: float,
        start_pt: tuple[float, float],
        check_transitions: bool = False,
) -> list[tuple[float, float]]:
    minx, miny, maxx, maxy = poly.bounds
    width = maxx - minx

    if width < 1e-9:
        return []

    n_swaths = max(1, int(math.ceil(width / swath)))

    swaths_by_x: list[Swath] = []
    start_idx = 0
    min_dist = float("inf")

    for i in range(n_swaths):
        x = minx + swath / 2 + i * swath
        if x > maxx:
            break

        line = LineString([(x, miny - 10), (x, maxy + 10)])
        intersection = line.intersection(safe_area)

        if intersection.is_empty:
            continue

        segments = []
        if isinstance(intersection, LineString):
            segments.append(list(intersection.coords))
        elif isinstance(intersection, MultiLineString):
            for seg in intersection.geoms:
                segments.append(list(seg.coords))

        if segments:
            dist = abs(x - start_pt[0])
            if dist < min_dist:
                min_dist = dist
                start_idx = len(swaths_by_x)
            swaths_by_x.append(Swath(segments))

    if not swaths_by_x:
        return []

    right_part = swaths_by_x[start_idx:]
    left_part = list(reversed(swaths_by_x[:start_idx]))

    first_swath = right_part[0] if right_part else left_part[0]
    first_y = sum(pt[1] for seg in first_swath.segments for pt in seg) / sum(len(seg) for seg in first_swath.segments)
    go_up = start_pt[1] <= first_y

    path, go_up = _cover_swath_list(
        right_part,
        go_up,
        check_transitions=check_transitions,
        safe_area=safe_area if check_transitions else None
    )

    if left_part:
        if path and not math.isnan(path[-1][0]):
            first_left_swath = left_part[0]
            segments_sorted = sorted(first_left_swath.segments, key=lambda s: min(pt[1] for pt in s))
            if not go_up:
                segments_sorted = list(reversed(segments_sorted))
            first_left_point = segments_sorted[0][0] if go_up else segments_sorted[0][-1]

            if check_transitions and safe_area is not None:
                if not _segment_inside_area(path[-1], first_left_point, safe_area):
                    path.append(_NAN)
                else:
                    path.append(first_left_point)
            else:
                path.append(first_left_point)
        else:
            path.append(_NAN)

        left_path, _ = _cover_swath_list(
            left_part,
            go_up,
            check_transitions=check_transitions,
            safe_area=safe_area if check_transitions else None
        )
        path.extend(left_path)

    return path


def _find_transition(
        cell_from: Cell,
        cell_to: Cell,
        point_from: tuple[float, float],
        point_to: tuple[float, float],
        safe_area: ShapelyPolygon | None = None,
) -> list[tuple[float, float]]:
    eps = 1e-6

    shared_x = None
    if abs(cell_from.x_max - cell_to.x_min) < eps:
        shared_x = (cell_from.x_max + cell_to.x_min) / 2
    elif abs(cell_from.x_min - cell_to.x_max) < eps:
        shared_x = (cell_from.x_min + cell_to.x_max) / 2

    if shared_x is None:
        if _pts_eq(point_from, point_to, eps=1e-6):
            return []
        raw = [point_to]
    elif abs(point_from[0] - shared_x) < eps:
        mid = (shared_x, point_to[1])
        raw = [mid] if not _pts_eq(point_from, mid, eps=eps) else []
    else:
        mid1 = (shared_x, point_from[1])
        mid2 = (shared_x, point_to[1])
        raw = []
        if not _pts_eq(point_from, mid1, eps=eps):
            raw.append(mid1)
        if not _pts_eq(mid1, mid2, eps=eps):
            raw.append(mid2)

    if not raw or safe_area is None:
        return raw

    result: list[tuple[float, float]] = []
    prev = point_from
    for pt in raw:
        if not math.isnan(prev[0]) and not _segment_inside_area(prev, pt, safe_area):
            result.append(_NAN)
        result.append(pt)
        prev = pt

    return result


def _last_real(path: list[tuple[float, float]], fallback: tuple[float, float]) -> tuple[float, float]:
    return next((p for p in reversed(path) if not math.isnan(p[0])), fallback)


def _assemble_bcd_path(
        cells: list[Cell],
        order: list[int],
        swath: float,
        start_pt: tuple[float, float],
        safe_area: ShapelyPolygon | None = None,
) -> list[tuple[float, float]]:
    full_path = []

    for idx, cid in enumerate(order):
        cell = cells[cid]

        if idx == 0:
            entry_point = start_pt
        else:
            entry_point = _last_real(full_path, start_pt)

        cell_path = _simple_boustrophedon(
            cell.poly,
            cell.poly,
            swath,
            entry_point,
            check_transitions=False
        )

        if not cell_path:
            continue

        if idx == 0:
            full_path = cell_path
        else:
            prev_cell = cells[order[idx - 1]]
            last = _last_real(full_path, start_pt)
            transition = _find_transition(prev_cell, cell, last, cell_path[0], safe_area)

            if transition:
                full_path.extend(transition)

            tail = full_path[-1]
            if not math.isnan(tail[0]) and not _pts_eq(tail, cell_path[0], eps=1e-6):
                full_path.append(cell_path[0])
            full_path.extend(cell_path[1:])

    return full_path


def _connect_sub_paths(
        sub_paths: list[list[tuple[float, float]]],
        start: tuple[float, float]
) -> list[tuple[float, float]]:
    if not sub_paths:
        return []

    result = []
    current = start
    remaining = list(range(len(sub_paths)))

    while remaining:
        best_idx = 0
        best_dist = math.inf
        best_reverse = False

        for i in remaining:
            sp = sub_paths[i]
            d_start = math.hypot(sp[0][0] - current[0], sp[0][1] - current[1])
            d_end = math.hypot(sp[-1][0] - current[0], sp[-1][1] - current[1])

            if d_start < best_dist:
                best_dist = d_start
                best_idx = i
                best_reverse = False

            if d_end < best_dist:
                best_dist = d_end
                best_idx = i
                best_reverse = True

        sp = sub_paths[best_idx]
        if best_reverse:
            sp = list(reversed(sp))

        if result:
            result.append(_NAN)
        result.extend(sp)
        current = sp[-1]
        remaining.remove(best_idx)

    return result


class BoustrophedonCoverage:
    def __init__(self, poly: ShapelyPolygon, angle: float = 0.0):
        self.original = poly
        self.angle = angle
        self.poly = rotate(poly, -angle, origin="centroid")

    def generate_coverage_path(
            self, swath: float, start: tuple[float, float]
    ) -> list[tuple[float, float]]:
        if self.poly.is_empty:
            return []

        start_rot = rotate(Point(start), -self.angle, origin=self.original.centroid)
        start_pt = (start_rot.x, start_rot.y)

        safe_area = self._make_safe_area(swath)

        if safe_area.is_empty:
            return []

        if isinstance(safe_area, MultiPolygon):
            return self._handle_multipolygon(safe_area, swath, start)

        raw_path = self._generate_for_safe_poly(self.poly, safe_area, swath, start_pt)
        return self._rotate_back(raw_path)

    def _make_safe_area(self, swath: float) -> ShapelyPolygon | MultiPolygon:
        outer = self.poly.buffer(-swath / 2)

        if outer.is_empty:
            return outer

        expanded_holes = []
        for interior in self.poly.interiors:
            hole_poly = ShapelyPolygon(interior)
            expanded = hole_poly.buffer(swath / 2)
            expanded_holes.append(expanded)

        if expanded_holes:
            holes_union = unary_union(expanded_holes)
            safe = outer.difference(holes_union)

            if isinstance(safe, (ShapelyPolygon, MultiPolygon)):
                return safe
            else:
                return ShapelyPolygon()

        return outer

    def _bcd_decompose(
            self,
            poly: ShapelyPolygon,
            safe_area: ShapelyPolygon,
    ) -> list[Cell]:
        minx, miny, maxx, maxy = poly.bounds

        critical_x = set()

        for pt in safe_area.exterior.coords:
            critical_x.add(pt[0])

        for interior in safe_area.interiors:
            for pt in interior.coords:
                critical_x.add(pt[0])

        critical_x.add(minx)
        critical_x.add(maxx)

        critical_x = sorted(critical_x)

        cells: list[Cell] = []
        cell_id = 0

        prev_x: float | None = None
        prev_intervals: list[tuple[float, float]] = []
        current_cells: list[ShapelyPolygon | None] = []

        for x in critical_x:
            line = LineString([(x, miny - 10), (x, maxy + 10)])
            inter = line.intersection(safe_area)
            intervals = self._extract_y_intervals(inter)

            if prev_x is not None:
                if prev_intervals and len(intervals) == len(prev_intervals):
                    for i in range(len(current_cells)):
                        cell = current_cells[i]
                        if cell is not None:
                            py0, py1 = prev_intervals[i]
                            cy0, cy1 = intervals[i]
                            slice_box = box(prev_x, min(py0, cy0), x, max(py1, cy1))
                            slice_poly = safe_area.intersection(slice_box)

                            if not slice_poly.is_empty:
                                current_cells[i] = cast(ShapelyPolygon, unary_union([cell, slice_poly]))
                else:
                    for cell_poly in current_cells:
                        if cell_poly is not None and cell_poly.area > 1e-9:
                            bounds = cell_poly.bounds
                            cells.append(Cell(
                                idx=cell_id,
                                poly=cell_poly,
                                x_min=bounds[0],
                                x_max=bounds[2]
                            ))
                            cell_id += 1

                    current_cells = []
                    for y_low, y_high in intervals:
                        slice_box = box(prev_x, y_low, x, y_high)
                        slice_poly = safe_area.intersection(slice_box)
                        current_cells.append(slice_poly if not slice_poly.is_empty else None)
            else:
                current_cells = [None] * len(intervals)

            prev_x = x
            prev_intervals = intervals

        for cell_poly in current_cells:
            if cell_poly is not None and cell_poly.area > 1e-9:
                bounds = cell_poly.bounds
                cells.append(Cell(
                    idx=cell_id,
                    poly=cell_poly,
                    x_min=bounds[0],
                    x_max=bounds[2]
                ))
                cell_id += 1

        return cells

    @staticmethod
    def _extract_y_intervals(geom) -> list[tuple[float, float]]:
        intervals = []
        if geom.is_empty:
            return intervals

        if geom.geom_type == "LineString":
            ys = sorted(pt[1] for pt in geom.coords)
            if ys:
                intervals.append((ys[0], ys[-1]))
        elif geom.geom_type in ("MultiLineString", "GeometryCollection"):
            for part in geom.geoms:
                intervals.extend(BoustrophedonCoverage._extract_y_intervals(part))

        intervals.sort(key=lambda s: s[0])
        return intervals

    def _handle_multipolygon(
            self,
            mp: MultiPolygon,
            swath: float,
            start: tuple[float, float]
    ) -> list[tuple[float, float]]:
        s_rot = rotate(Point(start), -self.angle, origin=self.original.centroid)
        cur = (s_rot.x, s_rot.y)

        sub_paths = []
        for poly in mp.geoms:
            planner = BoustrophedonCoverage(poly, 0.0)
            sub = planner._generate_for_safe_poly(poly, poly, swath,
                                                  (poly.centroid.x, poly.centroid.y))
            if sub:
                sub_paths.append(sub)

        full_path = _connect_sub_paths(sub_paths, cur)
        return self._rotate_back(full_path)

    def _generate_for_safe_poly(
            self,
            poly: ShapelyPolygon,
            safe_poly: ShapelyPolygon,
            swath: float,
            start: tuple[float, float]
    ) -> list[tuple[float, float]]:
        has_holes = len(list(safe_poly.interiors)) > 0

        if has_holes:
            cells = self._bcd_decompose(poly, safe_poly)
            if cells:
                _build_adjacency(cells)
                order = _traversal_order(cells, start)
                return _assemble_bcd_path(cells, order, swath, start, safe_poly)

        return _simple_boustrophedon(
            poly,
            safe_poly,
            swath,
            start,
            check_transitions=False
        )

    def _rotate_back(self, path: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if not path:
            return path
        cen = self.original.centroid
        rad = math.radians(self.angle)
        ca, sa = math.cos(rad), math.sin(rad)
        out = []
        for x, y in path:
            if math.isnan(x):
                out.append(_NAN)
                continue
            dx, dy = x - cen.x, y - cen.y
            out.append((cen.x + ca * dx - sa * dy, cen.y + sa * dx + ca * dy))
        return out


def generate_boustrophedon_coverage(
        polygon: ShapelyPolygon,
        swath: float,
        angle: float = 0.0,
        start_position: tuple[float, float] | None = None,
) -> list[tuple[float, float]]:
    if start_position is None:
        start_position = (polygon.centroid.x, polygon.centroid.y)
    return BoustrophedonCoverage(polygon, angle).generate_coverage_path(swath, start_position)


def path_length(path: list[tuple[float, float]]) -> float:
    total = 0.0
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        if math.isnan(x1) or math.isnan(x2):
            continue
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def _split_at_turns(
        path: list[tuple[float, float]],
        angle_threshold: float = math.pi / 4,
) -> list[list[tuple[float, float]]]:
    if not path:
        return []

    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []

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

        next_real: tuple[float, float] | None = None
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
        path: list[tuple[float, float]],
        start_position: tuple[float, float],
        speed: float,
        substance_rate: float = 0.0,
        tank_volume: float = float("inf"),
        max_flight_time: float = float("inf"),
        turn_time: float = 1.0,
) -> tuple[list[tuple[float, float]], int]:
    no_substance = substance_rate <= 0 or not math.isfinite(tank_volume)
    no_time = not math.isfinite(max_flight_time)

    if no_substance and no_time:
        return list(path), 0

    segments = _split_at_turns(path)

    result: list[tuple[float, float]] = []
    substance_used = 0.0
    flight_time_used = 0.0
    rth_count = 0
    current_pos: tuple[float, float] = start_position
    after_rth = True

    for seg in segments:
        if len(seg) == 1 and math.isnan(seg[0][0]):
            result.append(seg[0])
            after_rth = True
            continue

        seg_length = path_length(seg)
        seg_substance = seg_length * substance_rate if not no_substance else 0.0
        seg_time = (seg_length / speed if speed > 0 else 0.0) + (0.0 if after_rth else turn_time)

        end_pt = seg[-1]
        rth_dist_end = math.hypot(end_pt[0] - start_position[0],
                                  end_pt[1] - start_position[1])
        rth_time_end = rth_dist_end / speed if speed > 0 else 0.0

        substance_ok = no_substance or (substance_used + seg_substance <= tank_volume)
        time_ok = no_time or (flight_time_used + seg_time + rth_time_end <= max_flight_time)

        if not substance_ok or not time_ok:
            home_dist = math.hypot(current_pos[0] - start_position[0],
                                   current_pos[1] - start_position[1])
            home_time = home_dist / speed if speed > 0 else 0.0
            can_reach_home = no_time or (flight_time_used + home_time <= max_flight_time)

            if can_reach_home and not after_rth:
                if not result or not math.isnan(result[-1][0]):
                    result.append(_NAN)
                if not _pts_eq(start_position, result[-1] if result else current_pos):
                    result.append(start_position)
                result.append(_NAN)
                if not _pts_eq(current_pos, result[-1] if result else start_position):
                    result.append(current_pos)
                rth_count += 1

                substance_used = 0.0
                return_dist = math.hypot(current_pos[0] - start_position[0],
                                         current_pos[1] - start_position[1])
                flight_time_used = return_dist / speed if speed > 0 else 0.0
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
