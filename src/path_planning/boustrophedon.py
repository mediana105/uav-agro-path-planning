import logging
import math
from dataclasses import dataclass, field
from typing import cast

log = logging.getLogger(__name__)

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


def _build_adjacency(cells: list[Cell], eps: float = 0.05) -> None:
    n = len(cells)
    for i in range(n):
        for j in range(i + 1, n):
            if cells[i].poly.distance(cells[j].poly) < eps:
                cells[i].neighbours.add(j)
                cells[j].neighbours.add(i)


def _traversal_order(cells: list[Cell], start_pt: tuple[float, float]) -> list[int]:
    if not cells:
        return []

    def cell_center(c: Cell) -> tuple[float, float]:
        b = c.poly.bounds
        return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

    def distance_to_point(cell_idx: int, point: tuple[float, float]) -> float:
        cx, cy = cell_center(cells[cell_idx])
        return math.hypot(cx - point[0], cy - point[1])

    unvisited = set(range(len(cells)))
    current = min(unvisited, key=lambda i: distance_to_point(i, start_pt))

    order: list[int] = [current]
    unvisited.remove(current)

    while unvisited:
        adjacent_unvisited = cells[current].neighbours & unvisited
        cur_center = cell_center(cells[current])

        if adjacent_unvisited:
            next_cell = min(adjacent_unvisited, key=lambda i: distance_to_point(i, cur_center))
        else:
            next_cell = min(unvisited, key=lambda i: distance_to_point(i, cur_center))

        order.append(next_cell)
        unvisited.remove(next_cell)
        current = next_cell

    return order


def _segment_crosses_holes_strict(
        p1: tuple[float, float],
        p2: tuple[float, float],
        polygon: ShapelyPolygon,
) -> bool:
    if _pts_eq(p1, p2):
        return False

    segment = LineString([p1, p2])

    segment_buffer = segment.buffer(0.1, cap_style=2)

    for interior in polygon.interiors:
        hole = ShapelyPolygon(interior)

        if segment.crosses(hole.exterior):
            log.debug("  STRICT CROSS: segment crosses hole boundary")
            return True

        if segment_buffer.intersects(hole):
            intersection = segment_buffer.intersection(hole)
            if not intersection.is_empty and intersection.area > 1e-6:
                log.debug("  STRICT CROSS: segment intersects hole interior (area=%.6f)", intersection.area)
                return True

        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2
        mid_point = Point(mid_x, mid_y)
        if hole.buffer(0.5).contains(mid_point):
            log.debug("  STRICT CROSS: segment midpoint inside hole")
            return True

    return False


def _segment_inside_safe_area_strict(
        p1: tuple[float, float],
        p2: tuple[float, float],
        safe_area: ShapelyPolygon,
        eps: float = 1.0,
) -> bool:
    if _pts_eq(p1, p2):
        return safe_area.contains(Point(p1))

    segment = LineString([p1, p2])

    if not safe_area.buffer(eps).contains(segment):
        log.debug("  STRICT SAFE: segment outside safe area buffer")
        return False

    if _segment_crosses_holes_strict(p1, p2, safe_area):
        return False

    return True


def _cover_swath_list(
        swaths: list[Swath],
        go_up: bool,
        safe_area: ShapelyPolygon | None = None,
) -> tuple[list[tuple[float, float]], bool]:
    path: list[tuple[float, float]] = []

    for swath_idx, swath in enumerate(swaths):
        segments_sorted = sorted(
            [s for s in swath.segments if s and len(s) >= 2],
            key=lambda s: min(pt[1] for pt in s)
        )
        if not go_up:
            segments_sorted = list(reversed(segments_sorted))

        if not segments_sorted:
            continue

        for seg_idx, segment in enumerate(segments_sorted):
            if not segment or len(segment) < 2:
                continue

            if not go_up:
                segment = list(reversed(segment))

            if path and not math.isnan(path[-1][0]):
                last_pt = path[-1]
                first_pt = segment[0]

                if not _pts_eq(last_pt, first_pt, eps=1e-6):
                    is_safe = True

                    if safe_area is not None:
                        if _segment_crosses_holes_strict(last_pt, first_pt, safe_area):
                            is_safe = False
                            log.debug("  swath[%d] seg[%d]: transition CROSSES HOLE — inserting NaN",
                                      swath_idx, seg_idx)
                        elif not _segment_inside_safe_area_strict(last_pt, first_pt, safe_area):
                            is_safe = False
                            log.debug("  swath[%d] seg[%d]: transition OUTSIDE safe area — inserting NaN",
                                      swath_idx, seg_idx)

                    if not is_safe:
                        path.append(_NAN)

            if path and not math.isnan(path[-1][0]) and _pts_eq(path[-1], segment[0], eps=1e-6):
                path.extend(segment[1:])
            else:
                path.extend(segment)

        go_up = not go_up

    return path, go_up


def _simple_boustrophedon(
        poly: ShapelyPolygon,
        safe_area: ShapelyPolygon,
        swath: float,
        start_pt: tuple[float, float],
) -> list[tuple[float, float]]:
    minx, miny, maxx, maxy = safe_area.bounds
    width = maxx - minx
    height = maxy - miny

    if width < 1e-6 or height < 1e-6:
        log.debug("_simple_boustrophedon: degenerate cell")
        return []

    min_width_threshold = swath * 0.08
    if width < min_width_threshold:
        log.debug("_simple_boustrophedon: too narrow (%.3f < %.3f)", width, min_width_threshold)
        return []

    if width < swath:
        n_swaths = 1
        log.debug("_simple_boustrophedon: narrow cell, using single swath")
    else:
        n_swaths = max(1, int(math.ceil(width / swath)))

    log.debug("_simple_boustrophedon: bounds=(%.1f,%.1f,%.1f,%.1f) swath=%.2f n=%d",
              minx, miny, maxx, maxy, swath, n_swaths)

    swaths_by_x: list[Swath] = []

    for i in range(n_swaths):
        if n_swaths == 1:
            x = (minx + maxx) / 2
        else:
            x = minx + swath / 2 + i * swath
            if i == n_swaths - 1 and x > maxx - swath / 4:
                x = maxx - swath / 4

        if x > maxx + swath / 2:
            break

        line = LineString([(x, miny - 1), (x, maxy + 1)])
        intersection = line.intersection(safe_area)

        if intersection.is_empty:
            continue

        segments = []
        if isinstance(intersection, LineString):
            coords = list(intersection.coords)
            if coords and len(coords) >= 2:
                segments.append(coords)
        elif isinstance(intersection, MultiLineString):
            for seg in intersection.geoms:
                coords = list(seg.coords)
                if coords and len(coords) >= 2:
                    segments.append(coords)

        if segments:
            swaths_by_x.append(Swath(segments))
            log.debug("  swath[%d] at x=%.2f: %d segments", i, x, len(segments))

    if not swaths_by_x:
        log.debug("_simple_boustrophedon: no swaths generated")
        return []

    def _swath_x_center(sw: Swath) -> float:
        xs = [pt[0] for seg in sw.segments for pt in seg if seg]
        return sum(xs) / len(xs) if xs else 0.0

    x_left = _swath_x_center(swaths_by_x[0])
    x_right = _swath_x_center(swaths_by_x[-1])

    if abs(start_pt[0] - x_left) <= abs(start_pt[0] - x_right):
        ordered_swaths = swaths_by_x
        direction = "LEFT→RIGHT"
    else:
        ordered_swaths = list(reversed(swaths_by_x))
        direction = "RIGHT→LEFT"

    log.debug("_simple_boustrophedon: direction=%s", direction)

    first_swath = ordered_swaths[0]
    total_pts = sum(len(seg) for seg in first_swath.segments if seg)
    if total_pts > 0:
        first_y = sum(pt[1] for seg in first_swath.segments for pt in seg if seg) / total_pts
    else:
        first_y = start_pt[1]

    go_up = start_pt[1] <= first_y
    log.debug("_simple_boustrophedon: vertical direction=%s", "UP" if go_up else "DOWN")

    path, _ = _cover_swath_list(ordered_swaths, go_up, safe_area=safe_area)

    log.debug("_simple_boustrophedon: generated %d points", len(path))
    return path


def _last_real(path: list[tuple[float, float]], fallback: tuple[float, float]) -> tuple[float, float]:
    return next((p for p in reversed(path) if not math.isnan(p[0])), fallback)


def _assemble_bcd_path(
        cells: list[Cell],
        order: list[int],
        swath: float,
        start_pt: tuple[float, float],
        safe_area: ShapelyPolygon | None = None,
) -> list[tuple[float, float]]:
    log.info("_assemble_bcd_path: assembling %d cells", len(cells))
    full_path = []
    successful_cells = 0

    for idx, cid in enumerate(order):
        cell = cells[cid]

        if idx == 0:
            entry_point = start_pt
        else:
            entry_point = _last_real(full_path, start_pt)

        log.debug("  covering cell %d (id=%d)", idx, cid)
        cell_path = _simple_boustrophedon(cell.poly, cell.poly, swath, entry_point)

        if not cell_path:
            log.debug("  cell %d: EMPTY path", cid)
            continue

        successful_cells += 1

        if idx == 0:
            full_path = cell_path
        else:
            last = _last_real(full_path, start_pt)
            first_cell_pt = next((p for p in cell_path if not math.isnan(p[0])), None)

            if first_cell_pt and last:
                needs_nan = False
                distance = math.hypot(first_cell_pt[0] - last[0], first_cell_pt[1] - last[1])

                if distance > swath * 2:
                    needs_nan = True
                    log.debug("  cell transition: LARGE_DISTANCE (%.1f > %.1f)", distance, swath * 2)
                elif safe_area is not None:
                    if _segment_crosses_holes_strict(last, first_cell_pt, safe_area):
                        needs_nan = True
                        log.debug("  cell transition: CROSSES HOLE")

                if needs_nan:
                    full_path.append(_NAN)

            full_path.extend(cell_path)

    log.info("_assemble_bcd_path: covered %d/%d cells", successful_cells, len(cells))
    return full_path


def _match_intervals(
        prev_intervals: list[tuple[float, float]],
        curr_intervals: list[tuple[float, float]],
        eps: float = 1e-6
) -> list[tuple[int, int]] | None:
    if not prev_intervals or not curr_intervals:
        return [] if len(prev_intervals) == len(curr_intervals) == 0 else None

    if len(prev_intervals) != len(curr_intervals):
        return None

    matches: list[tuple[int, int]] = []
    used_curr = set()

    for i, (py0, py1) in enumerate(prev_intervals):
        best_match = None
        best_overlap = -math.inf

        for j, (cy0, cy1) in enumerate(curr_intervals):
            if j in used_curr:
                continue
            overlap = min(py1, cy1) - max(py0, cy0)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = j

        if best_match is None or best_overlap < -eps:
            return None

        matches.append((i, best_match))
        used_curr.add(best_match)

    for i in range(len(matches) - 1):
        if matches[i][1] >= matches[i + 1][1]:
            return None

    return matches


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

        self._log_coverage_stats(raw_path, safe_area, swath)

        self._validate_coverage_path(raw_path, safe_area, start)

        return self._rotate_back(raw_path)

    def _log_coverage_stats(
            self,
            path: list[tuple[float, float]],
            safe_area: ShapelyPolygon,
            swath: float
    ) -> None:
        if not path:
            log.debug("COVERAGE STATS: path is empty")
            return

        real_points = [p for p in path if not math.isnan(p[0])]
        if len(real_points) < 2:
            log.debug("COVERAGE STATS: less than 2 real points")
            return

        try:
            path_line = LineString(real_points)

            covered_area = path_line.buffer(swath / 2, cap_style=2)

            covered_in_safe = covered_area.intersection(safe_area)

            uncovered = safe_area.difference(covered_area)

            safe_area_size = safe_area.area
            covered_size = covered_in_safe.area
            uncovered_size = uncovered.area

            if safe_area_size > 0:
                coverage_percent = (covered_size / safe_area_size) * 100
                uncovered_percent = (uncovered_size / safe_area_size) * 100
            else:
                coverage_percent = 0
                uncovered_percent = 0

            log.info("COVERAGE STATS:")
            log.info("  Safe area:     %.1f m²", safe_area_size)
            log.info("  Covered:       %.1f m² (%.1f%%)", covered_size, coverage_percent)
            log.info("  Uncovered:     %.1f m² (%.1f%%)", uncovered_size, uncovered_percent)
            log.info("  Path points:   %d", len(real_points))
            log.info("  Path length:   %.1f m", path_length(real_points))

        except Exception as e:
            log.error("COVERAGE STATS: error calculating coverage - %s", e)

    def _validate_coverage_path(
            self,
            path: list[tuple[float, float]],
            safe_area: ShapelyPolygon,
            start_pos: tuple[float, float]
    ) -> None:
        if not path:
            return

        violations = 0

        segments = []
        current_segment = []

        for pt in path:
            if math.isnan(pt[0]):
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
            else:
                current_segment.append(pt)

        if current_segment:
            segments.append(current_segment)

        for segment_idx, segment in enumerate(segments):
            if len(segment) < 2:
                continue

            for i in range(len(segment) - 1):
                p1, p2 = segment[i], segment[i + 1]

                if _segment_crosses_holes_strict(p1, p2, safe_area):
                    violations += 1
                    if violations <= 5:
                        log.debug("  PATH VIOLATION: seg[%d] (%.1f,%.1f)→(%.1f,%.1f) intersects hole",
                                  segment_idx, p1[0], p1[1], p2[0], p2[1])

        if violations > 0:
            log.warning("PATH VALIDATION: %d coverage segments intersect holes (RTH ignored)", violations)

    def _make_safe_area(self, swath: float) -> ShapelyPolygon | MultiPolygon:
        buffer_dist = swath / 2

        outer = self.poly.buffer(-buffer_dist, join_style=2, mitre_limit=5.0)

        if outer.is_empty or not isinstance(outer, (ShapelyPolygon, MultiPolygon)):
            return ShapelyPolygon()

        expanded_holes = []
        for interior in self.poly.interiors:
            hole_poly = ShapelyPolygon(interior)
            expanded = hole_poly.buffer(buffer_dist, join_style=2, mitre_limit=5.0)
            if not expanded.is_empty:
                expanded_holes.append(expanded)

        if expanded_holes:
            holes_union = unary_union(expanded_holes)
            safe = outer.difference(holes_union)
            if isinstance(safe, (ShapelyPolygon, MultiPolygon)):
                return safe
            return ShapelyPolygon()

        return outer

    def _find_critical_points_adaptive(self, safe_area: ShapelyPolygon, swath: float) -> list[float]:
        critical_x = set()
        minx, miny, maxx, maxy = safe_area.bounds

        critical_x.add(minx)
        critical_x.add(maxx)

        for interior in safe_area.interiors:
            hole = ShapelyPolygon(interior)
            hole_bounds = hole.bounds
            hole_minx, hole_maxx = hole_bounds[0], hole_bounds[2]

            critical_x.add(hole_minx - swath / 3)
            critical_x.add(hole_minx)
            critical_x.add(hole_minx + swath / 3)
            critical_x.add(hole_maxx - swath / 3)
            critical_x.add(hole_maxx)
            critical_x.add(hole_maxx + swath / 3)

        total_width = maxx - minx
        adaptive_step = max(swath / 2, total_width / 30)
        n_grid = max(10, int(math.ceil(total_width / adaptive_step)))

        for i in range(1, n_grid):
            critical_x.add(minx + i * total_width / n_grid)

        for interior in safe_area.interiors:
            hole = ShapelyPolygon(interior)
            hole_area = hole.area
            if hole_area > swath * swath * 2:
                hole_bounds = hole.bounds
                hole_width = hole_bounds[2] - hole_bounds[0]
                hole_center = (hole_bounds[0] + hole_bounds[2]) / 2

                critical_x.add(hole_center - hole_width / 4)
                critical_x.add(hole_center + hole_width / 4)

        result = sorted(critical_x)
        log.debug("_find_critical_points_adaptive: %d points", len(result))
        return result

    def _bcd_decompose(
            self,
            poly: ShapelyPolygon,
            safe_area: ShapelyPolygon,
            swath: float,
    ) -> list[Cell]:
        minx, miny, maxx, maxy = safe_area.bounds

        critical_x = self._find_critical_points_adaptive(safe_area, swath)

        cells: list[Cell] = []
        cell_id = 0
        min_cell_width = swath * 0.03

        prev_x: float | None = None
        prev_intervals: list[tuple[float, float]] = []
        current_cells: list[ShapelyPolygon | None] = []

        for x_idx, x in enumerate(critical_x):
            line = LineString([(x, miny - 10), (x, maxy + 10)])
            inter = line.intersection(safe_area)
            intervals = self._extract_y_intervals(inter)

            if x_idx % 20 == 0:
                log.debug("  x[%d]=%.1f: %d intervals", x_idx, x, len(intervals))

            if prev_x is not None and prev_intervals:
                matches = _match_intervals(prev_intervals, intervals)

                if matches is not None and len(current_cells) == len(prev_intervals):
                    new_cells = [None] * len(intervals)

                    for prev_idx, curr_idx in matches:
                        cell = current_cells[prev_idx]
                        py0, py1 = prev_intervals[prev_idx]
                        cy0, cy1 = intervals[curr_idx]

                        y_min = min(py0, cy0)
                        y_max = max(py1, cy1)

                        slice_box = box(prev_x, y_min, x, y_max)
                        slice_poly = safe_area.intersection(slice_box)

                        if not slice_poly.is_empty:
                            if cell is not None:
                                new_cells[curr_idx] = cast(
                                    ShapelyPolygon,
                                    unary_union([cell, slice_poly])
                                )
                            else:
                                new_cells[curr_idx] = cast(ShapelyPolygon, slice_poly)

                    current_cells = new_cells
                else:
                    for cell_poly in current_cells:
                        if cell_poly is None or cell_poly.area <= 1e-9:
                            continue
                        parts = list(cell_poly.geoms) if isinstance(cell_poly, MultiPolygon) else [cell_poly]
                        for part in parts:
                            if part.area <= 1e-9:
                                continue
                            bounds = part.bounds
                            cell_width = bounds[2] - bounds[0]
                            if cell_width < min_cell_width:
                                log.debug("  skipping narrow cell: width=%.3f < %.3f",
                                          cell_width, min_cell_width)
                                continue
                            cells.append(Cell(idx=cell_id, poly=part, x_min=bounds[0], x_max=bounds[2]))
                            cell_id += 1

                    current_cells = [None] * len(intervals)
            else:
                current_cells = [None] * len(intervals)

            prev_x = x
            prev_intervals = intervals

        for cell_poly in current_cells:
            if cell_poly is None or cell_poly.area <= 1e-9:
                continue
            parts = list(cell_poly.geoms) if isinstance(cell_poly, MultiPolygon) else [cell_poly]
            for part in parts:
                if part.area <= 1e-9:
                    continue
                bounds = part.bounds
                cell_width = bounds[2] - bounds[0]
                if cell_width < min_cell_width:
                    log.debug("  skipping narrow cell (final): width=%.3f < %.3f",
                              cell_width, min_cell_width)
                    continue
                cells.append(Cell(idx=cell_id, poly=part, x_min=bounds[0], x_max=bounds[2]))
                cell_id += 1

        log.info("BCD decomposition created %d cells", len(cells))
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
            sub = _simple_boustrophedon(poly, poly, swath, cur)
            if sub:
                sub_paths.append(sub)
                cur = _last_real(sub, cur)

        if not sub_paths:
            return []

        result = []
        for i, sp in enumerate(sub_paths):
            if i > 0:
                result.append(_NAN)
            result.extend(sp)

        return self._rotate_back(result)

    def _generate_for_safe_poly(
            self,
            poly: ShapelyPolygon,
            safe_poly: ShapelyPolygon,
            swath: float,
            start: tuple[float, float]
    ) -> list[tuple[float, float]]:
        has_holes = len(list(safe_poly.interiors)) > 0

        if has_holes:
            log.info("Using BCD decomposition for polygon with %d holes", len(list(safe_poly.interiors)))
            cells = self._bcd_decompose(poly, safe_poly, swath)
            if cells:
                _build_adjacency(cells)
                order = _traversal_order(cells, start)
                return _assemble_bcd_path(cells, order, swath, start, safe_poly)
            else:
                log.warning("BCD decomposition returned 0 cells, falling back to simple boustrophedon")

        log.info("Using simple boustrophedon (no holes or BCD failed)")
        return _simple_boustrophedon(poly, safe_poly, swath, start)

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


def decompose_field(
        polygon: ShapelyPolygon,
        swath: float,
        angle: float = 0.0,
) -> tuple["BoustrophedonCoverage", list[Cell], "ShapelyPolygon | MultiPolygon"]:
    bc = BoustrophedonCoverage(polygon, angle)
    safe_area = bc._make_safe_area(swath)

    if safe_area.is_empty:
        return bc, [], ShapelyPolygon()

    decomp_area: ShapelyPolygon = (
        max(safe_area.geoms, key=lambda g: g.area)
        if isinstance(safe_area, MultiPolygon)
        else safe_area
    )

    cells = bc._bcd_decompose(bc.poly, decomp_area, swath)
    _build_adjacency(cells)

    return bc, cells, safe_area


def coverage_for_cell_subset(
        bc: "BoustrophedonCoverage",
        cells: list[Cell],
        cell_indices: list[int],
        swath: float,
        start_pt: tuple[float, float],
        safe_area: "ShapelyPolygon | MultiPolygon",
) -> list[tuple[float, float]]:
    if not cell_indices:
        return []

    start_rot = rotate(Point(start_pt), -bc.angle, origin=bc.original.centroid)
    start_rotated = (start_rot.x, start_rot.y)

    assigned = [cells[i] for i in cell_indices]
    order = _traversal_order(assigned, start_rotated)

    sa: ShapelyPolygon = (
        max(safe_area.geoms, key=lambda g: g.area)
        if isinstance(safe_area, MultiPolygon)
        else safe_area
    )

    raw_path = _assemble_bcd_path(assigned, order, swath, start_rotated, sa)
    return bc._rotate_back(raw_path)


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
    current_pos = start_position
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
        rth_dist_end = math.hypot(end_pt[0] - start_position[0], end_pt[1] - start_position[1])
        rth_time_end = rth_dist_end / speed if speed > 0 else 0.0

        substance_ok = no_substance or (substance_used + seg_substance <= tank_volume)
        time_ok = no_time or (flight_time_used + seg_time + rth_time_end <= max_flight_time)

        if not substance_ok or not time_ok:
            home_dist = math.hypot(current_pos[0] - start_position[0], current_pos[1] - start_position[1])
            home_time = home_dist / speed if speed > 0 else 0.0
            can_reach_home = no_time or (flight_time_used + home_time <= max_flight_time)

            if can_reach_home and not after_rth:
                last_real = next((p for p in reversed(result) if not math.isnan(p[0])), None)

                if not result or not math.isnan(result[-1][0]):
                    result.append(_NAN)
                if last_real is None or not _pts_eq(start_position, last_real):
                    result.append(start_position)
                result.append(_NAN)
                if not _pts_eq(current_pos, start_position):
                    result.append(current_pos)
                rth_count += 1

                substance_used = 0.0
                return_dist = math.hypot(current_pos[0] - start_position[0], current_pos[1] - start_position[1])
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
