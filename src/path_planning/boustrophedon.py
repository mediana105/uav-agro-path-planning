import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Union

from shapely.affinity import rotate
from shapely.geometry import (
    LineString,
    Point,
    Polygon as ShapelyPolygon,
    MultiPolygon,
    box,
)
from shapely.ops import unary_union, nearest_points

_NAN = (float("nan"), float("nan"))


def _pts_eq(a: Tuple[float, ...], b: Tuple[float, ...], eps: float = 1e-9) -> bool:
    return math.hypot(a[0] - b[0], a[1] - b[1]) < eps


@dataclass
class Cell:
    idx: int
    poly: ShapelyPolygon
    neighbours: Set[int] = field(default_factory=set)


def _build_adjacency(cells: List[Cell]) -> None:
    n = len(cells)
    for i in range(n):
        for j in range(i + 1, n):
            if cells[i].poly.touches(cells[j].poly):
                cells[i].neighbours.add(j)
                cells[j].neighbours.add(i)


def _traversal_order(cells: List[Cell], start_pt: Tuple[float, float]) -> List[int]:
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
    visited: Set[int] = set()
    order: List[int] = []

    def dfs(u: int):
        visited.add(u)
        order.append(u)
        for v in sorted(cells[u].neighbours):
            if v not in visited:
                dfs(v)

    dfs(start_idx)
    for i in range(len(cells)):
        if i not in visited:
            order.append(i)
    return order


class BoustrophedonCoverage:
    def __init__(self, poly: ShapelyPolygon, angle: float = 0.0):
        self.original = poly
        self.angle = angle
        self.poly = rotate(poly, -angle, origin="centroid")

    def generate_coverage_path(
            self, swath: float, start: Tuple[float, float]
    ) -> List[Tuple[float, float]]:
        safe_area = self._make_safe_area(swath)
        if isinstance(safe_area, MultiPolygon):
            return self._handle_multipolygon(safe_area, swath, start)
        start_rot = rotate(Point(start), -self.angle, origin=self.original.centroid)
        start_pt = (start_rot.x, start_rot.y)
        cells = self._decompose_safe(safe_area, swath)
        if not cells:
            return []
        _build_adjacency(cells)
        order = _traversal_order(cells, start_pt)
        raw_path = self._assemble_path(cells, order, swath, safe_area, start_pt)
        return self._rotate_back(raw_path)

    def _make_safe_area(self, swath: float) -> Union[ShapelyPolygon, MultiPolygon]:
        outer = self.poly.buffer(-swath / 2)
        if outer.is_empty:
            return outer
        expanded_holes = [
            ShapelyPolygon(inter).buffer(swath / 2)
            for inter in self.poly.interiors
        ]
        if expanded_holes:
            holes_union = unary_union(expanded_holes)
            safe = outer.difference(holes_union)
            return safe if isinstance(safe, (ShapelyPolygon, MultiPolygon)) else MultiPolygon([safe])
        return outer

    def _handle_multipolygon(
            self,
            mp: MultiPolygon,
            swath: float,
            start: Tuple[float, float],
    ) -> List[Tuple[float, float]]:
        s_rot = rotate(Point(start), -self.angle, origin=self.original.centroid)
        cur = (s_rot.x, s_rot.y)
        sub_paths: List[List[Tuple[float, float]]] = []
        for poly in mp.geoms:
            sub = self._generate_for_safe(poly, swath, (poly.centroid.x, poly.centroid.y))
            if sub:
                sub_paths.append(sub)
        ordered: List[List[Tuple[float, float]]] = []
        remaining = list(range(len(sub_paths)))
        while remaining:
            best = min(
                remaining,
                key=lambda i: min(
                    math.hypot(sub_paths[i][0][0] - cur[0], sub_paths[i][0][1] - cur[1]),
                    math.hypot(sub_paths[i][-1][0] - cur[0], sub_paths[i][-1][1] - cur[1]),
                ),
            )
            seg = sub_paths[best]
            if math.hypot(seg[-1][0] - cur[0], seg[-1][1] - cur[1]) < math.hypot(
                    seg[0][0] - cur[0], seg[0][1] - cur[1]
            ):
                seg = list(reversed(seg))
            ordered.append(seg)
            cur = seg[-1]
            remaining.remove(best)
        full: List[Tuple[float, float]] = []
        for i, seg in enumerate(ordered):
            if i:
                full.append(_NAN)
            full.extend(seg)
        return self._rotate_back(full)

    def _generate_for_safe(
            self,
            work: Union[ShapelyPolygon, MultiPolygon],
            swath: float,
            start: Tuple[float, float],
    ) -> List[Tuple[float, float]]:
        if isinstance(work, MultiPolygon):
            return self._handle_multipolygon(work, swath, start)
        start_pt = start
        cells = self._decompose_safe(work, swath)
        if not cells:
            return []
        _build_adjacency(cells)
        order = _traversal_order(cells, start_pt)
        return self._assemble_path(cells, order, swath, work, start_pt)

    def _decompose_safe(self, work: Union[ShapelyPolygon, MultiPolygon], swath: float) -> List[Cell]:
        xs: Set[float] = set()
        if isinstance(work, ShapelyPolygon):
            polys = [work]
        else:
            polys = list(work.geoms)
        for poly in polys:
            for x, _ in poly.exterior.coords:
                xs.add(x)
            for interior in poly.interiors:
                for x, _ in interior.coords:
                    xs.add(x)
        all_bounds = [p.bounds for p in polys]
        minx = min(b[0] for b in all_bounds)
        maxx = max(b[2] for b in all_bounds)
        width = maxx - minx
        n_cols = max(1, int(round(width / swath)))
        offset = (width - (n_cols - 1) * swath) / 2.0
        col_xs = [minx + offset + i * swath for i in range(n_cols)]
        slices: List[Tuple[float, List[Tuple[float, float]]]] = []
        for x in col_xs:
            line = LineString([(x, -1e9), (x, 1e9)])
            inter = line.intersection(work)
            intervals = self._y_intervals(inter)
            slices.append((x, intervals))
        cells: List[Cell] = []
        cell_id = 0
        open_cells: List[Optional[Cell]] = []
        prev_x, prev_ints = slices[0]
        for y_low, y_high in prev_ints:
            rect = box(prev_x, y_low, prev_x, y_high)
            poly = rect.intersection(work)
            open_cells.append(Cell(cell_id, poly))
            cell_id += 1
        for cur_x, cur_ints in slices[1:]:
            if len(cur_ints) == len(prev_ints):
                for i, (y_low, y_high) in enumerate(cur_ints):
                    rect = box(prev_x, y_low, cur_x, y_high)
                    poly = rect.intersection(work)
                    if not poly.is_empty:
                        open_cells[i].poly = unary_union([open_cells[i].poly, poly])
            else:
                cells.extend([c for c in open_cells if c and not c.poly.is_empty])
                open_cells = []
                for y_low, y_high in cur_ints:
                    rect = box(prev_x, y_low, cur_x, y_high)
                    poly = rect.intersection(work)
                    open_cells.append(Cell(cell_id, poly))
                    cell_id += 1
            prev_x, prev_ints = cur_x, cur_ints
        cells.extend([c for c in open_cells if c and not c.poly.is_empty])
        return cells

    @staticmethod
    def _y_intervals(geom) -> List[Tuple[float, float]]:
        intervals: List[Tuple[float, float]] = []
        if geom.is_empty:
            return intervals
        if geom.geom_type == "LineString":
            ys = sorted(pt[1] for pt in geom.coords)
            intervals.append((ys[0], ys[-1]))
        elif geom.geom_type in ("MultiLineString", "GeometryCollection"):
            for part in geom.geoms:
                intervals.extend(BoustrophedonCoverage._y_intervals(part))
        intervals.sort(key=lambda s: s[0])
        return intervals

    def _assemble_path(
            self,
            cells: List[Cell],
            order: List[int],
            swath: float,
            work: Union[ShapelyPolygon, MultiPolygon],
            global_start: Tuple[float, float],
    ) -> List[Tuple[float, float]]:
        full_path: List[Tuple[float, float]] = []
        for step, cid in enumerate(order):
            cell = cells[cid]
            if step == 0:
                entry_pt = global_start
            else:
                last_valid = next(
                    (p for p in reversed(full_path) if not math.isnan(p[0])), global_start
                )
                entry_pt = self._closest_point_on_boundary(cell.poly, last_valid)
            snake = self._snake_with_hole_detours(cell.poly, swath, entry_pt)
            if not snake:
                continue
            if not full_path:
                full_path = list(snake)
            else:
                conn = self._orthogonal_connect(full_path[-1], snake[0], work)
                full_path.extend(conn[1:])
                full_path.extend(snake[1:])
        return full_path

    def _snake_with_hole_detours(
            self,
            poly: ShapelyPolygon,
            swath: float,
            start_pt: Tuple[float, float],
    ) -> List[Tuple[float, float]]:
        minx, miny, maxx, maxy = poly.bounds
        width = maxx - minx
        n_cols = max(1, int(round(width / swath)))
        offset = (width - (n_cols - 1) * swath) / 2.0
        col_positions = [minx + offset + i * swath for i in range(n_cols)]
        columns: List[Tuple[float, List[Tuple[float, float]]]] = []
        for x in col_positions:
            line = LineString([(x, miny - 1), (x, maxy + 1)])
            inter = line.intersection(poly)
            if not inter.is_empty:
                segs = self._extract_segments(inter)
                if segs:
                    columns.append((x, segs))
        if not columns:
            return []
        go_up = start_pt[1] <= (miny + maxy) / 2
        path: List[Tuple[float, float]] = []
        for col_x, segs in columns:
            vertical_parts = [[(col_x, y_lo), (col_x, y_hi)] for y_lo, y_hi in segs]
            if not go_up:
                vertical_parts = [list(reversed(p)) for p in reversed(vertical_parts)]
            if path:
                # horizontal transition to next column — direct connection
                if not _pts_eq(path[-1], vertical_parts[0][0]):
                    path.append(vertical_parts[0][0])
            for idx_part, part in enumerate(vertical_parts):
                if idx_part > 0:
                    # segments in the same column separated by a hole — teleport
                    path.append(_NAN)
                    path.append(part[0])
                path.extend(part)
            go_up = not go_up
        return path

    @staticmethod
    def _extract_segments(geom) -> List[Tuple[float, float]]:
        segs = []
        if geom.geom_type == "LineString":
            ys = sorted(c[1] for c in geom.coords)
            segs.append((ys[0], ys[-1]))
        elif geom.geom_type in ("MultiLineString", "GeometryCollection"):
            for g in geom.geoms:
                segs.extend(BoustrophedonCoverage._extract_segments(g))
        segs.sort(key=lambda s: s[0])
        return segs

    @staticmethod
    def _closest_point_on_boundary(poly: ShapelyPolygon,
                                   pt: Tuple[float, float]) -> Tuple[float, float]:
        boundary = poly.boundary
        if boundary is None or boundary.is_empty:
            return poly.centroid.x, poly.centroid.y
        p = Point(pt)
        nearest, _ = nearest_points(boundary, p)
        return nearest.x, nearest.y

    @staticmethod
    def _orthogonal_connect(
            p1: Tuple[float, float],
            p2: Tuple[float, float],
            work: Union[ShapelyPolygon, MultiPolygon],
    ) -> List[Tuple[float, float]]:
        vert = LineString([p1, (p1[0], p2[1])])
        horiz = LineString([(p1[0], p2[1]), p2])
        if work.covers(vert) and work.covers(horiz):
            return [p1, (p1[0], p2[1]), p2]
        horiz2 = LineString([p1, (p2[0], p1[1])])
        vert2 = LineString([(p2[0], p1[1]), p2])
        if work.covers(horiz2) and work.covers(vert2):
            return [p1, (p2[0], p1[1]), p2]
        for ring in [work.exterior] + list(work.interiors):
            if isinstance(ring, (list, tuple)):
                continue
            try:
                idx1 = next(i for i, pt in enumerate(ring.coords) if _pts_eq(pt, p1))
                idx2 = next(i for i, pt in enumerate(ring.coords) if _pts_eq(pt, p2))
            except StopIteration:
                continue
            seg1 = list(ring.coords[idx1:idx2 + 1])
            seg2 = list(ring.coords[idx2:] + ring.coords[:idx1 + 1])
            if work.covers(LineString(seg1)):
                return [(pt[0], pt[1]) for pt in seg1]
            if work.covers(LineString(seg2)):
                return [(pt[0], pt[1]) for pt in seg2]
        return [p1, p2]

    def _rotate_back(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not path:
            return path
        cen = self.original.centroid
        rad = math.radians(self.angle)
        ca, sa = math.cos(rad), math.sin(rad)
        out: List[Tuple[float, float]] = []
        for x, y in path:
            if math.isnan(x):
                out.append(_NAN)
                continue
            dx, dy = x - cen.x, y - cen.y
            out.append((cen.x + ca * dx - sa * dy, cen.y + sa * dx + ca * dy))
        return out

    @staticmethod
    def path_time(
            path: List[Tuple[float, float]],
            velocity: float = 5.0,
            turn_time: float = 1.0,
            off_time: float = 2.0,
    ) -> float:
        total = 0.0
        prev_dir = None
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            if math.isnan(x1) or math.isnan(x2):
                total += off_time
                prev_dir = None
                continue
            seg_len = math.hypot(x2 - x1, y2 - y1)
            total += seg_len / velocity
            cur_dir = (x2 - x1, y2 - y1)
            if prev_dir:
                dot = prev_dir[0] * cur_dir[0] + prev_dir[1] * cur_dir[1]
                norm1 = math.hypot(*prev_dir)
                norm2 = math.hypot(*cur_dir)
                if norm1 > 0 and norm2 > 0:
                    cos_angle = max(-1.0, min(1.0, dot / (norm1 * norm2)))
                    angle = math.degrees(math.acos(cos_angle))
                    if angle >= 45.0:
                        total += turn_time
            prev_dir = cur_dir
        return total


def generate_boustrophedon_coverage(
        polygon: ShapelyPolygon,
        swath: float,
        angle: float = 0.0,
        start_position: Optional[Tuple[float, float]] = None,
) -> List[Tuple[float, float]]:
    if start_position is None:
        start_position = (polygon.centroid.x, polygon.centroid.y)
    return BoustrophedonCoverage(polygon, angle).generate_coverage_path(swath, start_position)


def generate_snake_simple(
        polygon: ShapelyPolygon,
        angle: float,
        swath: float,
        start_position: Optional[Tuple[float, float]] = None,
) -> List[Tuple[float, float]]:
    return generate_boustrophedon_coverage(polygon, swath, angle, start_position)


def path_length(path: List[Tuple[float, float]]) -> float:
    total = 0.0
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        if math.isnan(x1) or math.isnan(x2):
            continue
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def path_time(
        path: List[Tuple[float, float]],
        velocity: float = 5.0,
        turn_time: float = 1.0,
        off_time: float = 2.0,
) -> float:
    return BoustrophedonCoverage.path_time(path, velocity, turn_time, off_time)
