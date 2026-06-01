from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np

from shapely.errors import GEOSException
from shapely.geometry import LineString, MultiPolygon, box
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

from . import utils


@dataclass
class Cell:
    idx: int
    poly: ShapelyPolygon
    x_min: float
    x_max: float
    neighbours: set[int] = field(default_factory=set)


def _as_polygon(geom: ShapelyPolygon | MultiPolygon) -> ShapelyPolygon:
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    return geom


def _rebuild_cell_neighbours_from_geometry(
    cells: list[Cell], buf: float = 1e-5
) -> None:
    n = len(cells)
    for c in cells:
        c.neighbours.clear()
    for i in range(n):
        pi = cells[i].poly
        if pi.is_empty:
            continue
        try:
            bi = pi.buffer(buf)
        except GEOSException:
            continue
        for j in range(i + 1, n):
            pj = cells[j].poly
            if pj.is_empty:
                continue
            try:
                if bi.intersects(pj.buffer(buf)):
                    cells[i].neighbours.add(j)
                    cells[j].neighbours.add(i)
            except GEOSException:
                continue


def _interval_count_at_x(
    polygon: ShapelyPolygon,
    x: float,
    miny: float,
    maxy: float,
) -> int:
    sweep = LineString([(x, miny - 1.0), (x, maxy + 1.0)])
    try:
        inter = polygon.intersection(sweep)
    except GEOSException:
        return 0
    if inter.is_empty:
        return 0
    if inter.geom_type == "LineString":
        return 1 if inter.length > 1e-9 else 0
    if inter.geom_type == "MultiLineString":
        return sum(1 for g in inter.geoms if g.length > 1e-9)
    if inter.geom_type == "GeometryCollection":
        return sum(
            1 for g in inter.geoms if g.geom_type == "LineString" and g.length > 1e-9
        )
    return 0


def _ring_event_xs(
    ring_coords: list[tuple[float, float]],
) -> list[float]:
    n = len(ring_coords)
    out: list[float] = []
    if n < 3:
        return out
    for i in range(n):
        v = ring_coords[i]
        prev_v = ring_coords[(i - 1) % n]
        next_v = ring_coords[(i + 1) % n]
        prev_dx = prev_v[0] - v[0]
        next_dx = next_v[0] - v[0]
        vertical_attached = abs(prev_dx) < 1e-9 or abs(next_dx) < 1e-9
        both_same_side = (prev_dx < 0 and next_dx < 0) or (prev_dx > 0 and next_dx > 0)
        if vertical_attached or both_same_side:
            out.append(v[0])
    return out


def bcd_critical_x_values(
    safe_area: ShapelyPolygon | MultiPolygon,
    swath: float | None = None,
) -> list[float]:
    if safe_area.is_empty:
        return []

    safe_area = _as_polygon(safe_area)
    if safe_area.is_empty:
        return []

    minx, miny, maxx, maxy = safe_area.bounds
    candidates: set[float] = {minx, maxx}
    ring_labels: list[str] = ["exterior"] + [
        f"hole[{i}]" for i in range(len(safe_area.interiors))
    ]
    for label, ring_coords in zip(
        ring_labels,
        [list(safe_area.exterior.coords)[:-1]]
        + [list(r.coords)[:-1] for r in safe_area.interiors],
    ):
        xs = _ring_event_xs(ring_coords)
        candidates.update(xs)

    sorted_xs = sorted(candidates)
    span = max(maxx - minx, 1.0)
    min_gap = max(
        span * 1e-3,
        (swath * 0.5) if (swath is not None and swath > 0) else 0.0,
        1e-6,
    )

    merged: list[float] = [minx]
    for x in sorted_xs:
        if abs(x - minx) <= 1e-9 or abs(x - maxx) <= 1e-9:
            continue
        if x - merged[-1] <= min_gap or maxx - x <= min_gap:
            continue
        merged.append(x)
    merged.append(maxx)

    if len(merged) < 3:
        return merged

    counts = [
        _interval_count_at_x(safe_area, (merged[i] + merged[i + 1]) * 0.5, miny, maxy)
        for i in range(len(merged) - 1)
    ]

    keep: list[float] = [merged[0]]
    for i in range(1, len(merged) - 1):
        if counts[i - 1] != counts[i]:
            keep.append(merged[i])
    keep.append(merged[-1])

    return keep


def _strip_pieces(
    polygon: ShapelyPolygon,
    x_left: float,
    x_right: float,
    margin: float,
    swath: float | None = None,
) -> list[ShapelyPolygon]:
    if x_right - x_left < 1e-12:
        return []
    strip_box = box(x_left, -margin, x_right, margin)
    try:
        geom = polygon.intersection(strip_box)
    except GEOSException:
        return []
    if geom.is_empty:
        return []

    pieces: list[ShapelyPolygon] = []
    if isinstance(geom, ShapelyPolygon):
        if geom.area > 1e-9:
            pieces.append(geom)
    elif isinstance(geom, MultiPolygon):
        pieces = [
            p for p in geom.geoms if isinstance(p, ShapelyPolygon) and p.area > 1e-9
        ]
    elif hasattr(geom, "geoms"):
        for g in geom.geoms:
            if isinstance(g, ShapelyPolygon) and g.area > 1e-9:
                pieces.append(g)

    clean: list[ShapelyPolygon] = [p for p in pieces if p.area > 1e-9]

    if clean:
        union = unary_union(clean)
        if union.is_empty:
            return []
        if isinstance(union, MultiPolygon):
            clean = [
                p
                for p in union.geoms
                if isinstance(p, ShapelyPolygon) and p.area > 1e-9
            ]
        else:
            clean = [union]
    else:
        return []

    return clean


def _bcd_slice_decompose_polygon(
    safe_area: ShapelyPolygon,
    swath: float | None = None,
) -> list[Cell]:
    if safe_area.is_empty:
        return []
    critical_xs = bcd_critical_x_values(safe_area, swath=swath)
    if len(critical_xs) < 2:
        return []
    _, miny, _, maxy = safe_area.bounds
    margin = max(abs(miny), abs(maxy)) + (maxy - miny) * 0.5

    cells: list[Cell] = []
    prev_pieces: list[ShapelyPolygon] = []
    active_cells: list[int] = []

    for i in range(len(critical_xs) - 1):
        x_left = critical_xs[i]
        x_right = critical_xs[i + 1]
        pieces = _strip_pieces(safe_area, x_left, x_right, margin, swath=swath)
        if not pieces:
            prev_pieces = []
            active_cells = []
            continue

        if i == 0 or not prev_pieces:
            for pj, poly in enumerate(pieces):
                cell_idx = len(cells)
                cells.append(
                    Cell(
                        idx=cell_idx,
                        poly=poly,
                        x_min=poly.bounds[0],
                        x_max=poly.bounds[2],
                    )
                )
                active_cells.append(cell_idx)
            prev_pieces = pieces
            continue

        matches: list[list[int]] = []
        for c_poly in pieces:
            c_y0, c_y1 = c_poly.bounds[1], c_poly.bounds[3]
            overlapped: list[int] = []
            for pi, p_poly in enumerate(prev_pieces):
                p_y0, p_y1 = p_poly.bounds[1], p_poly.bounds[3]
                if min(c_y1, p_y1) - max(c_y0, p_y0) > 1e-9:
                    overlapped.append(pi)
            matches.append(overlapped)

        prev_users: list[list[int]] = [[] for _ in prev_pieces]
        for ci, prev_ids in enumerate(matches):
            for pi in prev_ids:
                prev_users[pi].append(ci)

        new_active: list[int] = [-1] * len(pieces)
        for ci, c_poly in enumerate(pieces):
            prev_ids = matches[ci]
            extend = len(prev_ids) == 1 and len(prev_users[prev_ids[0]]) == 1
            if extend:
                cell_idx = active_cells[prev_ids[0]]
                cell = cells[cell_idx]
                try:
                    cell.poly = cell.poly.union(c_poly)
                except GEOSException:
                    cell.poly = c_poly
                cell.x_min = min(cell.x_min, c_poly.bounds[0])
                cell.x_max = max(cell.x_max, c_poly.bounds[2])
                new_active[ci] = cell_idx
            else:
                cell_idx = len(cells)
                cells.append(
                    Cell(
                        idx=cell_idx,
                        poly=c_poly,
                        x_min=c_poly.bounds[0],
                        x_max=c_poly.bounds[2],
                    )
                )
                new_active[ci] = cell_idx

        for ci, prev_ids in enumerate(matches):
            cur_idx = new_active[ci]
            for pi in prev_ids:
                prev_idx = active_cells[pi]
                if cur_idx != prev_idx:
                    cells[cur_idx].neighbours.add(prev_idx)
                    cells[prev_idx].neighbours.add(cur_idx)

        prev_pieces = pieces
        active_cells = new_active

    return cells


def bcd_slice_decompose(
    safe_area: ShapelyPolygon | MultiPolygon,
    swath: float | None = None,
) -> list[Cell]:
    if safe_area.is_empty:
        return []

    if isinstance(safe_area, MultiPolygon):
        parts = [
            g
            for g in safe_area.geoms
            if isinstance(g, ShapelyPolygon) and not g.is_empty and g.area > 0
        ]
        if not parts:
            return []
        acc: list[Cell] = []
        for p in parts:
            acc.extend(_bcd_slice_decompose_polygon(p, swath))
        for i, c in enumerate(acc):
            c.idx = i
        _rebuild_cell_neighbours_from_geometry(acc)
        return acc

    if not isinstance(safe_area, ShapelyPolygon):
        return []

    return _bcd_slice_decompose_polygon(safe_area, swath)


def _traversal_order(cells: list[Cell], start_pt: tuple[float, float]) -> list[int]:
    if not cells:
        return []

    def center(c: Cell) -> tuple[float, float]:
        b = c.poly.bounds
        return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

    def d_pt(idx: int, pt: tuple[float, float]) -> float:
        cx, cy = center(cells[idx])
        return math.hypot(cx - pt[0], cy - pt[1])

    def d_cc(a: int, b: int) -> float:
        ax, ay = center(cells[a])
        bx, by = center(cells[b])
        return math.hypot(ax - bx, ay - by)

    unvisited = set(range(len(cells)))
    order = []
    stack = [min(unvisited, key=lambda p: d_pt(p, start_pt))]
    while unvisited:
        if not stack:
            ref = center(cells[order[-1]]) if order else start_pt
            stack.append(min(unvisited, key=lambda p: d_pt(p, ref)))
        v = stack.pop()
        if v not in unvisited:
            continue
        unvisited.remove(v)
        order.append(v)
        neighbours = [n for n in cells[v].neighbours if n in unvisited]
        neighbours.sort(key=lambda n: d_cc(v, n), reverse=True)
        stack.extend(neighbours)
    return order


def _two_opt_order(cells: list[Cell], order: list[int]) -> list[int]:
    if len(order) < 4:
        return list(order)

    def cx(idx: int) -> tuple[float, float]:
        b = cells[idx].poly.bounds
        return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

    def d(a: int, b: int) -> float:
        ax, ay = cx(a)
        bx, by = cx(b)
        return math.hypot(ax - bx, ay - by)

    def cost(i: int, j: int) -> float:
        return d(order[i], order[j])

    return utils.two_opt(order, cost)


def _get_vertical_strip_lines(
    cell: ShapelyPolygon, swath: float
) -> list[tuple[float, float, float]]:
    min_x, min_y, max_x, max_y = cell.bounds
    x_steps = np.arange(min_x, max_x, swath)
    if len(x_steps) == 0:
        x_steps = np.array([(min_x + max_x) * 0.5])

    lines = []
    for x in x_steps:
        vertical_line = LineString([(x, min_y - 5), (x, max_y + 5)])
        try:
            inter = cell.intersection(vertical_line)
        except GEOSException:
            continue

        if inter.is_empty:
            continue
        if inter.geom_type == "LineString":
            coords = list(inter.coords)
            lines.append((x, coords[0][1], coords[1][1]))
        elif hasattr(inter, "geoms"):
            valid_geoms = [g for g in inter.geoms if g.geom_type == "LineString"]
            if valid_geoms:
                longest = max(valid_geoms, key=lambda g: g.length)
                coords = list(longest.coords)
                lines.append((x, coords[0][1], coords[1][1]))
    return lines


def _build_snake_path(
    lines: list[tuple[float, float, float]], reverse_x: bool, flip_y: bool
) -> list[tuple[float, float]]:
    if not lines:
        return []
    working_lines = lines[::-1] if reverse_x else lines[:]
    path = []
    for i, (x, y_min, y_max) in enumerate(working_lines):
        go_top = (i % 2 == 0) if not flip_y else (i % 2 != 0)
        if go_top:
            path.append((x, y_min))
            path.append((x, y_max))
        else:
            path.append((x, y_max))
            path.append((x, y_min))
    return path


def generate_optimized_coverage_path(
    safe_area: ShapelyPolygon | MultiPolygon,
    swath: float,
    start_pt: tuple[float, float] = (0.0, 0.0),
) -> list[tuple[float, float]]:
    cells = bcd_slice_decompose(safe_area, swath=swath)
    if not cells:
        return []

    initial_order = _traversal_order(cells, start_pt)
    optimized_order = _two_opt_order(cells, initial_order)

    final_path: list[tuple[float, float]] = []
    current_pos = start_pt

    for idx in optimized_order:
        cell_poly = cells[idx].poly
        lines = _get_vertical_strip_lines(cell_poly, swath)
        if not lines:
            continue

        variants = [
            _build_snake_path(lines, reverse_x=False, flip_y=False),
            _build_snake_path(lines, reverse_x=False, flip_y=True),
            _build_snake_path(lines, reverse_x=True, flip_y=False),
            _build_snake_path(lines, reverse_x=True, flip_y=True),
        ]

        best_variant: list[tuple[float, float]] = []
        min_dist = float("inf")

        for v in variants:
            if not v:
                continue
            dist = math.hypot(v[0][0] - current_pos[0], v[0][1] - current_pos[1])
            if dist < min_dist:
                min_dist = dist
                best_variant = v

        if best_variant:
            if final_path and final_path[-1] == best_variant[0]:
                final_path.extend(best_variant[1:])
            else:
                final_path.extend(best_variant)
            current_pos = best_variant[-1]

    return final_path


def decompose_field(
    polygon: ShapelyPolygon | MultiPolygon,
    swath: float,
    angle: float = 0.0,
) -> tuple[object, list, ShapelyPolygon | MultiPolygon]:
    from shapely.affinity import rotate
    from .coverage_path import _build_spray_zone

    if polygon.is_empty or swath <= 0:
        return None, [], polygon

    centroid = polygon.centroid
    rotated = rotate(polygon, -angle, origin=centroid, use_radians=False)
    corridor = _build_spray_zone(rotated, swath)
    if corridor.is_empty:
        return None, [], polygon

    raw_cells = bcd_slice_decompose(corridor, swath=swath)

    result_cells = []
    idx_map = {cell.idx: i for i, cell in enumerate(raw_cells)}
    for i, cell in enumerate(raw_cells):
        poly_back = rotate(cell.poly, angle, origin=centroid, use_radians=False)
        result_cells.append(
            Cell(
                idx=i,
                poly=poly_back,
                x_min=poly_back.bounds[0],
                x_max=poly_back.bounds[2],
                neighbours={
                    idx_map[nb]
                    for nb in cell.neighbours
                    if nb in idx_map and idx_map[nb] != i
                },
            )
        )

    return None, result_cells, polygon
