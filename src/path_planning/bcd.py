from __future__ import annotations

import math
from dataclasses import dataclass, field

from shapely.errors import GEOSException
from shapely.geometry import LineString, MultiPolygon, box
from shapely.geometry import Polygon as ShapelyPolygon


@dataclass
class Cell:
    idx: int
    poly: ShapelyPolygon
    x_min: float
    x_max: float
    neighbours: set[int] = field(default_factory=set)


def _is_y_monotone(polygon: ShapelyPolygon) -> bool:
    coords = list(polygon.exterior.coords)[:-1]
    vertex_ys = sorted({c[1] for c in coords})
    if len(vertex_ys) < 2:
        return True

    minx, miny, maxx, maxy = polygon.bounds
    margin = (maxx - minx) + 1.0

    for y in vertex_ys[1:-1]:  # extremes never cause a split
        for dy in (-1e-9, 1e-9):
            test_y = y + dy
            if not (miny < test_y < maxy):
                continue
            sweep = LineString([(minx - margin, test_y), (maxx + margin, test_y)])
            try:
                inter = polygon.intersection(sweep)
            except GEOSException:
                continue
            if inter.is_empty:
                continue
            if inter.geom_type == "MultiLineString":
                if sum(1 for g in inter.geoms if g.length > 1e-9) > 1:  # type: ignore[union-attr]
                    return False
            elif inter.geom_type == "GeometryCollection":
                if sum(1 for g in inter.geoms  # type: ignore[union-attr]
                       if g.geom_type == "LineString" and g.length > 1e-9) > 1:
                    return False
    return True


def _as_polygon(geom: ShapelyPolygon | MultiPolygon) -> ShapelyPolygon:
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    return geom


def bcd_critical_x_values(
        safe_area: ShapelyPolygon | MultiPolygon,
        merge_eps: float | None = None,
) -> list[float]:
    safe_area = _as_polygon(safe_area)
    if safe_area.is_empty:
        return []

    minx, _miny, maxx, _maxy = safe_area.bounds
    if merge_eps is None:
        merge_eps = max(1e-7, (maxx - minx) * 1e-9)

    xs: list[float] = [minx, maxx]
    for x, _ in safe_area.exterior.coords:
        xs.append(float(x))
    for ring in safe_area.interiors:
        for x, _ in ring.coords:
            xs.append(float(x))

    xs.sort()

    merged: list[float] = []
    for x in xs:
        if not merged or x - merged[-1] > merge_eps:
            merged.append(x)
    return merged


def _strip_pieces(
        polygon: ShapelyPolygon,
        x_left: float,
        x_right: float,
        miny: float,
        maxy: float,
        margin: float,
) -> list[ShapelyPolygon]:
    if x_right - x_left < 1e-12:
        return []

    strip_box = box(x_left, miny - margin, x_right, maxy + margin)
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
    clean: list[ShapelyPolygon] = []
    for p in pieces:
        if list(p.interiors):
            outer = ShapelyPolygon(p.exterior.coords)
            if outer.area > 1e-9:
                clean.append(outer)
        else:
            clean.append(p)

    clean.sort(key=lambda p: p.centroid.y)
    return clean


def bcd_slice_decompose(
        safe_area: ShapelyPolygon | MultiPolygon, swath: float
) -> list[Cell]:
    safe_area = _as_polygon(safe_area)
    if safe_area.is_empty:
        return []

    minx, miny, maxx, maxy = safe_area.bounds
    margin = max((maxy - miny) * 0.1, swath, 10.0)
    critical_xs = bcd_critical_x_values(safe_area, merge_eps=swath * 1e-3)
    if len(critical_xs) < 2:
        return []

    cells: list[Cell] = []
    prev_strip: list[tuple[int, ShapelyPolygon]] = []

    for i in range(len(critical_xs) - 1):
        x_left = critical_xs[i]
        x_right = critical_xs[i + 1]

        pieces = _strip_pieces(safe_area, x_left, x_right, miny, maxy, margin)
        if not pieces:
            prev_strip = []
            continue

        curr_strip: list[tuple[int, ShapelyPolygon]] = []

        for poly in pieces:
            cell_idx = len(cells)
            cells.append(
                Cell(
                    idx=cell_idx,
                    poly=poly,
                    x_min=poly.bounds[0],
                    x_max=poly.bounds[2],
                )
            )
            curr_strip.append((cell_idx, poly))
        for c_idx, c_poly in curr_strip:
            c_y0, c_y1 = c_poly.bounds[1], c_poly.bounds[3]
            for p_idx, p_poly in prev_strip:
                p_y0, p_y1 = p_poly.bounds[1], p_poly.bounds[3]
                if min(c_y1, p_y1) - max(c_y0, p_y0) > 1e-9:
                    cells[c_idx].neighbours.add(p_idx)
                    cells[p_idx].neighbours.add(c_idx)

        prev_strip = curr_strip

    return _merge_cells(cells)


def _merge_cells(cells: list[Cell]) -> list[Cell]:
    if not cells:
        return cells

    active: dict[int, Cell] = {c.idx: c for c in cells}

    changed = True
    while changed:
        changed = False

        x_boundaries: set[float] = {c.x_max for c in active.values()}

        for x_b in sorted(x_boundaries):
            left_ids = [i for i, c in active.items() if abs(c.x_max - x_b) < 1e-6]
            right_ids = [i for i, c in active.items() if abs(c.x_min - x_b) < 1e-6]

            if not left_ids or not right_ids:
                continue

            adj: dict[int, list[int]] = {}
            for a in left_ids:
                for b in right_ids:
                    if b in active[a].neighbours:
                        adj.setdefault(a, []).append(b)
                        adj.setdefault(b, []).append(a)

            if not adj:
                continue

            visited: set[int] = set()
            components: list[tuple[frozenset[int], frozenset[int]]] = []

            for start in adj:
                if start in visited:
                    continue
                component: set[int] = set()
                queue = [start]
                while queue:
                    node = queue.pop()
                    if node in visited:
                        continue
                    visited.add(node)
                    component.add(node)
                    for nb in adj.get(node, []):
                        if nb not in visited:
                            queue.append(nb)

                l_set = frozenset(component & set(left_ids))
                r_set = frozenset(component & set(right_ids))
                if l_set and r_set:
                    components.append((l_set, r_set))

            merged_any = False
            for l_set, r_set in components:
                all_ids = l_set | r_set
                if len(all_ids) < 2:
                    continue
                if len(l_set) < len(r_set):
                    continue
                try:
                    it = iter(all_ids)
                    union = active[next(it)].poly
                    for idx in it:
                        union = union.union(active[idx].poly)
                except GEOSException:
                    continue

                if not (
                        isinstance(union, ShapelyPolygon)
                        and union.is_valid
                        and not list(union.interiors)
                        and union.area > 1e-9
                        and _is_y_monotone(union)
                ):
                    continue
                keep_idx = min(l_set)
                absorb = all_ids - {keep_idx}

                keep_cell = active[keep_idx]
                keep_cell.poly = union
                keep_cell.x_min = min(active[i].x_min for i in l_set)
                keep_cell.x_max = max(active[i].x_max for i in r_set)

                new_neighbors: set[int] = set()
                for idx in all_ids:
                    new_neighbors |= active[idx].neighbours
                new_neighbors -= all_ids
                keep_cell.neighbours = new_neighbors

                for n_idx in new_neighbors:
                    if n_idx in active:
                        for abs_idx in absorb:
                            active[n_idx].neighbours.discard(abs_idx)
                        active[n_idx].neighbours.add(keep_idx)

                for abs_idx in absorb:
                    del active[abs_idx]

                merged_any = True

            if merged_any:
                changed = True
                break

    result = sorted(active.values(), key=lambda c: c.x_min)
    idx_map = {c.idx: new_idx for new_idx, c in enumerate(result)}
    for c in result:
        c.idx = idx_map[c.idx]
        c.neighbours = {idx_map[n] for n in c.neighbours if n in idx_map}

    return result


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

    unvisited: set[int] = set(range(len(cells)))
    order: list[int] = []
    stack: list[int] = [min(unvisited, key=lambda p: d_pt(p, start_pt))]

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

    best = list(order)
    n = len(best)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                before = d(best[i - 1], best[i])
                if j + 1 < n:
                    before += d(best[j], best[j + 1])
                after = d(best[i - 1], best[j])
                if j + 1 < n:
                    after += d(best[i], best[j + 1])
                if after < before - 1e-9:
                    best[i: j + 1] = best[i: j + 1][::-1]
                    improved = True
                    break
            if improved:
                break
    return best
