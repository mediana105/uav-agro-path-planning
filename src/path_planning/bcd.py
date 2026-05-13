from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field

from shapely.errors import GEOSException
from shapely.geometry import LineString, MultiPolygon, box
from shapely.geometry import Polygon as ShapelyPolygon

logger = logging.getLogger(__name__)


def bcd_trace_enabled() -> bool:
    return os.environ.get("BCD_TRACE", "").strip().lower() in ("1", "true", "yes")


def _bcd_trace(msg: str, *args: object) -> None:
    if bcd_trace_enabled():
        logger.info("[bcd] " + msg, *args)


@dataclass
class Cell:
    idx: int
    poly: ShapelyPolygon
    x_min: float
    x_max: float
    neighbours: set[int] = field(default_factory=set)


def _bcd_budget_per_part(num_parts: int, total: int) -> list[int]:
    if num_parts <= 0 or total < num_parts:
        return [1] * max(0, num_parts)
    base, rem = divmod(total, num_parts)
    return [base + (1 if i < rem else 0) for i in range(num_parts)]


def bcd_coalesce_neighbour_cells(
        cells: list[Cell],
        target: int,
        *,
        adj_tol: float = 5e-4,
) -> list[Cell]:
    if target < 1 or len(cells) <= target:
        return cells

    if bcd_trace_enabled():
        _bcd_trace(
            "bcd_coalesce_neighbour_cells: merging %d cells down to ≤%d "
            "(greedy smallest-area polygon union of neighbour pairs)",
            len(cells),
            target,
        )

    work: list[Cell] = [
        Cell(idx=i, poly=c.poly, x_min=c.x_min, x_max=c.x_max, neighbours=set())
        for i, c in enumerate(cells)
    ]

    def rebuild_nb() -> None:
        for c in work:
            c.neighbours.clear()
        n = len(work)
        for i in range(n):
            ai = work[i].poly
            for j in range(i + 1, n):
                if ai.distance(work[j].poly) <= adj_tol:
                    work[i].neighbours.add(j)
                    work[j].neighbours.add(i)

    rebuild_nb()

    while len(work) > target:
        cand: list[tuple[float, int, int]] = []
        for i, ci in enumerate(work):
            for j in ci.neighbours:
                if j <= i:
                    continue
                cj = work[j]
                try:
                    u = ci.poly.union(cj.poly)
                except GEOSException:
                    continue
                if u.is_empty or u.geom_type != "Polygon":
                    continue
                cand.append((u.area, i, j))
        if not cand:
            break
        cand.sort(key=lambda t: t[0])
        _, i, j = cand[0]
        if bcd_trace_enabled():
            _bcd_trace(
                "  MERGE union cells[%d] ∪ cells[%d] (smallest merged area=%.4f); "
                "remaining before delete=%d",
                i,
                j,
                cand[0][0],
                len(work),
            )
        merged = work[i].poly.union(work[j].poly)
        b = merged.bounds
        work[i].poly = merged
        work[i].x_min, work[i].x_max = b[0], b[2]
        del work[j]
        rebuild_nb()

    if bcd_trace_enabled() and len(work) > target:
        _bcd_trace(
            "bcd_coalesce: stopped with %d cells (> target %d): "
            "no neighbour pair unions into a single simple polygon",
            len(work),
            target,
        )

    for k, c in enumerate(work):
        c.idx = k
    return work


def _as_polygon(geom: ShapelyPolygon | MultiPolygon) -> ShapelyPolygon:
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    return geom


def _rebuild_cell_neighbours_from_geometry(cells: list[Cell], buf: float = 1e-5) -> None:
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
        polygon: ShapelyPolygon, x: float, miny: float, maxy: float,
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
        return sum(1 for g in inter.geoms if g.length > 1e-9)  # type: ignore[union-attr]
    if inter.geom_type == "GeometryCollection":
        return sum(
            1
            for g in inter.geoms  # type: ignore[union-attr]
            if g.geom_type == "LineString" and g.length > 1e-9
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
        both_same_side = (
            (prev_dx < 0 and next_dx < 0) or (prev_dx > 0 and next_dx > 0)
        )
        if vertical_attached or both_same_side:
            out.append(v[0])
    return out


def bcd_critical_x_values(
        safe_area: ShapelyPolygon | MultiPolygon,
        swath: float | None = None,
) -> list[float]:
    input_geom = safe_area
    if safe_area.is_empty:
        return []

    if bcd_trace_enabled() and isinstance(input_geom, MultiPolygon):
        _bcd_trace(
            "bcd_critical_x_values: input MultiPolygon with %d part(s) — "
            "critical X are computed on the largest part only (_as_polygon).",
            len(input_geom.geoms),
        )

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
        if bcd_trace_enabled() and xs:
            _bcd_trace(
                "vertex-classification: %s → %d candidate x(s): %s",
                label,
                len(xs),
                [round(t, 4) for t in sorted(xs)],
            )

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
            if bcd_trace_enabled():
                _bcd_trace(
                    "dedupe-by-min_gap: drop x=%.6f (gap to prev=%.6f, to maxx=%.6f, "
                    "min_gap=%.6f from swath=%s)",
                    x,
                    x - merged[-1],
                    maxx - x,
                    min_gap,
                    swath,
                )
            continue
        merged.append(x)
    merged.append(maxx)

    if bcd_trace_enabled():
        _bcd_trace(
            "critical-X pipeline: %d raw candidates → %d after min_gap merge "
            "(min_gap=max(span*1e-3, swath/2); span=%.3f)",
            len(sorted_xs),
            len(merged),
            span,
        )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "bcd_critical_x_values: %d candidate Xs in [%.2f, %.2f] "
            "(merged to %d at gap>%.3f); exterior verts=%d, "
            "interior rings=%d",
            len(sorted_xs), minx, maxx, len(merged), min_gap,
            len(list(safe_area.exterior.coords)) - 1,
            len(list(safe_area.interiors)),
        )

    if len(merged) < 3:
        if bcd_trace_enabled():
            _bcd_trace(
                "early exit: only %d merged x-positions (<3) → no interior "
                "strip cuts; returning %s",
                len(merged),
                [round(x, 4) for x in merged],
            )
        return merged

    counts = [
        _interval_count_at_x(
            safe_area, (merged[i] + merged[i + 1]) * 0.5, miny, maxy
        )
        for i in range(len(merged) - 1)
    ]

    keep: list[float] = [merged[0]]
    for i in range(1, len(merged) - 1):
        if counts[i - 1] != counts[i]:
            keep.append(merged[i])
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "  KEEP x=%.3f (topology %d -> %d)",
                    merged[i], counts[i - 1], counts[i],
                )
            if bcd_trace_enabled():
                _bcd_trace(
                    "topology KEEP x=%.6f as strip boundary "
                    "(vertical interval count %d → %d in adjacent strips)",
                    merged[i],
                    counts[i - 1],
                    counts[i],
                )
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "  DROP x=%.3f (topology unchanged, count=%d)",
                    merged[i], counts[i],
                )
            if bcd_trace_enabled():
                _bcd_trace(
                    "topology DROP x=%.6f (interval count unchanged: %d)",
                    merged[i],
                    counts[i],
                )
    keep.append(merged[-1])

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("  -> %d critical Xs after topology filter", len(keep))

    if bcd_trace_enabled():
        _bcd_trace(
            "final critical Xs (%d): %s",
            len(keep),
            [round(x, 4) for x in keep],
        )

    return keep


def _obstacle_y_bands_in_strip(
        polygon: ShapelyPolygon,
        strip_box: ShapelyPolygon,
        swath: float | None = None,
) -> list[tuple[float, float]]:
    strip_xmin, _, strip_xmax, _ = strip_box.bounds
    strip_width = strip_xmax - strip_xmin
    if strip_width <= 1e-9:
        return []
    span_eps = max(strip_width * 1e-6, 1e-6)
    if swath is not None and swath > 0:
        span_eps = max(span_eps, swath * 0.5)

    bands: list[tuple[float, float]] = []
    for ring in polygon.interiors:
        ring_poly = ShapelyPolygon(ring.coords)
        try:
            clipped = ring_poly.intersection(strip_box)
        except GEOSException:
            continue
        area = getattr(clipped, "area", 0.0)
        if area <= 1e-9:
            continue
        b = clipped.bounds
        if b[3] - b[1] <= 1e-9:
            continue
        if (b[0] - strip_xmin) > span_eps or (strip_xmax - b[2]) > span_eps:
            continue
        bands.append((b[1], b[3]))
    bands.sort()
    if bcd_trace_enabled() and bands:
        _bcd_trace(
            "obstacle Y-bands blocking strip x=[%.4f, %.4f]: %s "
            "(hole must span strip width within span_eps; swath=%s)",
            strip_xmin,
            strip_xmax,
            [(round(lo, 3), round(hi, 3)) for lo, hi in bands],
            swath,
        )
    return bands


def _piece_wrt_hole_y_ru(
    piece: ShapelyPolygon,
    bands: list[tuple[float, float]],
    *,
    x_left: float,
    x_right: float,
) -> str:
    if not bands:
        return (
            "пересечение коридора с полосой "
            f"x∈[{x_left:.3f},{x_right:.3f}] без Y-разреза по дыре в этой полосе"
        )
    pymin, pymax = piece.bounds[1], piece.bounds[3]
    bits: list[str] = []
    for bi, (y_lo, y_hi) in enumerate(bands):
        if pymax <= y_lo + 1e-6:
            bits.append(
                f"ниже по Y относительно проекции дыры #{bi} на полосу "
                f"(дыра по y≈[{y_lo:.2f},{y_hi:.2f}]; кусок y≈[{pymin:.2f},{pymax:.2f}])"
            )
        elif pymin >= y_hi - 1e-6:
            bits.append(
                f"выше по Y относительно проекции дыры #{bi} "
                f"(дыра y≈[{y_lo:.2f},{y_hi:.2f}]; кусок y≈[{pymin:.2f},{pymax:.2f}])"
            )
        elif pymin >= y_lo - 1e-6 and pymax <= y_hi + 1e-6:
            bits.append(
                f"внутри y-диапазона дыры #{bi} [{y_lo:.2f},{y_hi:.2f}] после "
                f"intersection+difference — обычно «карман» коридора сбоку от отверстия, "
                f"не новый разрез по x (кусок y≈[{pymin:.2f},{pymax:.2f}])"
            )
        else:
            bits.append(
                f"пересекает по Y слой дыры #{bi} [{y_lo:.2f},{y_hi:.2f}] "
                f"(кусок y≈[{pymin:.2f},{pymax:.2f}])"
            )
    return " | ".join(bits)


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

    bands = _obstacle_y_bands_in_strip(polygon, strip_box, swath=swath)

    clean: list[ShapelyPolygon] = [p for p in pieces if p.area > 1e-9]

    clean.sort(key=lambda p: p.centroid.y)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "_strip_pieces[%.2f..%.2f]: %d piece(s) y-ranges=%s (intersection only)",
            x_left, x_right, len(clean),
            [(round(p.bounds[1], 2), round(p.bounds[3], 2)) for p in clean],
        )
    if bcd_trace_enabled():
        _bcd_trace(
            "_strip_pieces x=[%.6f, %.6f]: %d polygon(s) after x-strip ∩ corridor "
            "(no Y-band split; hole y-span in strip would be %s)",
            x_left,
            x_right,
            len(clean),
            bands if bands else "none",
        )
        if len(clean) > 1:
            _bcd_trace(
                "  → несколько кусков в одной x-полосе: пересечение дало MultiPolygon "
                "или несколько компонент (не горизонтальная нарезка по дыре)."
            )
            for ki, p in enumerate(clean):
                _bcd_trace(
                    "  subpiece %d/%d y=[%.3f,%.3f] area=%.4f | %s",
                    ki + 1,
                    len(clean),
                    p.bounds[1],
                    p.bounds[3],
                    p.area,
                    _piece_wrt_hole_y_ru(p, bands, x_left=x_left, x_right=x_right),
                )
    return clean


def _bcd_slice_decompose_polygon(
        safe_area: ShapelyPolygon,
        swath: float | None = None,
        coalesce_to: int | None = None,
) -> list[Cell]:
    if safe_area.is_empty:
        return []
    critical_xs = bcd_critical_x_values(safe_area, swath=swath)
    if len(critical_xs) < 2:
        return []
    _, miny, _, maxy = safe_area.bounds
    margin = max(abs(miny), abs(maxy)) + (maxy - miny) * 0.5
    if bcd_trace_enabled():
        _bcd_trace(
            "=== BCD vertical sweep: %d critical X → %d strips; "
            "swath=%s coalesce_to=%s; polygon area=%.4f holes=%d",
            len(critical_xs),
            len(critical_xs) - 1,
            swath,
            coalesce_to,
            safe_area.area,
            len(safe_area.interiors),
        )
        _bcd_trace("critical X list: %s", [round(x, 4) for x in critical_xs])
    cells: list[Cell] = []
    prev_pieces: list[ShapelyPolygon] = []
    active_cells: list[int] = []
    for i in range(len(critical_xs) - 1):
        x_left = critical_xs[i]
        x_right = critical_xs[i + 1]
        pieces = _strip_pieces(safe_area, x_left, x_right, margin, swath=swath)
        if not pieces:
            if bcd_trace_enabled():
                _bcd_trace(
                    "strip[%d] x∈[%.4f, %.4f]: empty intersection — reset active cells",
                    i,
                    x_left,
                    x_right,
                )
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
                if bcd_trace_enabled():
                    _sb = box(x_left, -margin, x_right, margin)
                    _bh = _obstacle_y_bands_in_strip(safe_area, _sb, swath=swath)
                    _bcd_trace(
                        "strip[%d] x∈[%.4f, %.4f]: OPEN raw_cell[%d] (первая полоса) "
                        "кусок %d/%d y=[%.3f, %.3f] area=%.4f | %s",
                        i,
                        x_left,
                        x_right,
                        cell_idx,
                        pj + 1,
                        len(pieces),
                        poly.bounds[1],
                        poly.bounds[3],
                        poly.area,
                        _piece_wrt_hole_y_ru(
                            poly, _bh, x_left=x_left, x_right=x_right
                        ),
                    )
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
            extend = (
                len(prev_ids) == 1 and len(prev_users[prev_ids[0]]) == 1
            )
            if extend:
                cell_idx = active_cells[prev_ids[0]]
                cell = cells[cell_idx]
                if bcd_trace_enabled():
                    _bcd_trace(
                        "strip[%d] x∈[%.4f, %.4f]: EXTEND cell[%d] "
                        "(1↔1 y-overlap with prev piece %d) new y=[%.3f,%.3f]",
                        i,
                        x_left,
                        x_right,
                        cell_idx,
                        prev_ids[0],
                        c_poly.bounds[1],
                        c_poly.bounds[3],
                    )
                try:
                    cell.poly = cell.poly.union(c_poly)
                except GEOSException:
                    cell.poly = c_poly
                cell.x_min = min(cell.x_min, c_poly.bounds[0])
                cell.x_max = max(cell.x_max, c_poly.bounds[2])
                new_active[ci] = cell_idx
            else:
                cell_idx = len(cells)
                if bcd_trace_enabled():
                    _sb = box(x_left, -margin, x_right, margin)
                    _bh = _obstacle_y_bands_in_strip(safe_area, _sb, swath=swath)
                    _bcd_trace(
                        "strip[%d] x∈[%.4f, %.4f]: OPEN raw_cell[%d] "
                        "кусок %d/%d полосы — НЕ новый critical X; "
                        "prev_ids=%s users_of_prev=%s | y=[%.3f,%.3f] area=%.4f | %s",
                        i,
                        x_left,
                        x_right,
                        cell_idx,
                        ci + 1,
                        len(pieces),
                        list(prev_ids),
                        [len(prev_users[pid]) for pid in prev_ids]
                        if prev_ids
                        else [],
                        c_poly.bounds[1],
                        c_poly.bounds[3],
                        c_poly.area,
                        _piece_wrt_hole_y_ru(
                            c_poly, _bh, x_left=x_left, x_right=x_right
                        ),
                    )
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
                    if bcd_trace_enabled():
                        _bcd_trace(
                            "strip[%d]: adjacency cell[%d] ↔ cell[%d] "
                            "(y-overlap across strip boundary)",
                            i,
                            cur_idx,
                            prev_idx,
                        )
                    cells[cur_idx].neighbours.add(prev_idx)
                    cells[prev_idx].neighbours.add(cur_idx)

        prev_pieces = pieces
        active_cells = new_active

    if bcd_trace_enabled():
        _bcd_trace(
            "=== sweep done: %d raw cell(s) before coalesce (coalesce_to=%s)",
            len(cells),
            coalesce_to,
        )

    if coalesce_to is not None and len(cells) > coalesce_to:
        if bcd_trace_enabled():
            _bcd_trace(
                "applying bcd_coalesce_neighbour_cells: %d → %d",
                len(cells),
                coalesce_to,
            )
        cells = bcd_coalesce_neighbour_cells(cells, coalesce_to)

    if bcd_trace_enabled():
        _bcd_trace(
            "Итог для PNG bcd_decomposition: подпись «id:ci» — индекс ci в "
            "финальном списке (после coalesce), не тот же номер, что «raw_cell[…]» "
            "в логах выше."
        )
        for c in cells:
            b = c.poly.bounds
            _bcd_trace(
                "final_cell[%d] y∈[%.3f,%.3f] x∈[%.3f,%.3f] area=%.4f",
                c.idx,
                b[1],
                b[3],
                b[0],
                b[2],
                c.poly.area,
            )
    return cells


def bcd_slice_decompose(
        safe_area: ShapelyPolygon | MultiPolygon,
        swath: float | None = None,
        coalesce_to: int | None = None,
) -> list[Cell]:
    if safe_area.is_empty:
        return []

    if isinstance(safe_area, MultiPolygon):
        if bcd_trace_enabled():
            _bcd_trace(
                "bcd_slice_decompose: MultiPolygon with %d part(s) — "
                "decompose each part, then rebuild cross-part adjacency",
                len(safe_area.geoms),
            )
        parts = [
            g
            for g in safe_area.geoms
            if isinstance(g, ShapelyPolygon) and not g.is_empty and g.area > 0
        ]
        if not parts:
            return []
        acc: list[Cell] = []
        for p in parts:
            acc.extend(_bcd_slice_decompose_polygon(p, swath, coalesce_to=None))
        for i, c in enumerate(acc):
            c.idx = i
        _rebuild_cell_neighbours_from_geometry(acc)
        if coalesce_to is not None and len(acc) > coalesce_to:
            acc = bcd_coalesce_neighbour_cells(acc, coalesce_to)
        return acc

    if not isinstance(safe_area, ShapelyPolygon):
        return []

    return _bcd_slice_decompose_polygon(safe_area, swath, coalesce_to)


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
                    best[i:j+1] = best[i:j+1][::-1]
                    improved = True
                    break
            if improved:
                break
    return best