"""
Boustrophedon Cell Decomposition (BCD) — core data structures and graph algorithms.

Contains:
  - Cell               dataclass representing a single decomposition cell
  - _traversal_order   DFS traversal of the cell adjacency graph (with distance-based neighbor ordering)
  - _match_intervals   interval-matching for sweep-line topology changes
"""

import math
from dataclasses import dataclass, field

from shapely.geometry import Polygon as ShapelyPolygon


@dataclass
class Cell:
    idx: int
    poly: ShapelyPolygon
    x_min: float
    x_max: float
    neighbours: set[int] = field(default_factory=set)  # indices in cells list


def _traversal_order(cells: list[Cell], start_pt: tuple[float, float]) -> list[int]:
    """
    Returns the order in which cells should be visited for coverage (indices).
    Uses depth-first search (DFS) with a "nearest neighbor" heuristic.
    Supports disconnected graph components.
    """
    if not cells:
        return []

    def cell_center(c: Cell) -> tuple[float, float]:
        b = c.poly.bounds
        return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

    def dist_pos_to_point(pos: int, pt: tuple[float, float]) -> float:
        cx, cy = cell_center(cells[pos])
        return math.hypot(cx - pt[0], cy - pt[1])

    def dist_pos_to_pos(a: int, b: int) -> float:
        ax, ay = cell_center(cells[a])
        bx, by = cell_center(cells[b])
        return math.hypot(ax - bx, ay - by)

    unvisited: set[int] = set(range(len(cells)))

    def pick_new_root(ref_pt: tuple[float, float]) -> int:
        return min(unvisited, key=lambda p: dist_pos_to_point(p, ref_pt))

    root = pick_new_root(start_pt)
    order: list[int] = []

    stack: list[int] = [root]

    while unvisited:
        if not stack:
            if order:
                last_center = cell_center(cells[order[-1]])
                stack.append(pick_new_root(last_center))
            else:
                stack.append(pick_new_root(start_pt))

        v = stack.pop()
        if v not in unvisited:
            continue

        unvisited.remove(v)
        order.append(v)

        neigh = [n for n in cells[v].neighbours if n in unvisited]
        neigh.sort(key=lambda n: dist_pos_to_pos(v, n), reverse=True)
        stack.extend(neigh)

    return order


def _match_intervals(
    prev_intervals: list[tuple[float, float]],
    curr_intervals: list[tuple[float, float]],
    eps: float = 1e-6,
) -> list[tuple[int, int]] | None:
    """
    Matches y-intervals between two consecutive sweep-line positions.
    Returns list of (prev_idx, curr_idx) pairs if topology unchanged,
    None if split/merge occurred (different count, no overlap, or order violation).
    """
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
