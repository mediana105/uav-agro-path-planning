from __future__ import annotations

import math

import pytest
from shapely.geometry import Polygon

from src.path_planning.bcd import (
    Cell,
    bcd_critical_x_values,
    bcd_slice_decompose,
    _traversal_order,
    _two_opt_order,
)




def rect(x0, y0, x1, y1) -> Polygon:
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def total_area(cells: list[Cell]) -> float:
    return sum(c.poly.area for c in cells)


def cells_cover(cells: list[Cell], polygon: Polygon, tol: float = 1e-6) -> bool:
    from shapely.ops import unary_union
    union = unary_union([c.poly for c in cells])
    diff = polygon.difference(union)
    return diff.area < tol




class TestBcdCriticalXValues:
    def test_empty_polygon(self):
        assert bcd_critical_x_values(Polygon()) == []

    def test_rectangle_has_two_x_values(self):
        xs = bcd_critical_x_values(rect(0, 0, 4, 3))
        assert xs[0] == pytest.approx(0.0)
        assert xs[-1] == pytest.approx(4.0)

    def test_sorted_ascending(self):
        poly = Polygon([(0, 0), (5, 0), (3, 4), (1, 4)])
        xs = bcd_critical_x_values(poly)
        assert xs == sorted(xs)

    def test_no_duplicates_within_eps(self):
        # Triangle: all x-coords are 0, 3, 5 — no near-duplicates
        poly = Polygon([(0, 0), (5, 0), (3, 4)])
        xs = bcd_critical_x_values(poly)
        for i in range(len(xs) - 1):
            assert xs[i + 1] - xs[i] > 0

    def test_polygon_with_hole_includes_hole_vertices(self):
        outer = [(0, 0), (10, 0), (10, 8), (0, 8)]
        hole = [(3, 2), (7, 2), (7, 6), (3, 6)]
        poly = Polygon(outer, [hole])
        xs = bcd_critical_x_values(poly)
        # Must contain x=3 and x=7 from the hole
        assert any(abs(x - 3.0) < 1e-6 for x in xs)
        assert any(abs(x - 7.0) < 1e-6 for x in xs)

    def test_l_shape_vertex_x_included(self):
        # L-shape: concavity at x=5
        poly = Polygon([(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)])
        xs = bcd_critical_x_values(poly)
        assert any(abs(x - 5.0) < 1e-6 for x in xs)


class TestBcdSliceDecompose:
    def test_empty_polygon_returns_empty(self):
        assert bcd_slice_decompose(Polygon(), swath=1.0) == []

    def test_rectangle_returns_one_cell(self):
        poly = rect(0, 0, 6, 4)
        cells = bcd_slice_decompose(poly, swath=1.0)
        assert len(cells) == 1
        assert cells[0].poly.area == pytest.approx(poly.area, rel=1e-3)

    def test_cells_are_indexed_from_zero(self):
        poly = rect(0, 0, 6, 4)
        cells = bcd_slice_decompose(poly, swath=1.0)
        indices = [c.idx for c in cells]
        assert indices == list(range(len(cells)))

    def test_total_area_matches_polygon(self):
        poly = Polygon([(0, 0), (10, 0), (8, 6), (2, 6)])
        cells = bcd_slice_decompose(poly, swath=1.0)
        assert total_area(cells) == pytest.approx(poly.area, rel=1e-3)

    def test_cells_cover_polygon(self):
        poly = Polygon([(0, 0), (10, 0), (8, 6), (2, 6)])
        cells = bcd_slice_decompose(poly, swath=1.0)
        assert cells_cover(cells, poly)

    def test_cells_have_valid_geometry(self):
        poly = Polygon([(0, 0), (10, 0), (10, 8), (0, 8)])
        cells = bcd_slice_decompose(poly, swath=1.0)
        for c in cells:
            assert c.poly.is_valid
            assert not c.poly.is_empty
            assert c.poly.area > 0

    def test_neighbour_ids_are_valid(self):
        poly = Polygon([(0, 0), (10, 0), (8, 6), (2, 6)])
        cells = bcd_slice_decompose(poly, swath=1.0)
        valid_ids = {c.idx for c in cells}
        for c in cells:
            for nb in c.neighbours:
                assert nb in valid_ids
                assert nb != c.idx

    def test_neighbour_relation_is_symmetric(self):
        poly = Polygon([(0, 0), (10, 0), (8, 6), (2, 6)])
        cells = bcd_slice_decompose(poly, swath=1.0)
        cell_by_idx = {c.idx: c for c in cells}
        for c in cells:
            for nb in c.neighbours:
                assert c.idx in cell_by_idx[nb].neighbours

    def test_x_min_x_max_consistent(self):
        poly = Polygon([(0, 0), (10, 0), (10, 8), (0, 8)])
        cells = bcd_slice_decompose(poly, swath=1.0)
        for c in cells:
            assert c.x_min <= c.x_max
            assert c.x_min == pytest.approx(c.poly.bounds[0], abs=1e-6)
            assert c.x_max == pytest.approx(c.poly.bounds[2], abs=1e-6)



class TestBcdConcavePolygons:
    def _l_shape(self):
        return Polygon([(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)])

    def test_l_shape_cell_count(self):
        # L-shape (10×10 square minus 5×5 top-right corner) decomposes into
        # two strips at x=5, then merge unifies them into 1 cell (simple polygon,
        # no holes). Area = 5×10 + 5×5 = 75.
        cells = bcd_slice_decompose(self._l_shape(), swath=1.0)
        assert len(cells) == 1
        assert cells[0].poly.area == pytest.approx(75.0, rel=1e-6)

    def test_l_shape_total_area(self):
        poly = self._l_shape()
        cells = bcd_slice_decompose(poly, swath=1.0)
        assert total_area(cells) == pytest.approx(poly.area, rel=1e-3)

    def test_l_shape_covers_polygon(self):
        poly = self._l_shape()
        cells = bcd_slice_decompose(poly, swath=1.0)
        assert cells_cover(cells, poly)

    def test_t_shape_total_area(self):
        # T-shape: two rectangles joined
        poly = Polygon([
            (0, 5), (10, 5), (10, 10), (0, 10),   # top bar
        ]).union(Polygon([(4, 0), (6, 0), (6, 5), (4, 5)]))   # stem
        cells = bcd_slice_decompose(poly, swath=1.0)
        assert total_area(cells) == pytest.approx(poly.area, rel=1e-3)

    def test_triangle(self):
        # Triangle strips at many x-values all merge back into 1 cell. Area = 0.5×8×6 = 24.
        poly = Polygon([(0, 0), (8, 0), (4, 6)])
        cells = bcd_slice_decompose(poly, swath=1.0)
        assert len(cells) == 1
        assert cells[0].poly.area == pytest.approx(24.0, rel=1e-6)

    def test_u_shape_covers_polygon(self):
        # U-shape (bottom 10×3 + two 3×7 arms). Strips at x=3 and x=7 merge
        # back into 1 cell because the union is a valid simple polygon. Area = 10×3 + 2×(3×7) = 72.
        outer = [(0, 0), (10, 0), (10, 10), (7, 10), (7, 3), (3, 3), (3, 10), (0, 10)]
        poly = Polygon(outer)
        cells = bcd_slice_decompose(poly, swath=1.0)
        assert len(cells) == 1
        assert cells[0].poly.area == pytest.approx(72.0, rel=1e-6)
        assert cells_cover(cells, poly)


class TestBcdWithHoles:
    def _donut(self):
        outer = [(0, 0), (12, 0), (12, 8), (0, 8)]
        hole = [(4, 2), (8, 2), (8, 6), (4, 6)]
        return Polygon(outer, [hole])

    def test_donut_total_area(self):
        poly = self._donut()
        cells = bcd_slice_decompose(poly, swath=1.0)
        assert total_area(cells) == pytest.approx(poly.area, rel=1e-3)

    def test_donut_more_than_one_cell(self):
        # Hole forces exactly 2 cells: left strip (x=0..4) and right strip (x=8..12).
        # The middle strips (x=4..8) contain the hole and cannot be merged with neighbours.
        cells = bcd_slice_decompose(self._donut(), swath=1.0)
        assert len(cells) == 2

    def test_donut_cells_valid(self):
        cells = bcd_slice_decompose(self._donut(), swath=1.0)
        for c in cells:
            assert c.poly.is_valid
            assert not list(c.poly.interiors), "cells must have no holes"

    def test_donut_covers_polygon(self):
        poly = self._donut()
        cells = bcd_slice_decompose(poly, swath=1.0)
        assert cells_cover(cells, poly)



class TestBcdSwathWidth:
    def test_large_swath_fewer_cells(self):
        poly = rect(0, 0, 20, 10)
        cells_fine = bcd_slice_decompose(poly, swath=0.5)
        cells_coarse = bcd_slice_decompose(poly, swath=5.0)
        # Coarser swath → fewer or equal number of cells
        assert len(cells_fine) >= len(cells_coarse)

    def test_swath_larger_than_polygon(self):
        # Swath wider than the polygon: should still return something (one cell)
        poly = rect(0, 0, 3, 2)
        cells = bcd_slice_decompose(poly, swath=10.0)
        assert len(cells) >= 1



class TestTraversalOrder:
    def _make_chain(self, n: int) -> list[Cell]:
        cells = []
        for i in range(n):
            poly = rect(i, 0, i + 1, 1)
            c = Cell(idx=i, poly=poly, x_min=i, x_max=i + 1)
            cells.append(c)
        for i in range(n):
            if i > 0:
                cells[i].neighbours.add(i - 1)
            if i < n - 1:
                cells[i].neighbours.add(i + 1)
        return cells

    def test_visits_all_cells(self):
        cells = self._make_chain(5)
        order = _traversal_order(cells, start_pt=(0.5, 0.5))
        assert sorted(order) == list(range(5))

    def test_single_cell(self):
        cells = [Cell(idx=0, poly=rect(0, 0, 1, 1), x_min=0, x_max=1)]
        order = _traversal_order(cells, start_pt=(0.5, 0.5))
        assert order == [0]

    def test_empty_cells(self):
        assert _traversal_order([], start_pt=(0, 0)) == []

    def test_start_pt_influences_first_cell(self):
        cells = self._make_chain(4)
        order_left = _traversal_order(cells, start_pt=(0.5, 0.5))
        order_right = _traversal_order(cells, start_pt=(3.5, 0.5))
        # Starting from the left → first cell should be index 0
        assert order_left[0] == 0
        # Starting from the right → first cell should be index 3
        assert order_right[0] == 3

    def test_no_repeated_cells(self):
        cells = self._make_chain(6)
        order = _traversal_order(cells, start_pt=(0, 0))
        assert len(order) == len(set(order))


class TestTwoOptOrder:
    def _make_cells_at(self, centers: list[tuple[float, float]]) -> list[Cell]:
        cells = []
        for i, (cx, cy) in enumerate(centers):
            poly = rect(cx - 0.5, cy - 0.5, cx + 0.5, cy + 0.5)
            cells.append(Cell(idx=i, poly=poly, x_min=cx - 0.5, x_max=cx + 0.5))
        return cells

    def test_visits_all_cells(self):
        centers = [(0, 0), (1, 0), (2, 0), (3, 0)]
        cells = self._make_cells_at(centers)
        order = _two_opt_order(cells, list(range(4)))
        assert sorted(order) == list(range(4))

    def test_already_optimal_unchanged(self):
        # Linear arrangement in order → already optimal
        centers = [(i, 0) for i in range(5)]
        cells = self._make_cells_at(centers)
        order = list(range(5))
        result = _two_opt_order(cells, order)
        assert sorted(result) == list(range(5))

    def test_improves_crossed_path(self):
        # Cells at corners; a "crossed" ordering should be un-crossed
        # 0=(0,0), 1=(2,2), 2=(2,0), 3=(0,2) — order 0,1,2,3 crosses
        centers = [(0, 0), (2, 2), (2, 0), (0, 2)]
        cells = self._make_cells_at(centers)
        bad_order = [0, 1, 2, 3]

        def tour_length(order):
            total = 0.0
            for k in range(len(order) - 1):
                ax, ay = centers[order[k]]
                bx, by = centers[order[k + 1]]
                total += math.hypot(bx - ax, by - ay)
            return total

        optimized = _two_opt_order(cells, bad_order)
        assert tour_length(optimized) <= tour_length(bad_order) + 1e-9

    def test_short_order_returned_as_is(self):
        centers = [(0, 0), (1, 0), (2, 0)]
        cells = self._make_cells_at(centers)
        order = [0, 1, 2]
        result = _two_opt_order(cells, order)
        assert sorted(result) == [0, 1, 2]