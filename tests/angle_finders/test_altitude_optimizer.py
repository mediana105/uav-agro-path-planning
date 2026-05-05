import math

import pytest
from shapely.affinity import rotate as shapely_rotate
from shapely.geometry import Polygon

from src.angle_finders.altitude_optimizer import (
    find_optimal_angle,
    get_general_altitude,
    minimum_altitude,
    num_parallel_passes,
)


def make_rect(w: float, h: float, cx: float = 0.0, cy: float = 0.0) -> Polygon:
    hw, hh = w / 2, h / 2
    return Polygon(
        [
            (cx - hw, cy - hh),
            (cx + hw, cy - hh),
            (cx + hw, cy + hh),
            (cx - hw, cy + hh),
        ]
    )


def make_tilted_rect(w: float, h: float, angle_deg: float) -> Polygon:
    rect = make_rect(w, h)
    return shapely_rotate(rect, angle_deg, origin=rect.centroid, use_radians=False)


def _rect_expected_altitude(w: float, h: float, theta: float) -> float:
    return w * abs(math.sin(theta)) + h * abs(math.cos(theta))


def _altitude_is_local_minimum(
    polygon: Polygon, theta: float, n_probes: int = 36
) -> bool:
    base = get_general_altitude(polygon, theta)
    for k in range(1, n_probes + 1):
        delta = k * math.pi / 180
        for t in (theta - delta, theta + delta):
            if get_general_altitude(polygon, t % math.pi) < base - 1e-6:
                return False
    return True


class TestGetGeneralAltitude:
    @pytest.mark.parametrize(
        "theta", [0.0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2]
    )
    def test_rect_4x2_matches_formula(self, theta):
        poly = make_rect(4.0, 2.0)
        expected = _rect_expected_altitude(4.0, 2.0, theta)
        assert get_general_altitude(poly, theta) == pytest.approx(expected, abs=1e-6)

    def test_rect_theta0_equals_height(self):
        poly = make_rect(6.0, 3.0)
        assert get_general_altitude(poly, 0.0) == pytest.approx(3.0, abs=1e-6)

    def test_rect_theta90_equals_width(self):
        poly = make_rect(6.0, 3.0)
        assert get_general_altitude(poly, math.pi / 2) == pytest.approx(6.0, abs=1e-6)

    def test_square_theta0_equals_theta90(self):
        poly = make_rect(5.0, 5.0)
        alt0 = get_general_altitude(poly, 0.0)
        alt90 = get_general_altitude(poly, math.pi / 2)
        assert alt0 == pytest.approx(alt90, abs=1e-6)

    def test_altitude_nonnegative(self):
        poly = make_rect(3.0, 7.0)
        for deg in range(0, 180, 10):
            assert get_general_altitude(poly, math.radians(deg)) >= 0.0

    def test_right_triangle(self):
        tri = Polygon([(0, 0), (4, 0), (0, 3)])
        assert get_general_altitude(tri, 0.0) == pytest.approx(3.0, abs=1e-6)
        assert get_general_altitude(tri, math.pi / 2) == pytest.approx(4.0, abs=1e-6)

    def test_shifted_polygon_same_altitude(self):
        poly1 = make_rect(4.0, 2.0, cx=0.0, cy=0.0)
        poly2 = make_rect(4.0, 2.0, cx=100.0, cy=50.0)
        for theta in [0.0, math.pi / 4, math.pi / 2]:
            assert get_general_altitude(poly1, theta) == pytest.approx(
                get_general_altitude(poly2, theta), abs=1e-6
            )


class TestMinimumAltitude:
    def test_wide_rect_min_altitude_value(self):
        poly = make_rect(4.0, 2.0)
        min_alt, _ = minimum_altitude(poly)
        assert min_alt == pytest.approx(2.0, abs=1e-6)

    def test_wide_rect_optimal_angle_is_zero(self):
        poly = make_rect(4.0, 2.0)
        _, best_angle = minimum_altitude(poly)
        assert math.sin(best_angle) == pytest.approx(0.0, abs=1e-6)

    def test_tall_rect_min_altitude_value(self):
        poly = make_rect(2.0, 6.0)
        min_alt, _ = minimum_altitude(poly)
        assert min_alt == pytest.approx(2.0, abs=1e-6)

    def test_tall_rect_optimal_angle_is_pi_over_2(self):
        poly = make_rect(2.0, 6.0)
        _, best_angle = minimum_altitude(poly)
        assert math.cos(best_angle) == pytest.approx(0.0, abs=1e-6)

    def test_square_min_altitude_equals_side(self):
        poly = make_rect(1.0, 1.0)
        min_alt, _ = minimum_altitude(poly)
        assert min_alt == pytest.approx(1.0, abs=1e-6)

    def test_returned_altitude_matches_get_general_altitude(self):
        poly = make_rect(5.0, 3.0)
        min_alt, best_angle = minimum_altitude(poly)
        assert get_general_altitude(poly, best_angle) == pytest.approx(
            min_alt, abs=1e-6
        )

    def test_returned_angle_is_local_minimum(self):
        poly = make_rect(5.0, 2.0)
        _, best_angle = minimum_altitude(poly)
        assert _altitude_is_local_minimum(poly, best_angle)

    @pytest.mark.parametrize("tilt_deg", [15.0, 30.0, 45.0, 60.0, 75.0])
    def test_tilted_rect_optimal_angle(self, tilt_deg):
        poly = make_tilted_rect(4.0, 2.0, tilt_deg)
        min_alt, best_angle = minimum_altitude(poly)
        assert min_alt == pytest.approx(2.0, abs=1e-4)
        assert _altitude_is_local_minimum(poly, best_angle, n_probes=20)

    def test_empty_polygon(self):
        poly = Polygon()
        min_alt, best_angle = minimum_altitude(poly)
        assert min_alt == 0.0
        assert best_angle == 0.0

    def test_right_triangle_minimum(self):
        tri = Polygon([(0, 0), (4, 0), (0, 3)])
        min_alt, _ = minimum_altitude(tri)
        assert min_alt == pytest.approx(3.0, abs=1e-4)

    def test_minimum_altitude_less_than_suboptimal(self):
        poly = make_rect(10.0, 1.0)
        min_alt, _ = minimum_altitude(poly)
        alt_at_90 = get_general_altitude(poly, math.pi / 2)
        assert min_alt < alt_at_90 - 1e-6


class TestFindOptimalAngle:
    def test_returns_same_as_minimum_altitude(self):
        poly = make_rect(7.0, 2.0)
        assert find_optimal_angle(poly) == pytest.approx(
            minimum_altitude(poly)[1], abs=1e-10
        )

    def test_angle_in_range(self):
        for w, h in [(3, 1), (1, 3), (5, 5), (2, 7)]:
            angle = find_optimal_angle(make_rect(w, h))
            assert 0.0 <= angle < math.pi


class TestNumParallelPasses:
    def test_basic(self):
        assert num_parallel_passes(10.0, 2.0) == 5

    def test_rounds_up(self):
        assert num_parallel_passes(10.0, 3.0) == 4

    def test_exact(self):
        assert num_parallel_passes(6.0, 2.0) == 3

    def test_zero_alpha(self):
        assert num_parallel_passes(0.0, 2.0) == 0

    def test_zero_spacing(self):
        assert num_parallel_passes(5.0, 0.0) == 0

    def test_at_least_one(self):
        assert num_parallel_passes(0.001, 100.0) == 1


def make_donut(
    outer_w: float,
    outer_h: float,
    hole_x0: float,
    hole_y0: float,
    hole_w: float,
    hole_h: float,
) -> Polygon:
    outer = [(0, 0), (outer_w, 0), (outer_w, outer_h), (0, outer_h)]
    hole = [
        (hole_x0, hole_y0),
        (hole_x0 + hole_w, hole_y0),
        (hole_x0 + hole_w, hole_y0 + hole_h),
        (hole_x0, hole_y0 + hole_h),
    ]
    return Polygon(outer, [hole])


class TestPolygonsWithHoles:
    def test_rect_hole_altitude_theta0(self):
        poly = make_donut(6, 4, 1, 1, 4, 2)
        assert get_general_altitude(poly, 0.0) == pytest.approx(6.0, abs=1e-5)

    def test_rect_hole_altitude_theta90(self):
        poly = make_donut(6, 4, 1, 1, 4, 2)
        assert get_general_altitude(poly, math.pi / 2) == pytest.approx(10.0, abs=1e-5)

    def test_rect_hole_altitude_greater_than_solid(self):
        solid = make_rect(6.0, 4.0)
        holed = make_donut(6, 4, 1, 1, 4, 2)
        assert get_general_altitude(holed, 0.0) > get_general_altitude(solid, 0.0)

    def test_non_centered_hole_same_formula(self):
        hole_h = 1.5
        poly_left = make_donut(6, 4, 0.5, 1, 1, hole_h)
        poly_right = make_donut(6, 4, 4.5, 1, 1, hole_h)
        assert get_general_altitude(poly_left, 0.0) == pytest.approx(
            get_general_altitude(poly_right, 0.0), abs=1e-5
        )

    def test_partial_y_range_hole(self):
        poly = make_donut(6, 4, 1, 0.5, 4, 1.5)
        assert get_general_altitude(poly, 0.0) == pytest.approx(4.0 + 1.5, abs=1e-5)

    def test_wide_outer_wide_hole_min_angle_is_zero(self):
        poly = make_donut(6, 4, 1, 1, 4, 2)
        min_alt, best_angle = minimum_altitude(poly)
        assert min_alt == pytest.approx(6.0, abs=1e-4)
        assert math.sin(best_angle) == pytest.approx(0.0, abs=1e-5)

    def test_tall_outer_tall_hole_flips_optimal_angle(self):
        poly = make_donut(4, 6, 1, 1, 2, 4)
        min_alt, best_angle = minimum_altitude(poly)
        assert min_alt == pytest.approx(6.0, abs=1e-4)
        assert math.cos(best_angle) == pytest.approx(0.0, abs=1e-5)

    def test_square_outer_square_hole_equal_altitudes(self):
        poly = make_donut(4, 4, 1, 1, 2, 2)
        alt0 = get_general_altitude(poly, 0.0)
        alt90 = get_general_altitude(poly, math.pi / 2)
        assert alt0 == pytest.approx(alt90, abs=1e-5)
        assert alt0 == pytest.approx(6.0, abs=1e-5)

    def test_optimal_angle_is_local_minimum_with_hole(self):
        poly = make_donut(8, 3, 1, 0.5, 6, 2)
        _, best_angle = minimum_altitude(poly)
        assert _altitude_is_local_minimum(poly, best_angle, n_probes=20)

    def test_triangular_hole_altitude_theta0(self):
        outer = [(0, 0), (6, 0), (6, 4), (0, 4)]
        hole = [(2, 1), (4, 1), (3, 3)]
        poly = Polygon(outer, [hole])
        assert get_general_altitude(poly, 0.0) == pytest.approx(6.0, abs=1e-5)

    def test_triangular_hole_altitude_theta90(self):
        outer = [(0, 0), (6, 0), (6, 4), (0, 4)]
        hole = [(2, 1), (4, 1), (3, 3)]
        poly = Polygon(outer, [hole])
        assert get_general_altitude(poly, math.pi / 2) == pytest.approx(8.0, abs=1e-5)

    def test_triangular_hole_min_angle(self):
        outer = [(0, 0), (6, 0), (6, 4), (0, 4)]
        hole = [(2, 1), (4, 1), (3, 3)]
        poly = Polygon(outer, [hole])
        min_alt, best_angle = minimum_altitude(poly)
        assert min_alt == pytest.approx(6.0, abs=1e-4)
        assert math.sin(best_angle) == pytest.approx(0.0, abs=1e-5)

    def test_two_holes_altitude_theta0(self):
        outer = [(0, 0), (8, 0), (8, 4), (0, 4)]
        hole1 = [(1, 1), (2, 1), (2, 3), (1, 3)]
        hole2 = [(5, 1), (6, 1), (6, 3), (5, 3)]
        poly = Polygon(outer, [hole1, hole2])
        assert get_general_altitude(poly, 0.0) == pytest.approx(8.0, abs=1e-5)

    def test_two_holes_at_different_y_ranges(self):
        outer = [(0, 0), (6, 0), (6, 6), (0, 6)]
        hole1 = [(1, 1), (4, 1), (4, 2), (1, 2)]
        hole2 = [(1, 4), (4, 4), (4, 5), (1, 5)]
        poly = Polygon(outer, [hole1, hole2])
        assert get_general_altitude(poly, 0.0) == pytest.approx(8.0, abs=1e-5)

    def test_wrong_optimal_angle_due_to_hole_bug(self):
        outer = [(0, 0), (4, 0), (4, 6), (0, 6)]
        hole = [(0.5, 2), (3.5, 2), (3.5, 2.5), (0.5, 2.5)]
        poly = Polygon(outer, [hole])

        assert get_general_altitude(poly, 0.0) == pytest.approx(6.5, abs=1e-4)
        assert get_general_altitude(poly, math.pi / 2) == pytest.approx(7.0, abs=1e-4)

        _, best_angle = minimum_altitude(poly)
        assert math.sin(best_angle) == pytest.approx(0.0, abs=1e-5)

    def test_tilted_donut_optimal_angle_is_local_minimum(self):
        poly = make_donut(6, 2, 1, 0.5, 4, 1)
        poly_tilted = shapely_rotate(poly, 30, origin=poly.centroid, use_radians=False)
        _, best_angle = minimum_altitude(poly_tilted)
        assert _altitude_is_local_minimum(poly_tilted, best_angle, n_probes=20)
