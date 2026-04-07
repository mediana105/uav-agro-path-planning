import math

from shapely.geometry import Polygon

from ..angle_finders.altitude_optimizer import find_optimal_angle
from ..path_planning.boustrophedon import generate_boustrophedon_coverage, path_length


def build_zone_snake(
    zone_polygon: Polygon,
    swath_width: float,
    angle_rad: float = None,
    start_position=None,
) -> tuple[list[tuple[float, float]], float, int]:

    if zone_polygon.is_empty:
        return [], 0.0, 0

    if angle_rad is None:
        angle_rad = find_optimal_angle(zone_polygon)

    angle_deg = math.degrees(angle_rad)
    path = generate_boustrophedon_coverage(zone_polygon, swath_width, angle_deg, start_position=start_position)

    num_turns = count_turns(path)

    return path, angle_rad, num_turns


def zone_path_length(path: list[tuple[float, float]]) -> float:
    return path_length(path)


def count_turns(path: list, angle_threshold_deg: float = 10.0) -> int:
    real_points = [p for p in path if not math.isnan(p[0])]

    if len(real_points) < 3:
        return 0

    turns = 0
    threshold = math.radians(angle_threshold_deg)

    for i in range(1, len(real_points) - 1):
        x0, y0 = real_points[i - 1]
        x1, y1 = real_points[i]
        x2, y2 = real_points[i + 1]

        dx1, dy1 = x1 - x0, y1 - y0
        dx2, dy2 = x2 - x1, y2 - y1

        len1 = math.hypot(dx1, dy1)
        len2 = math.hypot(dx2, dy2)
        if len1 < 1e-10 or len2 < 1e-10:
            continue

        cos_a = (dx1 * dx2 + dy1 * dy2) / (len1 * len2)
        cos_a = max(-1.0, min(1.0, cos_a))
        angle = math.acos(cos_a)

        if angle > threshold:
            turns += 1

    return turns
