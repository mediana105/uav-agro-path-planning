import math

from shapely.affinity import rotate
from shapely.geometry import Polygon

_SWEEP_DEGENERACY_RAD = 1e-7


def get_general_altitude(p: Polygon, theta: float) -> float:
    centroid = p.centroid
    p_rot = rotate(
        p,
        -(theta + _SWEEP_DEGENERACY_RAD),
        origin=centroid,
        use_radians=True,
    )

    all_vertices = []

    exterior_coords = list(p_rot.exterior.coords)
    n = len(exterior_coords) - 1
    for i in range(n):
        x, y = exterior_coords[i]
        _, prev_y = exterior_coords[(i - 1) % n]
        _, next_y = exterior_coords[(i + 1) % n]
        all_vertices.append(
            {"y": y, "prev_y": prev_y, "next_y": next_y, "interior": False}
        )

    for ring in p_rot.interiors:
        hole_coords = list(ring.coords)
        m = len(hole_coords) - 1
        for i in range(m):
            x, y = hole_coords[i]
            _, prev_y = hole_coords[(i - 1) % m]
            _, next_y = hole_coords[(i + 1) % m]
            all_vertices.append(
                {"y": y, "prev_y": prev_y, "next_y": next_y, "interior": True}
            )

    all_vertices.sort(key=lambda v: v["y"])

    counter = 0
    alpha = 0.0
    prev_y = None

    for i, v in enumerate(all_vertices):
        y = v["y"]
        prev_y_v = v["prev_y"]
        next_y_v = v["next_y"]
        is_interior = v["interior"]

        if i == 0:
            prev_y = y
            both_above = prev_y_v > y and next_y_v > y
            both_below = prev_y_v < y and next_y_v < y
            if both_above:
                counter += 1
            elif both_below:
                counter -= 1
            continue

        alpha += counter * (y - prev_y)

        both_above = prev_y_v > y and next_y_v > y
        both_below = prev_y_v < y and next_y_v < y

        if both_above:
            counter += 1
        elif both_below:
            counter -= 1

        prev_y = y

    return alpha


def minimum_altitude(polygon: Polygon) -> tuple[float, float]:
    if polygon.is_empty or polygon.area <= 0:
        return 0.0, 0.0

    best_angle = 0.0
    min_alt = float("inf")

    all_boundaries = [polygon.exterior.coords]
    for interior in polygon.interiors:
        all_boundaries.append(interior.coords)

    for boundary_coords in all_boundaries:
        coords = list(boundary_coords)
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            dx, dy = x2 - x1, y2 - y1
            side_angle = math.atan2(dy, dx)
            cur_angle = (side_angle + math.pi / 2) % math.pi

            altitude = get_general_altitude(polygon, cur_angle)
            if altitude < min_alt:
                min_alt = altitude
                best_angle = cur_angle

    return min_alt, best_angle


def find_optimal_angle(polygon: Polygon) -> float:
    return minimum_altitude(polygon)[1]


def num_parallel_passes(alpha: float, stripe_spacing: float) -> int:
    if stripe_spacing <= 0 or alpha <= 0:
        return 0
    return max(1, int(math.ceil(alpha / stripe_spacing)))
