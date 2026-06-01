import math

from ..path_planning.coverage_path import build_zone_snake
from ..path_planning import utils as _pp_utils

path_length = _pp_utils.path_length


def find_angle_by_bruteforce(
    polygon,
    swath,
    step_deg=5.0,
    angle_min=0.0,
    angle_max=180.0,
):
    best_angle = None
    best_cost = float("inf")

    angle = angle_min

    while angle < angle_max:
        path, _, _ = build_zone_snake(polygon, swath, angle_rad=math.radians(angle))

        if len(path) < 2:
            angle += step_deg
            continue

        cost = path_length(path)

        if cost < best_cost:
            best_cost = cost
            best_angle = angle

        angle += step_deg

    return best_angle
