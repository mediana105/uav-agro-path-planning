from ..path_planning.boustrophedon import generate_boustrophedon_coverage, path_length


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
        path = generate_boustrophedon_coverage(polygon, swath, angle)

        if len(path) < 2:
            angle += step_deg
            continue

        cost = path_length(path)

        if cost < best_cost:
            best_cost = cost
            best_angle = angle

        angle += step_deg

    return best_angle
