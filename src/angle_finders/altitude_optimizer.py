import math

from shapely.affinity import rotate
from shapely.geometry import Polygon


def get_general_altitude(p: Polygon, theta: float) -> float:
    """
    Calculate general altitude (width) of polygon in given direction using scanning line algorithm.

    Based on: Stanislav Bochkarev, Stephen L. Smith.
    "On Minimizing Turns in Robot Coverage Path Planning"
    """
    # Step 1: Initialize counters
    counter = 0
    alpha = 0.0

    # Step 2: Rotate polygon to align with scanning direction
    centroid = p.centroid
    p_rot = rotate(p, -theta, origin=centroid, use_radians=True)

    # Step 3: Collect all vertices with their neighbors
    all_vertices = []

    # Process exterior boundary
    exterior_cords = list(p_rot.exterior.coords)
    n_exterior = len(exterior_cords) - 1

    for i in range(n_exterior):
        x, y = exterior_cords[i]
        prev_x, prev_y = exterior_cords[(i - 1) % n_exterior]
        next_x, next_y = exterior_cords[(i + 1) % n_exterior]

        all_vertices.append({
            'x': x, 'y': y,
            'prev_x': prev_x, 'next_x': next_x,
            'type': 'exterior'
        })

    # Step 4: Process interior boundaries (holes)
    for interior in p_rot.interiors:
        interior_cords = list(interior.coords)
        n_interior = len(interior_cords) - 1

        for i in range(n_interior):
            x, y = interior_cords[i]
            prev_x, prev_y = interior_cords[(i - 1) % n_interior]
            next_x, next_y = interior_cords[(i + 1) % n_interior]

            all_vertices.append({
                'x': x, 'y': y,
                'prev_x': prev_x, 'next_x': next_x,
                'type': 'interior'
            })

    # Step 5: Sort vertices by x-coordinate for scanning
    all_vertices.sort(key=lambda v: v['x'])

    # Step 6: Scan from left to right
    prev_x = None

    for i, vertex in enumerate(all_vertices):
        x = vertex['x']
        prev_x_i = vertex['prev_x']
        next_x_i = vertex['next_x']

        # Step 7: Handle first vertex
        if i == 0:
            prev_x = x
            # Count initial intersections
            if prev_x_i > x and next_x_i > x:
                counter += 1
            elif prev_x_i < x and next_x_i < x:
                counter -= 1
            continue

        # Step 8: Accumulate altitude
        # alpha += (number of active segments) * (distance moved)
        delta_x = x - prev_x
        alpha += counter * delta_x

        # Step 9: Update intersection counter
        # Check if vertex represents an edge crossing
        both_on_right = (prev_x_i > x) and (next_x_i > x)  # Edge to the right
        both_on_left = (prev_x_i < x) and (next_x_i < x)  # Edge to the left

        if both_on_right:
            # Step 10: Scan line enters polygon
            counter += 1
        elif both_on_left:
            # Step 11: Scan line exits polygon
            counter -= 1

        prev_x = x

    # Step 12: Return absolute altitude value
    return abs(alpha)


def find_optimal_angle(polygon: Polygon) -> float:
    """
    Find optimal angle that minimizes altitude (and thus turns) for coverage path.

    Theorem: Optimal direction is orthogonal to one of polygon edges.
    """
    best_angle = 0.0
    min_altitude = float('inf')

    # Collect all boundaries to check
    all_boundaries = [polygon.exterior.coords]  # Exterior boundary
    for interior in polygon.interiors:  # Interior boundaries (holes)
        all_boundaries.append(interior.coords)

    # Check angles orthogonal to each edge
    for boundary_cords in all_boundaries:
        cords = list(boundary_cords)
        for i in range(len(cords) - 1):
            # Calculate edge angle
            x1, y1 = cords[i]
            x2, y2 = cords[i + 1]
            dx, dy = x2 - x1, y2 - y1

            # Calculate edge direction angle
            side_angle = math.atan2(dy, dx)

            # Calculate orthogonal coverage angle
            # Coverage lines are perpendicular to edges
            cur_angle = side_angle + math.pi / 2

            # Normalize angle to [0, π) range
            cur_angle = cur_angle % math.pi

            # Calculate altitude for this angle
            altitude = get_general_altitude(polygon, cur_angle)

            # Update best angle if this is better
            if altitude < min_altitude:
                min_altitude = altitude
                best_angle = cur_angle

    return best_angle
