from shapely.geometry import Polygon
from shapely.affinity import rotate


def get_general_altitude(P: Polygon, theta: float) -> float:
    # Step 1
    counter = 0
    alpha = 0.0

    # Step 2
    centroid = P.centroid
    P_rot = rotate(P, -theta, origin=centroid, use_radians=True)

    all_vertices = []
    exterior_coords = list(P_rot.exterior.coords)
    n_exterior = len(exterior_coords) - 1

    for i in range(n_exterior):
        x, y = exterior_coords[i]
        prev_x, prev_y = exterior_coords[(i - 1) % n_exterior]
        next_x, next_y = exterior_coords[(i + 1) % n_exterior]

        all_vertices.append({
            'x': x, 'y': y,
            'prev_x': prev_x, 'next_x': next_x,
            'type': 'exterior'
        })

    for interior in P_rot.interiors:
        interior_coords = list(interior.coords)
        n_interior = len(interior_coords) - 1

        for i in range(n_interior):
            x, y = interior_coords[i]
            prev_x, prev_y = interior_coords[(i - 1) % n_interior]
            next_x, next_y = interior_coords[(i + 1) % n_interior]

            all_vertices.append({
                'x': x, 'y': y,
                'prev_x': prev_x, 'next_x': next_x,
                'type': 'interior'
            })

    # Step 3
    all_vertices.sort(key=lambda v: v['x'])

    # Step 4
    prev_x = None

    for i, vertex in enumerate(all_vertices):
        x = vertex['x']
        prev_x_i = vertex['prev_x']
        next_x_i = vertex['next_x']
        if i == 0:
            prev_x = x
            if prev_x_i > x and next_x_i > x:
                counter += 1
            elif prev_x_i < x and next_x_i < x:
                counter -= 1
            continue

        # Step 5
        delta_x = x - prev_x
        alpha += counter * delta_x

        # Steps 6-9
        both_on_right = (prev_x_i > x) and (next_x_i > x)
        both_on_left = (prev_x_i < x) and (next_x_i < x)

        if both_on_right:
            # Step 7
            counter += 1
        elif both_on_left:
            # Step 9
            counter -= 1

        prev_x = x

    # Step 10
    return abs(alpha)
