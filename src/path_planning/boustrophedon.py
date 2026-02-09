import math
from shapely.affinity import rotate
from shapely.geometry import LineString


def generate_snake_simple(polygon, angle, swath):
    rotated = rotate(polygon, -angle, origin="centroid")
    min_x, min_y, max_x, max_y = rotated.bounds

    path = []
    x = min_x
    go_down = True

    while x < max_x:
        line = LineString([(x, min_y), (x, max_y)])
        segment = line.intersection(rotated)

        if not segment.is_empty:
            points = list(segment.coords)
            if not go_down:
                points.reverse()
            path.extend(points)
            go_down = not go_down

        x += swath

    centroid = polygon.centroid
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    final_path = []
    for px, py in path:
        dx = px - centroid.x
        dy = py - centroid.y
        final_path.append((
            centroid.x + cos_a * dx - sin_a * dy,
            centroid.y + sin_a * dx + cos_a * dy
        ))

    return final_path


def path_length(path):
    if len(path) < 2:
        return 0

    total = 0
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        total += math.hypot(x2 - x1, y2 - y1)

    return total