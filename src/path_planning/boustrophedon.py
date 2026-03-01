import math
from shapely.affinity import rotate
from shapely.geometry import LineString


def generate_snake_simple(polygon, angle, swath):
    """
    Generate snake (boustrophedon) coverage path for a polygon.
    Handles connected but possibly concave polygons from DARP decomposition.

    Args:
        polygon: Shapely polygon to cover
        angle: Coverage angle in degrees
        swath: Width of coverage swath (meters)

    Returns:
        List of (x, y) waypoints
    """
    rotated = rotate(polygon, -angle, origin="centroid")
    min_x, min_y, max_x, max_y = rotated.bounds

    path = []
    x = min_x
    go_down = True

    while x < max_x:
        line = LineString([(x, min_y - 1), (x, max_y + 1)])
        segment = line.intersection(rotated)

        if segment.is_empty:
            x += swath
            continue

        # Extract points based on geometry type
        points = _extract_segment_points(segment)

        if not points:
            x += swath
            continue

        # Reverse direction for alternating passes
        if not go_down:
            points.reverse()

        path.extend(points)
        go_down = not go_down
        x += swath

    # Rotate path back to original orientation
    final_path = _rotate_path_back(path, polygon.centroid, angle)

    return final_path


def _extract_segment_points(segment):
    """
    Extract ordered points from intersection segment.

    Args:
        segment: Shapely geometry (LineString, MultiLineString, Point, etc.)

    Returns:
        List of (x, y) points sorted by Y coordinate
    """
    points = []

    if segment.geom_type == 'LineString':
        points = list(segment.coords)

    elif segment.geom_type == 'MultiLineString':
        # Collect all points from all segments
        for seg in segment.geoms:
            points.extend(list(seg.coords))
        # Sort by Y to ensure correct traversal order
        points.sort(key=lambda p: p[1])

    elif segment.geom_type == 'Point':
        points = [(segment.x, segment.y)]

    elif segment.geom_type == 'GeometryCollection':
        for geom in segment.geoms:
            points.extend(_extract_segment_points(geom))
        points.sort(key=lambda p: p[1])

    return points


def _rotate_path_back(path, centroid, angle):
    """
    Rotate path back to original polygon orientation.

    Args:
        path: List of (x, y) points in rotated coordinates
        centroid: Center point for rotation
        angle: Original rotation angle in degrees

    Returns:
        List of (x, y) points in original coordinates
    """
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
    """
    Calculate total length of a path.

    Args:
        path: List of (x, y) waypoints

    Returns:
        float: Total path length
    """
    if len(path) < 2:
        return 0

    total = 0
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        total += math.hypot(x2 - x1, y2 - y1)

    return total
