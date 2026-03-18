import math

from shapely.affinity import rotate
from shapely.geometry import LineString

_NAN = (float("nan"), float("nan"))


def generate_snake_simple(polygon, angle, swath, start_position=None):
    rotated = rotate(polygon, -angle, origin="centroid")
    min_x, min_y, max_x, max_y = rotated.bounds

    columns = []
    x = min_x
    while x < max_x:
        line = LineString([(x, min_y - 1), (x, max_y + 1)])
        intersection = line.intersection(rotated)
        if not intersection.is_empty:
            sub_segs = _extract_sub_segments(intersection)
            if sub_segs:
                columns.append(sub_segs)
        x += swath

    if not columns:
        return []

    rotated_start = _rotate_point(start_position, polygon.centroid, -angle)

    first_x = columns[0][0][0][0]
    last_x = columns[-1][0][0][0]
    if abs(rotated_start[0] - last_x) < abs(rotated_start[0] - first_x):
        columns.reverse()

    go_up = True
    if rotated_start is not None:
        bottom_pt = columns[0][0][0]
        top_pt = columns[0][-1][-1]
        d_bot = math.hypot(bottom_pt[0] - rotated_start[0], bottom_pt[1] - rotated_start[1])
        d_top = math.hypot(top_pt[0] - rotated_start[0], top_pt[1] - rotated_start[1])
        go_up = d_bot <= d_top

    path = []
    for col in columns:
        if go_up:
            ordered = col
        else:
            ordered = [list(reversed(seg)) for seg in reversed(col)]

        for k, seg_pts in enumerate(ordered):
            if k > 0:
                path.append(_NAN)
            path.extend(seg_pts)

        go_up = not go_up

    return _rotate_path_back(path, polygon.centroid, angle)


def _extract_sub_segments(segment):
    geom_type = segment.geom_type

    if geom_type == "LineString":
        pts = sorted(segment.coords, key=lambda p: p[1])
        return [pts] if pts else []

    if geom_type == "MultiLineString":
        result = []
        for geom in segment.geoms:
            pts = sorted(geom.coords, key=lambda p: p[1])
            if pts:
                result.append(pts)
        result.sort(key=lambda s: s[0][1])
        return result

    if geom_type == "Point":
        return [[(segment.x, segment.y)]]

    if geom_type == "GeometryCollection":
        result = []
        for geom in segment.geoms:
            result.extend(_extract_sub_segments(geom))
        result.sort(key=lambda s: s[0][1])
        return result

    return []


def _rotate_point(point, centroid, angle_deg):
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    dx = point[0] - centroid.x
    dy = point[1] - centroid.y
    return (
        centroid.x + cos_a * dx - sin_a * dy,
        centroid.y + sin_a * dx + cos_a * dy,
    )


def _rotate_path_back(path, centroid, angle):
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    result = []
    for px, py in path:
        if math.isnan(px):
            result.append(_NAN)
            continue
        dx = px - centroid.x
        dy = py - centroid.y
        result.append((
            centroid.x + cos_a * dx - sin_a * dy,
            centroid.y + sin_a * dx + cos_a * dy,
        ))
    return result


def path_length(path):
    if len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        if math.isnan(x1) or math.isnan(x2):
            continue
        total += math.hypot(x2 - x1, y2 - y1)
    return total
