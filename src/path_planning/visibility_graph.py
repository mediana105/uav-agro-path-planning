import heapq
import math
import logging

from shapely.geometry import LineString, Point, Polygon, MultiPolygon
from shapely.ops import nearest_points

logger = logging.getLogger(__name__)


class VisibilityGraph:
    _EPS = 1e-6

    def __init__(self, polygon: Polygon | MultiPolygon) -> None:
        self._original_polygon = polygon

        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        self._work_poly = polygon.simplify(0.05)

        self._safe_area = self._work_poly.buffer(1e-3, join_style=2)

        if isinstance(self._work_poly, MultiPolygon):
            parts = list(self._work_poly.geoms)
        else:
            parts = [self._work_poly]

        self._nodes: list[tuple[float, float]] = []
        self._adj: list[list[tuple[int, float]]] = []

        perimeter_edges = set()
        node_idx = 0

        for part in parts:
            ext_coords = list(part.exterior.coords)[:-1]
            start_idx = node_idx
            for v in ext_coords:
                self._nodes.append(v)
                node_idx += 1

            n_ext = len(ext_coords)
            for i in range(n_ext):
                u = start_idx + i
                v = start_idx + (i + 1) % n_ext
                perimeter_edges.add((min(u, v), max(u, v)))

            for interior in part.interiors:
                int_coords = list(interior.coords)[:-1]
                start_idx = node_idx
                for v in int_coords:
                    self._nodes.append(v)
                    node_idx += 1

                n_int = len(int_coords)
                for i in range(n_int):
                    u = start_idx + i
                    v = start_idx + (i + 1) % n_int
                    perimeter_edges.add((min(u, v), max(u, v)))

        H = len(self._nodes)
        self._adj = [[] for _ in range(H)]

        for i in range(H):
            for j in range(i + 1, H):
                is_perimeter = (i, j) in perimeter_edges

                if is_perimeter or self._visible(self._nodes[i], self._nodes[j]):
                    d = math.hypot(
                        self._nodes[i][0] - self._nodes[j][0],
                        self._nodes[i][1] - self._nodes[j][1],
                    )
                    self._adj[i].append((j, d))
                    self._adj[j].append((i, d))

    def _visible(self, u: tuple[float, float], v: tuple[float, float]) -> bool:
        if math.hypot(u[0] - v[0], u[1] - v[1]) < 1e-9:
            return True

        seg = LineString([u, v])
        return self._safe_area.covers(seg)

    def shortest_path(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> list[tuple[float, float]]:

        if self._visible(start, end):
            return [start, end]

        start = self._snap_to_safe_area(start)
        end = self._snap_to_safe_area(end)

        H = len(self._nodes)
        nodes = self._nodes
        n = H + 2

        extra_adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        for i in range(H):
            extra_adj[i] = list(self._adj[i])

        for tmp_idx, pt in [(H, start), (H + 1, end)]:
            for j in range(H):
                if self._visible(pt, nodes[j]):
                    d = math.hypot(pt[0] - nodes[j][0], pt[1] - nodes[j][1])
                    extra_adj[tmp_idx].append((j, d))
                    extra_adj[j].append((tmp_idx, d))

        dist = [math.inf] * n
        prev = [-1] * n
        dist[H] = 0.0
        heap = [(0.0, H)]

        while heap:
            cost, u = heapq.heappop(heap)
            if cost > dist[u] + 1e-12:
                continue
            if u == H + 1:
                break
            for v, w in extra_adj[u]:
                nc = dist[u] + w
                if nc < dist[v] - 1e-12:
                    dist[v] = nc
                    prev[v] = u
                    heapq.heappush(heap, (nc, v))

        if math.isinf(dist[H + 1]):
            return [start, end]

        path: list[tuple[float, float]] = []
        cur = H + 1
        while cur != -1:
            path.append(nodes[cur] if cur < H else (start if cur == H else end))
            cur = prev[cur]
        path.reverse()

        return self._simplify_path(path)

    def _snap_to_safe_area(self, pt: tuple[float, float]) -> tuple[float, float]:
        pt_geom = Point(pt)
        if self._safe_area.covers(pt_geom):
            return pt
        nearest = nearest_points(self._work_poly, pt_geom)[0]
        return (nearest.x, nearest.y)

    def _simplify_path(
        self, path: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        if len(path) <= 2:
            return path

        simplified = [path[0]]
        cur_idx = 0

        while cur_idx < len(path) - 1:
            next_idx = cur_idx + 1
            for j in range(len(path) - 1, cur_idx, -1):
                if self._visible(path[cur_idx], path[j]):
                    next_idx = j
                    break
            simplified.append(path[next_idx])
            cur_idx = next_idx

        return simplified


class SimplePolygonInteriorVG(VisibilityGraph):
    pass


def _transition_waypoints(
    start: tuple[float, float],
    end: tuple[float, float],
    polygon: Polygon | MultiPolygon,
) -> list[tuple[float, float]]:
    vg = VisibilityGraph(polygon)
    return vg.shortest_path(start, end)
