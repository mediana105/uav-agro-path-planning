import heapq
import math

from shapely.geometry import LinearRing, LineString, Point, Polygon


class VisibilityGraph:

    _EPS = 1e-6

    def __init__(self, polygon) -> None:
        from shapely.geometry import MultiPolygon as _MP

        if isinstance(polygon, _MP):
            parts = list(polygon.geoms)
        else:
            parts = [polygon]
        self._parts = parts

        self._holes: list[Polygon] = []
        for part in parts:
            for interior in part.interiors:
                self._holes.append(Polygon(interior.coords).buffer(-self._EPS))

        self._nodes: list[tuple[float, float]] = []
        for part in parts:
            for interior in part.interiors:
                self._nodes.extend(list(interior.coords)[:-1])

        H = len(self._nodes)

        self._adj: list[list[tuple[int, float]]] = [[] for _ in range(H)]
        for i in range(H):
            for j in range(i + 1, H):
                if self._visible(self._nodes[i], self._nodes[j]):
                    d = math.hypot(
                        self._nodes[i][0] - self._nodes[j][0],
                        self._nodes[i][1] - self._nodes[j][1],
                    )
                    self._adj[i].append((j, d))
                    self._adj[j].append((i, d))

    def _visible(self, u: tuple[float, float], v: tuple[float, float]) -> bool:
        seg = LineString([u, v])
        return not any(h.intersects(seg) for h in self._holes)

    def shortest_path(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> list[tuple[float, float]]:
        if self._visible(start, end):
            return [start, end]

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
            if self._visible(start, end):
                d = math.hypot(start[0] - end[0], start[1] - end[1])
                extra_adj[H].append((H + 1, d))
                extra_adj[H + 1].append((H, d))

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
        return path


class SimplePolygonInteriorVG(VisibilityGraph):
    pass


def _transition_waypoints(
    start: tuple[float, float],
    end: tuple[float, float],
    polygon: Polygon,
) -> list[tuple[float, float]]:
    waypoints: list[tuple[float, float]] = [start]
    cur = start

    for interior in polygon.interiors:
        seg = LineString([cur, end])
        hole_poly = Polygon(interior.coords)

        mid = seg.interpolate(0.5, normalized=True)
        if not hole_poly.buffer(-1e-3).contains(mid):
            continue

        ring = LinearRing(interior.coords)
        inter = seg.intersection(ring)

        pts: list[tuple[float, float]] = []
        if inter.is_empty:
            pass
        elif inter.geom_type == "Point":
            pts = [(inter.x, inter.y)]
        elif inter.geom_type == "MultiPoint":
            pts = [(p.x, p.y) for p in inter.geoms]
        elif inter.geom_type in ("LineString", "GeometryCollection", "MultiLineString"):
            if hasattr(inter, "coords"):
                pts = [inter.coords[0], inter.coords[-1]]
            elif hasattr(inter, "geoms"):
                for g in inter.geoms:
                    if hasattr(g, "coords"):
                        pts.append(g.coords[0])
                        pts.append(g.coords[-1])

        if len(pts) < 2:
            start_on_ring = ring.distance(Point(cur)) < 1e-6
            end_on_ring = ring.distance(Point(end)) < 1e-6

            if len(pts) == 1:
                only_pt = pts[0]
                near_end = math.hypot(only_pt[0] - end[0], only_pt[1] - end[1]) < 1e-6
                near_start = math.hypot(only_pt[0] - cur[0], only_pt[1] - cur[1]) < 1e-6

                if near_end and start_on_ring:
                    pts = [cur, only_pt]
                elif near_start and end_on_ring:
                    pts = [only_pt, end]
                elif end_on_ring:
                    pts.append(end)
                elif start_on_ring:
                    pts.insert(0, cur)
                else:
                    continue

            elif len(pts) == 0:
                if start_on_ring and end_on_ring:
                    pts = [cur, end]
                else:
                    continue

        pts.sort(key=lambda p: math.hypot(p[0] - cur[0], p[1] - cur[1]))
        p_in = pts[0]
        p_out = pts[-1]

        ring_coords: list[tuple[float, float]] = list(interior.coords)[:-1]
        n = len(ring_coords)

        def seg_index(pt: tuple[float, float]) -> int:
            best_i, best_d = 0, math.inf
            for i in range(n):
                j = (i + 1) % n
                s = LineString([ring_coords[i], ring_coords[j]])
                d = s.distance(Point(pt))
                if d < best_d:
                    best_d = d
                    best_i = i
            return best_i

        i_in = seg_index(p_in)
        i_out = seg_index(p_out)

        def make_cw() -> list[tuple[float, float]]:
            route: list[tuple[float, float]] = [p_in]
            i = (i_in + 1) % n
            steps = 0
            while i != (i_out + 1) % n and steps < n:
                route.append(ring_coords[i])
                i = (i + 1) % n
                steps += 1
            route.append(p_out)
            return route

        def make_ccw() -> list[tuple[float, float]]:
            route: list[tuple[float, float]] = [p_in]
            i = i_in
            steps = 0
            while i != i_out and steps < n:
                route.append(ring_coords[i])
                i = (i - 1) % n
                steps += 1
            route.append(ring_coords[i_out])
            route.append(p_out)
            return route

        def route_len(verts: list[tuple[float, float]]) -> float:
            return sum(
                math.hypot(verts[k + 1][0] - verts[k][0], verts[k + 1][1] - verts[k][1])
                for k in range(len(verts) - 1)
            )

        cw = make_cw()
        ccw = make_ccw()
        detour = cw if route_len(cw) <= route_len(ccw) else ccw

        waypoints.extend(detour)
        cur = p_out

    if math.hypot(waypoints[-1][0] - end[0], waypoints[-1][1] - end[1]) > 1e-9:
        waypoints.append(end)
    return waypoints
