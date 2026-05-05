import math

from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from ..angle_finders.altitude_optimizer import find_optimal_angle
from ..decomposition.darp import DARP
from ..decomposition.field_decomposition import FieldDecomposition
from ..path_planning.coverage_path import build_zone_snake, count_turns
from ..path_planning.boustrophedon import apply_resource_limits, path_length
from . import DroneConfig, MissionResult, ZoneResult


def interior_polygon(interior) -> Polygon:
    return Polygon(interior.coords)


def _compute_drone_time(drone: DroneConfig, path: list, num_turns: int) -> float:
    if len(path) < 2:
        return 0.0

    flight_distance = path_length(path)
    flight_time = flight_distance / drone.speed
    turn_time = num_turns * drone.turn_time

    return flight_time + turn_time


def calculate_portions_rth_aware(drone_configs, field_polygon: Polygon) -> list[float]:
    centroid = field_polygon.centroid

    effective_productivity = []
    for drone in drone_configs:
        sortie_by_substance = (
            drone.tank_volume / drone.substance_rate
            if drone.substance_rate > 0 and math.isfinite(drone.tank_volume)
            else math.inf
        )
        sortie_by_time = (
            drone.max_flight_time * drone.speed
            if math.isfinite(drone.max_flight_time)
            else math.inf
        )
        l_sortie = min(sortie_by_substance, sortie_by_time)

        if math.isfinite(l_sortie) and l_sortie > 0:
            d_home = math.hypot(
                drone.start_position[0] - centroid.x,
                drone.start_position[1] - centroid.y,
            )
            speed_eff = drone.speed * l_sortie / (l_sortie + 2.0 * d_home)
        else:
            speed_eff = drone.speed  # no resource limits — no correction needed

        effective_productivity.append(drone.swath_width * speed_eff)

    total = sum(effective_productivity)
    if total == 0:
        return [1.0 / len(drone_configs)] * len(drone_configs)

    return [p / total for p in effective_productivity]


class MissionOptimizer:
    def __init__(
        self,
        field_polygon: Polygon,
        drones: list[DroneConfig],
        cell_size: float = 1.0,
        strategy: str = "greedy_safe",
    ):
        self.field_polygon = field_polygon
        self.drones = drones
        self.num_drones = len(drones)
        self.cell_size = cell_size
        self.strategy = strategy

        self.initial_positions = [drone.start_position for drone in drones]

        self.min_x, self.min_y, self.max_x, self.max_y = field_polygon.bounds

    def _prepare_grid(self):
        self.field_decomp = FieldDecomposition(self.cell_size)
        self.field_decomp.from_polygon(self.field_polygon)
        self.grid_rows = self.field_decomp.grid_rows
        self.grid_cols = self.field_decomp.grid_cols

    def _cords_to_index(self, x: float, y: float) -> int:
        col = int((x - self.min_x) / self.cell_size)

        row = int((y - self.min_y) / self.cell_size)

        col = max(0, min(col, self.grid_cols - 1))
        row = max(0, min(row, self.grid_rows - 1))

        return row * self.grid_cols + col

    def _run_darp(self, portions: list[float]) -> DARP:
        if not hasattr(self, "field_decomp"):
            self._prepare_grid()

        grid_positions = [self._cords_to_index(x, y) for x, y in self.initial_positions]
        _, _, obstacle_positions = self.field_decomp.to_darp_grid()

        darp = DARP(
            nx=self.grid_rows,
            ny=self.grid_cols,
            notEqualPortions=True,
            given_initial_positions=grid_positions,
            given_portions=portions,
            obstacles_positions=obstacle_positions,
            visualization=False,
            MaxIter=1000,
        )

        darp.divideRegions()
        return darp

    def _extract_zone_polygon(
        self, assignment_matrix, drone_id: int
    ) -> Polygon | BaseGeometry:
        cells = []

        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                if assignment_matrix[row, col] == drone_id:
                    x = self.min_x + col * self.cell_size  # real cords
                    y = self.min_y + row * self.cell_size  # real cords
                    cell = box(x, y, x + self.cell_size, y + self.cell_size)
                    cells.append(cell)

        if not cells:
            return Polygon()

        zone = unary_union(cells)

        for interior in self.field_polygon.interiors:
            zone = zone.difference(interior_polygon(interior))

        if zone.is_empty:
            return Polygon()

        return zone

    def evaluate(self, portions: list[float], exact: bool = False) -> MissionResult:
        if portions is None:
            portions = calculate_portions_rth_aware(self.drones, self.field_polygon)

        darp = self._run_darp(portions)

        zone_results = []

        for drone in self.drones:
            zone_polygon = self._extract_zone_polygon(darp.A, drone.id)

            if zone_polygon.is_empty:
                zone_results.append(
                    ZoneResult(
                        drone_id=drone.id,
                        zone_polygon=zone_polygon,
                        optimal_angle=0.0,
                        path=[],
                        total_time=0.0,
                    )
                )
                continue

            if zone_polygon.geom_type == "MultiPolygon":
                sub_polygons = list(zone_polygon.geoms)
            else:
                sub_polygons = [zone_polygon]

            largest = max(sub_polygons, key=lambda g: g.area)
            optimal_angle = find_optimal_angle(largest)

            path = []
            current_start = drone.start_position
            for sub_poly in sub_polygons:
                sub_path, ang, _ = build_zone_snake(
                    sub_poly,
                    drone.swath_width,
                    angle_rad=optimal_angle,
                    start_position=current_start,
                    strategy=self.strategy,
                    exact=exact,
                )
                if not sub_path:
                    continue
                path.extend(sub_path[1:] if path else sub_path)
                real_pts = [p for p in sub_path if not math.isnan(p[0])]
                if real_pts:
                    current_start = real_pts[-1]

            path, rth_count = apply_resource_limits(
                path,
                start_position=drone.start_position,
                speed=drone.speed,
                substance_rate=drone.substance_rate,
                tank_volume=drone.tank_volume,
                max_flight_time=drone.max_flight_time,
                turn_time=drone.turn_time,
                obstacles_polygon=zone_polygon,
            )

            num_turns = count_turns(path)

            total_time = _compute_drone_time(drone, path, num_turns)

            zone_results.append(
                ZoneResult(
                    drone_id=drone.id,
                    zone_polygon=zone_polygon,
                    optimal_angle=optimal_angle,
                    path=path,
                    total_time=total_time,
                    rth_count=rth_count,
                )
            )

        return MissionResult(zones=zone_results)
