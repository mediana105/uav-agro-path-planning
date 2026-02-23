from typing import List, Optional
from shapely.geometry import Polygon, box
from shapely.lib import unary_union

from . import DroneConfig
from . import MissionResult, ZoneResult
from ..angle_finders.altitude_optimizer import find_optimal_angle
from ..decomposition.darp import DARP
from ..decomposition.field_decomposition import FieldDecomposition
from ..path_planning.boustrophedon import *


def _compute_drone_time(drone: DroneConfig, path: list, num_turns: int) -> float:
    if len(path) < 2:
        return 0.0

    flight_distance = path_length(path)
    flight_time = flight_distance / drone.speed
    turn_time = num_turns * drone.turn_time

    return flight_time + turn_time


class MissionOptimizer:

    def __init__(
            self,
            field_polygon: Polygon,
            drones: List[DroneConfig],
            cell_size: float = 1.0
    ):
        self.field_polygon = field_polygon
        self.drones = drones
        self.num_drones = len(drones)
        self.cell_size = cell_size

        self.initial_positions = [drone.start_position for drone in drones]

        self.min_x, self.min_y, self.max_x, self.max_y = field_polygon.bounds

    def _prepare_grid(self):
        self.field_decomp = FieldDecomposition(self.cell_size)
        self.field_decomp.from_polygon(self.field_polygon)
        self.grid_rows = self.field_decomp.grid_rows
        self.grid_cols = self.field_decomp.grid_cols

    def _coords_to_index(self, x: float, y: float) -> int:
        col = int((x - self.min_x) / self.cell_size)

        row = int((y - self.min_y) / self.cell_size)

        col = max(0, min(col, self.grid_cols - 1))
        row = max(0, min(row, self.grid_rows - 1))

        return row * self.grid_cols + col

    def _run_darp(self, portions: List[float]) -> DARP:
        grid_positions = [
            self._coords_to_index(x, y)
            for x, y in self.initial_positions
        ]
        _, _, obstacle_positions = self.field_decomp.to_darp_grid()

        not_equal = not all(abs(p - portions[0]) < 1e-6 for p in portions)

        darp = DARP(
            nx=self.grid_rows,
            ny=self.grid_cols,
            notEqualPortions=not_equal,
            given_initial_positions=grid_positions,
            given_portions=portions,
            obstacles_positions=obstacle_positions,
            visualization=False
        )

        darp.divideRegions()
        return darp

    def _extract_zone_polygon(self, assignment_matrix, drone_id: int) -> Polygon:
        cells = []

        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                if assignment_matrix[row, col] == drone_id:
                    x = self.min_x + col * self.cell_size
                    y = self.min_y + row * self.cell_size
                    cell = box(x, y, x + self.cell_size, y + self.cell_size)
                    cells.append(cell)

        if not cells:
            return Polygon()

        zone = unary_union(cells)

        if zone.geom_type == 'MultiPolygon':
            zone = max(zone.geoms, key=lambda g: g.area)

        return zone

    def evaluate(self, portions: Optional[List[float]] = None) -> MissionResult:
        if portions is None:
            portions = [1.0 / self.num_drones] * self.num_drones

        darp = self._run_darp(portions)

        zone_results = []

        for drone in self.drones:
            zone_polygon = self._extract_zone_polygon(darp.A, drone.id)

            if zone_polygon.is_empty:
                zone_results.append(ZoneResult(
                    drone_id=drone.id,
                    zone_polygon=zone_polygon,
                    optimal_angle=0.0,
                    path=[],
                    total_time=0.0
                ))
                continue

            optimal_angle = find_optimal_angle(zone_polygon)

            angle_deg = math.degrees(optimal_angle)
            path = generate_snake_simple(zone_polygon, angle_deg, drone.swath_width)

            num_turns = max(0, len(path) // 2 - 1)

            total_time = _compute_drone_time(drone, path, num_turns)

            zone_results.append(ZoneResult(
                drone_id=drone.id,
                zone_polygon=zone_polygon,
                optimal_angle=optimal_angle,
                path=path,
                total_time=total_time
            ))

        return MissionResult(zones=zone_results)
