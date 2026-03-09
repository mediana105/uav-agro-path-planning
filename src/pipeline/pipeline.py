import math
from typing import List, Optional
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from src.angle_finders.altitude_optimizer import find_optimal_angle
from src.decomposition.darp import DARP
from src.decomposition.field_decomposition import FieldDecomposition
from src.path_planning.boustrophedon import generate_snake_simple, path_length
from src.pipeline import DroneConfig, MissionResult, ZoneResult


def _compute_drone_time(drone: DroneConfig, path: list, num_turns: int) -> float:
    """
    Calculate total mission time for a single drone.

    Args:
        drone: Drone configuration with speed and turn_time parameters
        path: List of waypoints for the drone to follow
        num_turns: Number of turns in the path

    Returns:
        float: Total time in seconds (flight time + turn time)
    """
    if len(path) < 2:
        return 0.0

    flight_distance = path_length(path)
    flight_time = flight_distance / drone.speed
    turn_time = num_turns * drone.turn_time

    return flight_time + turn_time


def calculate_portions_by_productivity(drone_configs) -> List[float]:
    """
    Calculate field portions for each drone based on their productivity.
    More productive drones get larger portions to balance overall mission time.

    Productivity = swath_width * speed (area covered per unit time)

    Args:
        drone_configs: List of drone configurations with their parameters

    Returns:
        List[float]: Portions for each drone (sum = 1.0)
    """
    productivity = []
    for drone in drone_configs:
        prod = drone.swath_width * drone.speed
        productivity.append(prod)

    total_productivity = sum(productivity)
    if total_productivity == 0:
        return [1.0 / len(drone_configs)] * len(drone_configs)

    portions = [p / total_productivity for p in productivity]
    return portions


class MissionOptimizer:
    """
    Main class for optimizing agricultural drone mission planning.
    Handles field decomposition, path planning, and joint optimization.
    """

    def __init__(
            self,
            field_polygon: Polygon,
            drones: List[DroneConfig],
            cell_size: float = 1.0
    ):
        """
        Initialize mission optimizer with field and drone parameters.

        Args:
            field_polygon: Shapely polygon representing the field boundaries
            drones: List of drone configurations with their parameters
            cell_size: Grid cell size for field discretization (meters)
        """
        self.field_polygon = field_polygon
        self.drones = drones
        self.num_drones = len(drones)
        self.cell_size = cell_size

        self.initial_positions = [drone.start_position for drone in drones]

        self.min_x, self.min_y, self.max_x, self.max_y = field_polygon.bounds

    def _prepare_grid(self):
        """
        Prepare grid representation of the field for DARP algorithm.
        Discretizes the field polygon into a grid of cells.
        """
        self.field_decomp = FieldDecomposition(self.cell_size)
        self.field_decomp.from_polygon(self.field_polygon)
        self.grid_rows = self.field_decomp.grid_rows
        self.grid_cols = self.field_decomp.grid_cols

    def _cords_to_index(self, x: float, y: float) -> int:
        """
        Convert continuous coordinates to grid cell index.

        Args:
            x, y: Continuous coordinates

        Returns:
            int: Grid cell index in row-major order
        """
        col = int((x - self.min_x) / self.cell_size)

        row = int((y - self.min_y) / self.cell_size)

        col = max(0, min(col, self.grid_cols - 1))
        row = max(0, min(row, self.grid_rows - 1))

        return row * self.grid_cols + col

    def _run_darp(self, portions: List[float]) -> DARP:
        """
        Run DARP (Divide Areas Algorithm for Robots) to decompose field.

        Args:
            portions: Target portion for each drone (sum should be 1.0)

        Returns:
            DARP: Executed DARP instance with assignment matrix
        """
        if not hasattr(self, 'field_decomp'):
            self._prepare_grid()

        grid_positions = [
            self._cords_to_index(x, y)
            for x, y in self.initial_positions
        ]
        _, _, obstacle_positions = self.field_decomp.to_darp_grid()

        darp = DARP(
            nx=self.grid_rows,
            ny=self.grid_cols,
            notEqualPortions=True,
            given_initial_positions=grid_positions,
            given_portions=portions,
            obstacles_positions=obstacle_positions,
            visualization=False
        )

        darp.divideRegions()
        return darp

    def _extract_zone_polygon(self, assignment_matrix, drone_id: int) -> Polygon:
        """
        Extract polygon representing the zone assigned to a specific drone.

        Args:
            assignment_matrix: DARP assignment matrix (cell -> drone_id)
            drone_id: ID of the drone to extract zone for

        Returns:
            Polygon: Shapely polygon representing the drone's zone
        """
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

        if zone.geom_type == 'MultiPolygon':
            zone = max(zone.geoms, key=lambda g: g.area)

        return zone

    def evaluate(self, portions: Optional[List[float]] = None) -> MissionResult:
        """
        Evaluate mission performance for given field portions.
        Main pipeline: decomposition -> angle optimization -> path planning -> time calculation.

        Args:
            portions: Field portions for each drone. If None, calculated automatically.

        Returns:
            MissionResult: Complete mission evaluation with time for each drone
        """
        if portions is None:
            portions = calculate_portions_by_productivity(self.drones)

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
