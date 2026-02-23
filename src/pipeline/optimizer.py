from typing import List, Optional
from shapely.geometry import Polygon

from .drone_config import DroneConfig
from .mission_result import MissionResult, ZoneResult


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

    def evaluate(self, portions: Optional[List[float]] = None) -> MissionResult:
        # TODO
        pass
