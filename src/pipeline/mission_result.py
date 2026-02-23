from dataclasses import dataclass
from typing import List, Tuple
from shapely.geometry import Polygon


@dataclass
class ZoneResult:
    drone_id: int
    zone_polygon: Polygon
    optimal_angle: float
    path: List[Tuple[float, float]]
    total_time: float


@dataclass
class MissionResult:
    zones: List[ZoneResult]

    @property
    def mission_time(self) -> float:
        return max(z.total_time for z in self.zones)
