from dataclasses import dataclass

from shapely.geometry import Polygon


@dataclass
class ZoneResult:
    drone_id: int
    zone_polygon: Polygon
    optimal_angle: float
    path: list[tuple[float, float]]
    total_time: float
    rth_count: int = 0  # number of refuel/recharge stops


@dataclass
class MissionResult:
    zones: list[ZoneResult]

    @property
    def mission_time(self) -> float:
        return max(z.total_time for z in self.zones)
