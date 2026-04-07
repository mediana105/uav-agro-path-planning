from dataclasses import dataclass

from shapely.geometry import Polygon


@dataclass
class ZoneResult:
    """
    Results of mission planning for a single drone zone.

    Contains the assigned zone, optimal flight parameters, and performance metrics.
    """
    drone_id: int
    zone_polygon: Polygon
    optimal_angle: float
    path: list[tuple[float, float]]
    total_time: float
    rth_count: int = 0  # number of refuel/recharge stops


@dataclass
class MissionResult:
    """
    Complete mission planning results for all drones.

    Aggregates individual zone results and provides overall mission metrics.
    """
    zones: list[ZoneResult]

    @property
    def mission_time(self) -> float:
        return max(z.total_time for z in self.zones)
