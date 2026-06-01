import math

from shapely.geometry import Polygon

from .drone_config import DroneConfig
from .mission_result import MissionResult, ZoneResult
from .steps.decomposition_step import DecompositionStep
from .steps.trajectory_step import TrajectoryStep
from .steps.visibility_step import VisibilityStep


def calculate_portions_rth_aware(
    drone_configs: list[DroneConfig], field_polygon: Polygon
) -> list[float]:
    centroid = field_polygon.centroid

    effective_productivity = []
    for drone in drone_configs:
        if (
            drone.substance_rate > 0
            and math.isfinite(drone.tank_volume)
            and drone.speed > 0
        ):
            flight_time_min = drone.tank_volume / drone.substance_rate
            flight_time_sec = flight_time_min * 60.0
            sortie_by_substance = drone.speed * flight_time_sec
        else:
            sortie_by_substance = math.inf

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
            speed_eff = drone.speed

        effective_productivity.append(drone.swath_width * speed_eff)

    total = sum(effective_productivity)
    if total == 0:
        return [1.0 / len(drone_configs)] * len(drone_configs)

    return [p / total for p in effective_productivity]


class MissionPlanner:
    def __init__(
        self,
        field_polygon: Polygon,
        drones: list[DroneConfig],
        cell_size: float = 1.0,
    ):
        self.field_polygon = field_polygon
        self.drones = drones
        self.num_drones = len(drones)
        self.cell_size = cell_size

        self.initial_positions = [drone.start_position for drone in drones]

        self._decomposition = DecompositionStep(
            field_polygon=field_polygon,
            initial_positions=self.initial_positions,
            cell_size=cell_size,
        )
        self._trajectory = TrajectoryStep(field_polygon=field_polygon)
        self._visibility = VisibilityStep()

    def evaluate(
        self, portions: list[float] | None = None, exact: bool = False
    ) -> MissionResult:
        if portions is None:
            portions = calculate_portions_rth_aware(self.drones, self.field_polygon)
        zones = self._decomposition.run(portions)
        trajectories = self._trajectory.run(
            zones=zones,
            drone_ids=[d.id for d in self.drones],
            swath_widths=[d.swath_width for d in self.drones],
            start_positions=self.initial_positions,
            exact=exact,
        )
        visibility_results, T_max = self._visibility.run(
            trajectories=trajectories,
            drones=self.drones,
            field_polygon=self.field_polygon,
        )

        zone_results = []
        for vis in visibility_results:
            traj = next(t for t in trajectories if t.drone_id == vis.drone_id)
            zone_results.append(
                ZoneResult(
                    drone_id=vis.drone_id,
                    zone_polygon=traj.zone_polygon,
                    optimal_angle=traj.optimal_angle,
                    path=vis.path,
                    total_time=vis.total_time,
                    rth_count=vis.rth_count,
                )
            )

        return MissionResult(zones=zone_results)
