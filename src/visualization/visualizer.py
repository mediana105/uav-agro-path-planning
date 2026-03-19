from typing import List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from src.optimization.joint_optimizer import JointOptimizer
from src.pipeline import DroneConfig, MissionResult
from src.pipeline.pipeline import MissionOptimizer, calculate_portions_by_productivity

_DRONE_COLORS = [
    "#2196F3",  # blue
    "#F44336",  # red
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
    "#E91E63",  # pink
    "#8BC34A",  # light green
]


class MissionVisualizer:
    def __init__(
        self,
        field_polygon: Polygon,
        drones: List[DroneConfig],
        cell_size: float = 1.0,
    ) -> None:
        self.field_polygon = field_polygon
        self.drones = drones
        self.cell_size = cell_size
        self._optimizer = MissionOptimizer(field_polygon, drones, cell_size)


    def visualize(
        self,
        sa_iterations: int = 100,
        sa_seed: Optional[int] = 42,
        fig_size: Tuple[int, int] = (16, 8),
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        initial_portions, initial_result = self._run_initial()
        sa_portions, sa_result, sa_meta = self._run_sa(
            initial_portions, sa_iterations, sa_seed
        )

        fig, (ax_init, ax_sa) = plt.subplots(1, 2, figsize=fig_size)

        self.draw_panel(
            ax_init,
            initial_result,
            initial_portions,
            title=(
                "Initial Decomposition\n"
                "(productivity-based portions)"
            ),
        )
        self.draw_panel(
            ax_sa,
            sa_result,
            sa_portions,
            title=(
                f"SA-Optimized Decomposition\n"
                f"({sa_meta['iterations']} iterations)"
            ),
        )

        fig.suptitle("UAV Agricultural Mission Planning", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig


    def _run_initial(self) -> Tuple[List[float], MissionResult]:
        portions = calculate_portions_by_productivity(self.drones)
        result = self._optimizer.evaluate(portions)
        return portions, result

    def _run_sa(
        self,
        initial_portions: List[float],
        max_iterations: int,
        seed: Optional[int],
    ) -> Tuple[List[float], MissionResult, dict]:
        joint = JointOptimizer(self._optimizer)
        meta = joint.optimize(
            initial_portions=initial_portions,
            max_iterations=max_iterations,
            seed=seed,
        )
        return meta["optimized_portions"], meta["final_mission_result"], meta

    def draw_panel(
        self,
        ax: plt.Axes,
        result: MissionResult,
        portions: List[float],
        title: str,
    ) -> None:
        fx, fy = self.field_polygon.exterior.xy
        ax.fill(fx, fy, alpha=0.06, color="gray")
        ax.plot(fx, fy, "k-", linewidth=2)

        for interior in self.field_polygon.interiors:
            hx, hy = interior.xy
            ax.fill(hx, hy, color="white", zorder=2)
            ax.plot(hx, hy, "k--", linewidth=1.5, zorder=3)

        legend_handles = []

        for i, zone_result in enumerate(result.zones):
            color = _DRONE_COLORS[i % len(_DRONE_COLORS)]
            drone = self.drones[i]

            if not zone_result.zone_polygon.is_empty:
                zx, zy = zone_result.zone_polygon.exterior.xy
                ax.fill(zx, zy, alpha=0.22, color=color)
                ax.plot(zx, zy, "-", color=color, linewidth=1.2)

                for interior in zone_result.zone_polygon.interiors:
                    ix, iy = interior.xy
                    ax.fill(ix, iy, alpha=1.0, color="white")
                    ax.plot(ix, iy, "-", color=color, linewidth=1.0)

            if zone_result.path:
                import math as _math
                px = [p[0] for p in zone_result.path]
                py = [p[1] for p in zone_result.path]
                ax.plot(px, py, "-", color=color, linewidth=0.9, alpha=0.80)
                first_real = next(
                    (p for p in zone_result.path if not _math.isnan(p[0])), None
                )
                if first_real:
                    ax.plot(first_real[0], first_real[1], "o", color=color,
                            markersize=5, markeredgecolor="white", markeredgewidth=0.6)
                arrow_step = max(2, len(zone_result.path) // 12)
                for ai in range(0, len(zone_result.path) - 1, arrow_step):
                    x1, y1 = zone_result.path[ai]
                    x2, y2 = zone_result.path[ai + 1]
                    if _math.isnan(x1) or _math.isnan(x2):
                        continue
                    ax.annotate(
                        "", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color=color,
                            lw=1.2,
                            mutation_scale=10,
                        ),
                        zorder=4,
                    )
            sx, sy = drone.start_position
            ax.plot(
                sx, sy, "*",
                color=color, markersize=14,
                markeredgecolor="black", markeredgewidth=0.5,
                zorder=5,
            )

            label = (
                f"Drone {drone.id}  "
                f"({portions[i]:.1%})  "
                f"t = {zone_result.total_time:.1f} s"
            )
            legend_handles.append(mpatches.Patch(color=color, label=label))

        ax.set_title(
            f"{title}\nMission time: {result.mission_time:.1f} s",
            fontsize=10,
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)