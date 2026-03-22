import math
from typing import List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from shapely.geometry import Polygon, Polygon as ShapelyPolygon

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


def _draw_pass_arrows(ax: plt.Axes, path: list,
                      optimal_angle: float, color: str) -> None:
    pass_dir = math.pi / 2 + optimal_angle
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        if math.isnan(x1) or math.isnan(x2):
            continue
        seg_angle = math.atan2(y2 - y1, x2 - x1)
        diff = abs((seg_angle - pass_dir) % math.pi)
        diff = min(diff, math.pi - diff)
        if diff > math.pi / 4:
            continue
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 1e-9:
            continue
        eps = length * 0.15
        dx = (x2 - x1) / length * eps
        dy = (y2 - y1) / length * eps
        ax.annotate("", xy=(mx + dx, my + dy), xytext=(mx - dx, my - dy),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2,
                                    mutation_scale=12), zorder=4)


class MissionVisualizer:
    def __init__(self, field_polygon: Polygon, drones: List[DroneConfig], cell_size: float = 1.0) -> None:
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
        sa_portions, sa_result, sa_meta = self._run_sa(initial_portions, sa_iterations, sa_seed)
        fig, (ax_init, ax_sa) = plt.subplots(1, 2, figsize=fig_size)
        self.draw_panel(ax_init, initial_result, initial_portions,
                        title="Initial Decomposition\n(productivity‑based portions)")
        self.draw_panel(ax_sa, sa_result, sa_portions,
                        title=f"SA‑Optimized Decomposition\n({sa_meta['iterations']} iterations)")
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

    def _run_sa(self, initial_portions: List[float], max_iterations: int,
                seed: Optional[int]) -> Tuple[List[float], MissionResult, dict]:
        joint = JointOptimizer(self._optimizer)
        meta = joint.optimize(initial_portions=initial_portions,
                              max_iterations=max_iterations,
                              seed=seed)
        return meta["optimized_portions"], meta["final_mission_result"], meta

    def draw_panel(self, ax: plt.Axes, result: MissionResult,
                   portions: List[float], title: str) -> None:
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
                if zone_result.zone_polygon.geom_type == "GeometryCollection":
                    for sub in zone_result.zone_polygon.geoms:
                        if not isinstance(sub, ShapelyPolygon):
                            continue
                        zx, zy = sub.exterior.xy
                        ax.fill(zx, zy, alpha=0.22, color=color)
                        ax.plot(zx, zy, "-", color=color, linewidth=1.2)
                else:
                    zx, zy = zone_result.zone_polygon.exterior.xy
                    ax.fill(zx, zy, alpha=0.22, color=color)
                    ax.plot(zx, zy, "-", color=color, linewidth=1.2)
                for interior in zone_result.zone_polygon.interiors:
                    ix, iy = interior.xy
                    ax.fill(ix, iy, alpha=1.0, color="white")
                    ax.plot(ix, iy, "-", color=color, linewidth=1.0)

            if zone_result.path:
                px = [p[0] for p in zone_result.path]
                py = [p[1] for p in zone_result.path]
                ax.plot(px, py, "-", color=color, linewidth=0.9, alpha=0.80)
                first_real = next((p for p in zone_result.path if not math.isnan(p[0])), None)
                if first_real:
                    ax.plot(first_real[0], first_real[1], "o", color=color,
                            markersize=5, markeredgecolor="white", markeredgewidth=0.6)

            sx, sy = drone.start_position
            ax.plot(sx, sy, "*", color=color, markersize=14,
                    markeredgecolor="black", markeredgewidth=0.5, zorder=5)

            label = f"Drone {drone.id}  ({portions[i]:.1%})  t = {zone_result.total_time:.1f} s"
            legend_handles.append(mpatches.Patch(color=color, label=label))

        ax.set_title(f"{title}\nMission time: {result.mission_time:.1f} s", fontsize=10)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    def simulate(self, result: MissionResult, interval: int = 50,
                 speed_factor: float = 1.0,
                 fig_size: Tuple[int, int] = (12, 8),
                 save_path: Optional[str] = None) -> FuncAnimation:
        fig, ax = plt.subplots(figsize=fig_size)
        fx, fy = self.field_polygon.exterior.xy
        ax.fill(fx, fy, alpha=0.06, color="gray")
        ax.plot(fx, fy, "k-", linewidth=2)

        for i, zone_result in enumerate(result.zones):
            color = _DRONE_COLORS[i % len(_DRONE_COLORS)]
            if not zone_result.zone_polygon.is_empty:
                zx, zy = zone_result.zone_polygon.exterior.xy
                ax.fill(zx, zy, alpha=0.12, color=color)
                for interior in zone_result.zone_polygon.interiors:
                    ix, iy = interior.xy
                    ax.fill(ix, iy, color="white", zorder=2)
                    ax.plot(ix, iy, "k--", linewidth=1.0, zorder=3)

            if zone_result.path:
                seg_x, seg_y = [], []
                for pt in zone_result.path:
                    if math.isnan(pt[0]):
                        ax.plot(seg_x, seg_y, "-", color=color,
                                linewidth=0.7, alpha=0.3)
                        seg_x, seg_y = [], []
                    else:
                        seg_x.append(pt[0])
                        seg_y.append(pt[1])
                if seg_x:
                    ax.plot(seg_x, seg_y, "-", color=color,
                            linewidth=0.7, alpha=0.3)

        for interior in self.field_polygon.interiors:
            hx, hy = interior.xy
            ax.fill(hx, hy, color="white", zorder=4)
            ax.plot(hx, hy, "k--", linewidth=1.5, zorder=4)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("UAV Flight Simulation")
        ax.grid(True, alpha=0.3)

        drone_keyframes = []
        for idx, (zone_result, drone) in enumerate(zip(result.zones, self.drones)):
            speed = drone.speed * speed_factor
            keyframes = [(0.0, *drone.start_position)]
            t = 0.0
            prev = drone.start_position
            for pt in zone_result.path or []:
                x, y = pt
                if math.isnan(x):
                    t += drone.turn_time                     # пауза, спрей выключён
                    keyframes.append((t, *prev))
                    continue
                dist = math.hypot(x - prev[0], y - prev[1])
                t += dist / speed
                keyframes.append((t, x, y))
                prev = (x, y)
            drone_keyframes.append(keyframes)

        total_time = max(kf[-1][0] if kf else 0.0 for kf in drone_keyframes)
        fps = 1000.0 / interval
        n_frames = int(total_time * fps) + 1

        drone_markers = []
        drone_trails = []
        trail_data = [{"x": [], "y": []} for _ in self.drones]
        legend_handles = []

        for i, drone in enumerate(self.drones):
            color = _DRONE_COLORS[i % len(_DRONE_COLORS)]
            marker, = ax.plot([], [], "^", color=color, markersize=12,
                              markeredgecolor="black", markeredgewidth=0.5, zorder=6)
            trail, = ax.plot([], [], "-", color=color, linewidth=1.8, alpha=0.85, zorder=5)
            drone_markers.append(marker)
            drone_trails.append(trail)
            legend_handles.append(mpatches.Patch(color=color, label=f"Drone {drone.id}"))

        ax.legend(handles=legend_handles, loc="upper right", fontsize=9)
        time_text = ax.text(0.02, 0.97, "t = 0.0 s", transform=ax.transAxes,
                            fontsize=11, verticalalignment="top",
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

        def _pos_at(keyframes, t):
            if not keyframes:
                return None
            if t <= keyframes[0][0]:
                return keyframes[0][1], keyframes[0][2]
            if t >= keyframes[-1][0]:
                return keyframes[-1][1], keyframes[-1][2]
            for j in range(len(keyframes) - 1):
                t0, x0, y0 = keyframes[j]
                t1, x1, y1 = keyframes[j + 1]
                if t0 <= t <= t1:
                    if math.isclose(t0, t1):
                        return x1, y1
                    a = (t - t0) / (t1 - t0)
                    return x0 + a * (x1 - x0), y0 + a * (y1 - y0)
            return keyframes[-1][1], keyframes[-1][2]

        def update(frame):
            cur_t = frame / fps
            time_text.set_text(f"t = {cur_t:.1f} s")
            for idx, (marker, trail, kf) in enumerate(zip(drone_markers,
                                                          drone_trails,
                                                          drone_keyframes)):
                pos = _pos_at(kf, cur_t)
                if pos is None:
                    continue
                px, py = pos
                marker.set_data([px], [py])

                # если текущий кадр попал в паузу (координаты не меняются)
                if trail_data[idx]["x"]:
                    lx, ly = trail_data[idx]["x"][-1], trail_data[idx]["y"][-1]
                    if math.hypot(px - lx, py - ly) < 1e-6:
                        trail_data[idx]["x"].append(float("nan"))
                        trail_data[idx]["y"].append(float("nan"))

                trail_data[idx]["x"].append(px)
                trail_data[idx]["y"].append(py)
                trail.set_data(trail_data[idx]["x"], trail_data[idx]["y"])
            return drone_markers + drone_trails + [time_text]

        anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=True)

        if save_path:
            anim.save(save_path, fps=int(fps))

        plt.tight_layout()
        return anim
