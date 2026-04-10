import math
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from shapely.geometry import Polygon
from shapely.geometry import Polygon as ShapelyPolygon

from ..optimization.joint_optimizer import JointOptimizer
from ..pipeline import DroneConfig, MissionResult
from ..pipeline.pipeline import MissionOptimizer, calculate_portions_rth_aware

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
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.2,
                                    "mutation_scale": 12}, zorder=4)


def _rth_legs(path: list, start_pos: tuple, eps: float = 1e-3):
    legs = []
    sx, sy = start_pos
    n = len(path)
    i = 0
    while i < n - 3:
        p0, p1, p2, p3 = path[i], path[i + 1], path[i + 2], path[i + 3]
        if (math.isnan(p0[0])
                and not math.isnan(p1[0])
                and math.hypot(p1[0] - sx, p1[1] - sy) <= eps
                and math.isnan(p2[0])
                and not math.isnan(p3[0])):
            depart = next(
                (path[k] for k in range(i - 1, -1, -1) if not math.isnan(path[k][0])),
                None,
            )
            if depart is not None:
                legs.append((depart, p1, p3))
            i += 3
            continue
        i += 1
    return legs


class MissionVisualizer:
    def __init__(self, field_polygon: Polygon, drones: list[DroneConfig], cell_size: float = 1.0) -> None:
        self.field_polygon = field_polygon
        self.drones = drones
        self.cell_size = cell_size
        self._optimizer = MissionOptimizer(field_polygon, drones, cell_size)

    def visualize(
        self,
        sa_iterations: int = 100,
        sa_seed: int | None = 42,
        algorithm: str = "sa",
        fig_size: tuple[int, int] = (16, 8),
        show: bool = True,
        save_path: str | None = None,
    ) -> tuple[Figure, Any]:
        initial_portions, initial_result = self._run_initial()

        algo_label = "Tabu Search" if algorithm == "tabu" else "SA"

        fig, (ax_init, ax_opt) = plt.subplots(1, 2, figsize=fig_size)
        fig.suptitle("UAV Agricultural Mission Planning", fontsize=14, fontweight="bold")

        self.draw_panel(ax_init, initial_result, initial_portions,
                        title="Initial Decomposition\n(productivity‑based portions)")
        plt.tight_layout()
        plt.ion()
        plt.show()
        plt.pause(0.05)

        joint = JointOptimizer(self._optimizer)
        _best_seen = [float("inf")]

        def _on_iter(iteration, cur_portions, cur_value,
                     best_portions, best_value, temp, accepted):
            if best_value < _best_seen[0]:
                _best_seen[0] = best_value
                result = joint.get_last_mission_result()
                ax_opt.cla()
                if algorithm == "tabu":
                    iter_title = (f"Tabu Search — iter {iteration + 1} / {sa_iterations}\n"
                                  f"t_best = {best_value:.1f} s")
                else:
                    iter_title = (f"SA — iter {iteration + 1} / {sa_iterations}\n"
                                  f"t_best = {best_value:.1f} s   T = {temp:.2f}")
                self.draw_panel(ax_opt, result, list(best_portions), title=iter_title)
                fig.canvas.draw()
                plt.pause(0.02)

        meta = joint.optimize(
            initial_portions=initial_portions,
            algorithm=algorithm,
            max_iterations=sa_iterations,
            seed=sa_seed,
            iteration_callback=_on_iter,
        )
        opt_portions = meta["optimized_portions"]
        opt_result = meta["final_mission_result"]

        plt.ioff()

        ax_opt.cla()
        self.draw_panel(ax_opt, opt_result, opt_portions,
                        title=f"{algo_label}‑Optimized Decomposition\n({meta['iterations']} iterations)")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig, opt_result

    def _run_initial(self) -> tuple[list[float], MissionResult]:
        portions = calculate_portions_rth_aware(self.drones, self.field_polygon)
        result = self._optimizer.evaluate(portions)
        return portions, result

    def draw_panel(self, ax: plt.Axes, result: MissionResult,
                   portions: list[float], title: str) -> None:
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

                for depart, home, resume in _rth_legs(zone_result.path, drone.start_position):
                    ax.plot([depart[0], home[0]], [depart[1], home[1]],
                            "--", color=color, linewidth=1.4, alpha=0.55, zorder=3)
                    ax.plot([home[0], resume[0]], [home[1], resume[1]],
                            "--", color=color, linewidth=1.4, alpha=0.55, zorder=3)
                    ax.plot(home[0], home[1], "D", color=color, markersize=8,
                            markeredgecolor="black", markeredgewidth=0.7, zorder=7)

            sx, sy = drone.start_position
            ax.plot(sx, sy, "*", color=color, markersize=14,
                    markeredgecolor="black", markeredgewidth=0.5, zorder=5)
            ax.plot(sx, sy, "o", color="none", markersize=22,
                    markeredgecolor=color, markeredgewidth=2.0, zorder=4)
            ax.text(sx, sy - 6, f"CS{drone.id}", ha="center", va="top",
                    fontsize=7, color=color, fontweight="bold", zorder=6)

            rth_label = f"  |  RTH: {zone_result.rth_count}x" if zone_result.rth_count else ""
            label = f"Drone {drone.id}  ({portions[i]:.1%})  t = {zone_result.total_time:.1f} s{rth_label}"
            legend_handles.append(mpatches.Patch(color=color, label=label))

        ax.set_title(f"{title}\nMission time: {result.mission_time:.1f} s", fontsize=10)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    def simulate(self, result: MissionResult, interval: int = 50,
                 speed_factor: float = 1.0,
                 fig_size: tuple[int, int] = (12, 8),
                 save_path: str | None = None) -> FuncAnimation:
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

        for i, drone in enumerate(self.drones):
            color = _DRONE_COLORS[i % len(_DRONE_COLORS)]
            sx, sy = drone.start_position
            ax.plot(sx, sy, "*", color=color, markersize=14,
                    markeredgecolor="black", markeredgewidth=0.5, zorder=5)
            ax.plot(sx, sy, "o", color="none", markersize=22,
                    markeredgecolor=color, markeredgewidth=2.0, zorder=4)
            ax.text(sx, sy - 6, f"CS{drone.id}", ha="center", va="top",
                    fontsize=7, color=color, fontweight="bold", zorder=6)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("UAV Flight Simulation")
        ax.grid(True, alpha=0.3)

        _REFUEL_PAUSE = 5.0 / speed_factor

        drone_keyframes = []
        for _idx, (zone_result, drone) in enumerate(zip(result.zones, self.drones, strict=False)):
            speed = drone.speed * speed_factor
            # keyframe: (t, x, y, is_rth)
            keyframes = [(0.0, *drone.start_position, False)]
            t = 0.0
            prev = drone.start_position
            sx, sy = drone.start_position
            path_pts = zone_result.path or []
            n_pts = len(path_pts)
            pi = 0
            while pi < n_pts:
                x, y = path_pts[pi]
                if math.isnan(x):
                    if (pi + 3 < n_pts
                            and not math.isnan(path_pts[pi + 1][0])
                            and math.hypot(path_pts[pi + 1][0] - sx,
                                           path_pts[pi + 1][1] - sy) < 1e-3
                            and math.isnan(path_pts[pi + 2][0])
                            and not math.isnan(path_pts[pi + 3][0])):
                        hx, hy = path_pts[pi + 1]
                        rx, ry = path_pts[pi + 3]
                        t += math.hypot(hx - prev[0], hy - prev[1]) / speed
                        keyframes.append((t, hx, hy, True))
                        t += _REFUEL_PAUSE
                        keyframes.append((t, hx, hy, True))
                        t += math.hypot(rx - hx, ry - hy) / speed
                        keyframes.append((t, rx, ry, False))
                        prev = (rx, ry)
                        pi += 4
                        continue
                    else:
                        t += drone.turn_time
                        keyframes.append((t, *prev, False))
                else:
                    dist = math.hypot(x - prev[0], y - prev[1])
                    t += dist / speed
                    keyframes.append((t, x, y, False))
                    prev = (x, y)
                pi += 1
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
                            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7})

        def _pos_at(keyframes, t):
            """Return (x, y, is_rth) interpolated at time t."""
            if not keyframes:
                return None
            if t <= keyframes[0][0]:
                return keyframes[0][1], keyframes[0][2], keyframes[0][3]
            if t >= keyframes[-1][0]:
                return keyframes[-1][1], keyframes[-1][2], keyframes[-1][3]
            for j in range(len(keyframes) - 1):
                t0, x0, y0, rth0 = keyframes[j]
                t1, x1, y1, rth1 = keyframes[j + 1]
                if t0 <= t <= t1:
                    if math.isclose(t0, t1):
                        return x1, y1, rth1
                    a = (t - t0) / (t1 - t0)
                    return x0 + a * (x1 - x0), y0 + a * (y1 - y0), rth0
            return keyframes[-1][1], keyframes[-1][2], keyframes[-1][3]

        def update(frame):
            cur_t = frame / fps
            time_text.set_text(f"t = {cur_t:.1f} s")
            for idx, (marker, trail, kf) in enumerate(zip(drone_markers,  # noqa: B007
                                                          drone_trails,
                                                          drone_keyframes,
                                                          strict=False)):
                pos = _pos_at(kf, cur_t)
                if pos is None:
                    continue
                px, py, is_rth = pos
                marker.set_data([px], [py])

                if is_rth:
                    marker.set_marker("s")
                    marker.set_markersize(10)
                    marker.set_alpha(0.65)
                else:
                    marker.set_marker("^")
                    marker.set_markersize(12)
                    marker.set_alpha(1.0)

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
            print(f"[video] Video {n_frames} cadrs → {save_path}")

            def _progress(current_frame, total_frames):
                if current_frame % max(1, total_frames // 20) == 0:
                    pct = current_frame / total_frames * 100
                    bar = "#" * int(pct // 5) + "." * (20 - int(pct // 5))
                    print(f"\r[video] [{bar}] {pct:5.1f}%  ({current_frame}/{total_frames})",
                          end="", flush=True)

            anim.save(save_path, fps=int(fps), progress_callback=_progress)
            print(f"\r[video] [####################] 100.0%  ({n_frames}/{n_frames})")
            print(f"[video] done → {save_path}")

        plt.tight_layout()
        return anim
