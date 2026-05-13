import math
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry import Polygon as ShapelyPolygon

from ..optimization.joint_optimizer import JointOptimizer
from ..path_planning.boustrophedon import decompose_field
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

_FIG_BG = "#f5f4ef"
_AX_BG = "#fcfbf7"
_FIELD_FILL = "#e6e1d3"
_FIELD_EDGE = "#2c2c2c"
_HOLE_EDGE = "#4d4d4d"
_RTH_DASH = (0, (5, 4))


def _is_nan_point(pt) -> bool:
    if not isinstance(pt, (tuple, list)):
        return False
    if len(pt) >= 2 and math.isnan(pt[0]):
        return True
    return False


def _draw_pass_arrows(
    ax: plt.Axes, path: list, optimal_angle: float, color: str
) -> None:
    pass_dir = math.pi / 2 + optimal_angle
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        if _is_nan_point((x1, y1)) or _is_nan_point((x2, y2)):
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
        ax.annotate(
            "",
            xy=(mx + dx, my + dy),
            xytext=(mx - dx, my - dy),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": 1.2,
                "mutation_scale": 12,
            },
            zorder=4,
        )


def _zone_polygon_parts(zone_polygon: Polygon | MultiPolygon) -> list[Polygon]:
    if zone_polygon.is_empty:
        return []
    gt = zone_polygon.geom_type
    if gt == "Polygon":
        return [zone_polygon]
    if gt == "MultiPolygon":
        return list(zone_polygon.geoms)
    if gt == "GeometryCollection":
        return [g for g in zone_polygon.geoms if isinstance(g, Polygon)]
    return []


def _rth_legs(path: list, start_pos: tuple, eps: float = 1e-3):
    legs = []
    sx, sy = start_pos
    n = len(path)
    i = 0
    while i < n - 3:
        p0, p1, p2, p3 = path[i], path[i + 1], path[i + 2], path[i + 3]
        if any(isinstance(p, tuple) and len(p) == 3 for p in (p0, p1, p2, p3)):
            i += 1
            continue
        if (
            _is_nan_point(p0)
            and not _is_nan_point(p1)
            and math.hypot(p1[0] - sx, p1[1] - sy) <= eps
            and _is_nan_point(p2)
            and not _is_nan_point(p3)
        ):
            depart = next(
                (path[k] for k in range(i - 1, -1, -1) if not _is_nan_point(path[k])),
                None,
            )
            if depart is not None:
                legs.append((depart, p1, p3))
            i += 3
            continue
        i += 1
    return legs


def _coverage_segments_and_rth_legs(path: list, start_pos: tuple, eps: float = 1e-3):
    coverage_segments: list[list[tuple[float, float]]] = []
    rth_legs = []
    current: list[tuple[float, float]] = []
    sx, sy = start_pos
    n = len(path)
    i = 0
    while i < n:
        if (
            i + 3 < n
            and _is_nan_point(path[i])
            and not _is_nan_point(path[i + 1])
            and math.hypot(path[i + 1][0] - sx, path[i + 1][1] - sy) <= eps
            and _is_nan_point(path[i + 2])
            and not _is_nan_point(path[i + 3])
        ):
            depart = current[-1] if current else None
            home = path[i + 1]
            resume = path[i + 3]
            if len(current) >= 2:
                coverage_segments.append(current)
            if depart is not None:
                rth_legs.append((depart, home, resume))
            current = [resume]
            i += 4
            continue

        pt = path[i]
        if _is_nan_point(pt):
            if len(current) >= 2:
                coverage_segments.append(current)
            current = []
            i += 1
            continue

        current.append((pt[0], pt[1]))
        i += 1

    if len(current) >= 2:
        coverage_segments.append(current)
    return coverage_segments, rth_legs


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(_AX_BG)
    ax.grid(True, alpha=0.16, color="#6c757d", linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)


def _draw_field_base(ax: plt.Axes, field_polygon: Polygon) -> None:
    fx, fy = field_polygon.exterior.xy
    ax.fill(fx, fy, alpha=1.0, color=_FIELD_FILL, zorder=0)
    ax.plot(fx, fy, color=_FIELD_EDGE, linewidth=2.2, zorder=1)
    for interior in field_polygon.interiors:
        hx, hy = interior.xy
        ax.fill(hx, hy, color="white", zorder=2)
        ax.plot(hx, hy, color=_HOLE_EDGE, linewidth=1.5, linestyle=_RTH_DASH, zorder=3)


def _draw_swath_ribbon(
    ax: plt.Axes,
    segment: list[tuple[float, float]],
    color: str,
    swath_width: float,
    *,
    zorder: float,
    alpha_scale: float = 1.0,
) -> None:
    if len(segment) < 2:
        return
    xs = [p[0] for p in segment]
    ys = [p[1] for p in segment]
    ribbon_lw = max(6.0, min(18.0, swath_width * 1.8))
    ax.plot(
        xs,
        ys,
        "-",
        color=color,
        linewidth=ribbon_lw,
        alpha=0.10 * alpha_scale,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=zorder,
    )


def _draw_segment_arrows(
    ax: plt.Axes,
    segment: list[tuple[float, float]],
    color: str,
    *,
    zorder: float,
) -> None:
    if len(segment) < 3:
        return
    step = max(3, len(segment) // 4)
    for i in range(step - 1, len(segment) - 1, step):
        x1, y1 = segment[i]
        x2, y2 = segment[i + 1]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 4.0:
            continue
        trim = min(length * 0.22, 4.0)
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        ax.annotate(
            "",
            xy=(x2 - ux * trim * 0.15, y2 - uy * trim * 0.15),
            xytext=(x2 - ux * trim, y2 - uy * trim),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": 1.2,
                "mutation_scale": 10,
                "alpha": 0.95,
            },
            zorder=zorder,
        )


def _auto_fig_size(
    field_polygon: Polygon,
    base_size: tuple[float, float],
    *,
    wide_threshold: float = 4.5,
) -> tuple[float, float]:
    min_x, min_y, max_x, max_y = field_polygon.bounds
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    scale = max(1.15, min(max(width, height) / 180.0, 2.0))
    fig_w = base_size[0] * scale
    fig_h = base_size[1] * scale
    if width / height >= wide_threshold:
        fig_h = max(fig_h * 1.7, base_size[1] * 1.7)
        fig_w = max(fig_w * 1.05, base_size[0])
    return fig_w, fig_h


class MissionVisualizer:
    def __init__(
        self,
        field_polygon: Polygon,
        drones: list[DroneConfig],
        cell_size: float = 1.0,
        strategy: str = "greedy_safe",
    ) -> None:
        self.field_polygon = field_polygon
        self.drones = drones
        self.cell_size = cell_size
        self._optimizer = MissionOptimizer(
            field_polygon,
            drones,
            cell_size,
            strategy=strategy,
        )

    def visualize(
        self,
        sa_iterations: int = 100,
        sa_seed: int | None = 42,
        algorithm: str = "sa",
        fig_size: tuple[int, int] | None = None,
        show: bool = True,
        save_path: str | None = None,
    ) -> tuple[Figure, Any]:
        initial_portions, initial_result = self._run_initial()
        algo_label = "Tabu Search" if algorithm == "tabu" else "SA"

        fig_size = fig_size or (18.0, 10.0)
        min_x, min_y, max_x, max_y = self.field_polygon.bounds
        wide_field = ((max_x - min_x) / max(max_y - min_y, 1.0)) >= 4.5

        if wide_field:
            adaptive_fig_size = _auto_fig_size(self.field_polygon, fig_size)
            fig, (ax_init, ax_opt) = plt.subplots(2, 1, figsize=adaptive_fig_size)
            plt.subplots_adjust(hspace=0.38, bottom=0.09, top=0.92)
        else:
            adaptive_fig_size = _auto_fig_size(self.field_polygon, fig_size)
            fig, (ax_init, ax_opt) = plt.subplots(1, 2, figsize=adaptive_fig_size)
            plt.subplots_adjust(bottom=0.30)
        fig.patch.set_facecolor(_FIG_BG)
        fig.suptitle(
            "UAV Agricultural Mission Planning", fontsize=15, fontweight="bold"
        )

        self.draw_panel(
            ax_init,
            initial_result,
            initial_portions,
            title="Initial Decomposition\n(productivity‑based portions)",
        )

        joint = JointOptimizer(self._optimizer)
        meta = joint.optimize(
            initial_portions=initial_portions,
            algorithm=algorithm,
            max_iterations=sa_iterations,
            seed=sa_seed,
        )
        opt_portions = meta["optimized_portions"]
        opt_result = meta["final_mission_result"]

        self.draw_panel(
            ax_opt,
            opt_result,
            opt_portions,
            title=f"{algo_label}‑Optimized Decomposition\n({meta['iterations']} iterations)",
        )

        if save_path:
            fig.savefig(save_path, dpi=190, bbox_inches="tight")
        if show:
            plt.show()
        return fig, opt_result

    def _run_initial(self) -> tuple[list[float], MissionResult]:
        portions = calculate_portions_rth_aware(self.drones, self.field_polygon)
        result = self._optimizer.evaluate(portions)
        return portions, result

    def draw_panel(
        self, ax: plt.Axes, result: MissionResult, portions: list[float], title: str
    ) -> None:
        _style_axes(ax)
        _draw_field_base(ax, self.field_polygon)

        legend_handles = []
        for i, zone_result in enumerate(result.zones):
            color = _DRONE_COLORS[i % len(_DRONE_COLORS)]
            drone = self.drones[i]

            if not zone_result.zone_polygon.is_empty:
                if hasattr(zone_result.zone_polygon, "geoms"):
                    sub_polys = [
                        g
                        for g in zone_result.zone_polygon.geoms
                        if isinstance(g, ShapelyPolygon)
                    ]
                else:
                    sub_polys = [zone_result.zone_polygon]
                for sub in sub_polys:
                    zx, zy = sub.exterior.xy
                    ax.fill(zx, zy, alpha=0.18, color=color, zorder=1.5)
                    ax.plot(zx, zy, "-", color=color, linewidth=1.1, alpha=0.9, zorder=2)
                    for interior in sub.interiors:
                        ix, iy = interior.xy
                        ax.fill(ix, iy, alpha=1.0, color="white")
                        ax.plot(ix, iy, "-", color=color, linewidth=0.9, alpha=0.8)

            if zone_result.path:
                coverage_segments, rth_legs = _coverage_segments_and_rth_legs(
                    zone_result.path,
                    drone.start_position,
                )
                for seg in coverage_segments:
                    _draw_swath_ribbon(
                        ax,
                        seg,
                        color,
                        drone.swath_width,
                        zorder=3.6,
                        alpha_scale=1.0,
                    )
                    ax.plot(
                        [p[0] for p in seg],
                        [p[1] for p in seg],
                        "-",
                        color=color,
                        linewidth=1.4,
                        alpha=0.92,
                        solid_capstyle="round",
                        solid_joinstyle="round",
                        zorder=4,
                    )
                    _draw_segment_arrows(ax, seg, color, zorder=4.4)

                first_real = next(
                    (p for p in zone_result.path if not _is_nan_point(p)), None
                )
                if first_real:
                    ax.plot(
                        first_real[0],
                        first_real[1],
                        "o",
                        color=color,
                        markersize=6,
                        markeredgecolor="white",
                        markeredgewidth=0.8,
                        zorder=6,
                    )

                for depart, home, resume in rth_legs:
                    ax.plot(
                        [depart[0], home[0]],
                        [depart[1], home[1]],
                        linestyle=_RTH_DASH,
                        color=color,
                        linewidth=1.8,
                        alpha=0.75,
                        zorder=5,
                    )
                    ax.plot(
                        [home[0], resume[0]],
                        [home[1], resume[1]],
                        linestyle=_RTH_DASH,
                        color=color,
                        linewidth=1.8,
                        alpha=0.75,
                        zorder=5,
                    )
                    ax.plot(
                        home[0],
                        home[1],
                        "D",
                        color=color,
                        markersize=7,
                        markeredgecolor="black",
                        markeredgewidth=0.7,
                        zorder=7,
                    )

            sx, sy = drone.start_position
            ax.plot(
                sx,
                sy,
                "*",
                color=color,
                markersize=13,
                markeredgecolor="black",
                markeredgewidth=0.5,
                zorder=7,
            )
            ax.plot(
                sx,
                sy,
                "o",
                color="none",
                markersize=19,
                markeredgecolor=color,
                markeredgewidth=2.2,
                zorder=6,
            )
            ax.text(
                sx,
                sy - 6,
                f"CS{drone.id}",
                ha="center",
                va="top",
                fontsize=7,
                color=color,
                fontweight="bold",
                zorder=6,
            )

            rth_label = (
                f"  |  RTH: {zone_result.rth_count}x" if zone_result.rth_count else ""
            )
            label = f"Drone {drone.id}  ({portions[i]:.1%})  t = {zone_result.total_time:.1f} s{rth_label}"
            legend_handles.append(mpatches.Patch(color=color, label=label))

        ax.set_title(f"{title}\nMission time: {result.mission_time:.1f} s", fontsize=10)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal", adjustable="box")

        # Легенда снизу, ещё ниже, чтобы не перекрывать подписи осей и сетку
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.3),
            fontsize=8,
            ncol=2,
            frameon=True,
            fancybox=True,
            shadow=True,
        )
        ax.text(
            0.99,
            0.02,
            "Solid = coverage path   Dashed = RTH/refuel legs",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            color="#444444",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#fffdf8", "alpha": 0.9, "edgecolor": "none"},
        )
        ax.margins(x=0.03, y=0.08)

    def save_bcd_decomposition_figure(
        self,
        result: MissionResult,
        save_path: str,
        fig_size: tuple[float, float] | None = None,
    ) -> None:
        fig_size = fig_size or (16.0, 12.0)
        fig, ax = plt.subplots(figsize=_auto_fig_size(self.field_polygon, fig_size))
        fig.patch.set_facecolor(_FIG_BG)
        _style_axes(ax)
        _draw_field_base(ax, self.field_polygon)

        total_cells = 0
        for i, zone_result in enumerate(result.zones):
            color = _DRONE_COLORS[i % len(_DRONE_COLORS)]
            drone = self.drones[i]
            swath = drone.swath_width
            angle_deg = math.degrees(zone_result.optimal_angle)

            _, cells, _safe = decompose_field(
                zone_result.zone_polygon,
                swath,
                angle_deg,
                coalesce_to=drone.bcd_coalesce_to,
            )
            total_cells += len(cells)
            for ci, cell in enumerate(cells):
                poly = cell.poly
                if poly.is_empty:
                    continue
                gx, gy = poly.exterior.xy
                ax.fill(
                    gx,
                    gy,
                    facecolor=color,
                    alpha=0.16,
                    edgecolor=color,
                    linewidth=1.0,
                    zorder=4,
                )
                cx, cy = poly.centroid.x, poly.centroid.y
                ax.text(
                    cx,
                    cy,
                    f"{drone.id}:{ci}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=color,
                    fontweight="bold",
                    zorder=5,
                )

        ax.set_title(
            "BCD (spray corridor = zone buffer −swath/2, aligned to mission angle)\n"
            f"Total cells: {total_cells}",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal", adjustable="box")
        leg = [
            mpatches.Patch(
                color=_DRONE_COLORS[j % len(_DRONE_COLORS)],
                label=f"Drone {d.id}",
            )
            for j, d in enumerate(self.drones)
        ]
        ax.legend(handles=leg, loc="upper right", fontsize=8, framealpha=0.9)
        fig.savefig(save_path, dpi=190, bbox_inches="tight")
        plt.close(fig)

    def save_bcd_adjacency_figure(
        self,
        result: MissionResult,
        save_path: str,
        fig_size: tuple[float, float] | None = None,
    ) -> None:
        fig_size = fig_size or (16.0, 12.0)
        fig, ax = plt.subplots(figsize=_auto_fig_size(self.field_polygon, fig_size))
        fig.patch.set_facecolor(_FIG_BG)
        _style_axes(ax)
        _draw_field_base(ax, self.field_polygon)

        total_cells = 0
        total_edges = 0
        legend_handles = []

        for i, zone_result in enumerate(result.zones):
            color = _DRONE_COLORS[i % len(_DRONE_COLORS)]
            drone = self.drones[i]
            _, cells, _safe = decompose_field(
                zone_result.zone_polygon,
                drone.swath_width,
                math.degrees(zone_result.optimal_angle),
                coalesce_to=drone.bcd_coalesce_to,
            )

            total_cells += len(cells)
            edges = set()
            centers: dict[int, tuple[float, float]] = {}
            valid_ids = {cell.idx for cell in cells}
            for cell in cells:
                poly = cell.poly
                if poly.is_empty:
                    continue
                gx, gy = poly.exterior.xy
                ax.fill(
                    gx,
                    gy,
                    facecolor=color,
                    alpha=0.12,
                    edgecolor=color,
                    linewidth=0.9,
                    zorder=4,
                )
                centers[cell.idx] = (poly.centroid.x, poly.centroid.y)
                ax.text(
                    poly.centroid.x,
                    poly.centroid.y,
                    f"{drone.id}:{cell.idx}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=color,
                    fontweight="bold",
                    zorder=6,
                )
                for nb in cell.neighbours:
                    if nb in valid_ids:
                        edges.add(tuple(sorted((cell.idx, nb))))

            for a, b in sorted(edges):
                if a not in centers or b not in centers:
                    continue
                ax.plot(
                    [centers[a][0], centers[b][0]],
                    [centers[a][1], centers[b][1]],
                    color=color,
                    linewidth=1.8,
                    alpha=0.85,
                    zorder=5,
                )
            total_edges += len(edges)
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linewidth=2.0,
                    label=f"Drone {drone.id} ({len(cells)} cells, {len(edges)} edges)",
                )
            )

        ax.set_title(
            "BCD Adjacency Graph (final decomposition)\n"
            f"Total cells: {total_cells}, total adjacency edges: {total_edges}",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)
        fig.savefig(save_path, dpi=190, bbox_inches="tight")
        plt.close(fig)

    def simulate(
        self,
        result: MissionResult,
        interval: int = 50,
        speed_factor: float = 1.0,
        fig_size: tuple[int, int] | None = None,
        save_path: str | None = None,
    ) -> FuncAnimation:
        fig_size = fig_size or (14.0, 10.0)
        fig, ax = plt.subplots(figsize=_auto_fig_size(self.field_polygon, fig_size))
        fig.patch.set_facecolor(_FIG_BG)
        _style_axes(ax)
        _draw_field_base(ax, self.field_polygon)

        for i, zone_result in enumerate(result.zones):
            color = _DRONE_COLORS[i % len(_DRONE_COLORS)]
            if not zone_result.zone_polygon.is_empty:
                zx, zy = zone_result.zone_polygon.exterior.xy
                ax.fill(zx, zy, alpha=0.11, color=color, zorder=1.5)
                for interior in zone_result.zone_polygon.interiors:
                    ix, iy = interior.xy
                    ax.fill(ix, iy, color="white", zorder=2)
                    ax.plot(ix, iy, color=_HOLE_EDGE, linestyle=_RTH_DASH, linewidth=1.0, zorder=3)

            if zone_result.path:
                coverage_segments, rth_legs = _coverage_segments_and_rth_legs(
                    zone_result.path,
                    self.drones[i].start_position,
                )
                for seg in coverage_segments:
                    _draw_swath_ribbon(
                        ax,
                        seg,
                        color,
                        self.drones[i].swath_width,
                        zorder=2.8,
                        alpha_scale=0.75,
                    )
                    ax.plot(
                        [p[0] for p in seg],
                        [p[1] for p in seg],
                        "-",
                        color=color,
                        linewidth=1.0,
                        alpha=0.24,
                        zorder=3.2,
                    )
                for depart, home, resume in rth_legs:
                    ax.plot(
                        [depart[0], home[0]],
                        [depart[1], home[1]],
                        color=color,
                        linestyle=_RTH_DASH,
                        linewidth=1.1,
                        alpha=0.26,
                        zorder=3,
                    )
                    ax.plot(
                        [home[0], resume[0]],
                        [home[1], resume[1]],
                        color=color,
                        linestyle=_RTH_DASH,
                        linewidth=1.1,
                        alpha=0.26,
                        zorder=3,
                    )

        for i, drone in enumerate(self.drones):
            color = _DRONE_COLORS[i % len(_DRONE_COLORS)]
            sx, sy = drone.start_position
            ax.plot(
                sx,
                sy,
                "*",
                color=color,
                markersize=13,
                markeredgecolor="black",
                markeredgewidth=0.5,
                zorder=7,
            )
            ax.plot(
                sx,
                sy,
                "o",
                color="none",
                markersize=19,
                markeredgecolor=color,
                markeredgewidth=2.2,
                zorder=6,
            )
            ax.text(
                sx,
                sy - 6,
                f"CS{drone.id}",
                ha="center",
                va="top",
                fontsize=7,
                color=color,
                fontweight="bold",
                zorder=6,
            )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("UAV Flight Simulation")

        _REFUEL_PAUSE = 5.0 / speed_factor

        drone_keyframes = []
        for zone_result, drone in zip(result.zones, self.drones, strict=False):
            speed = drone.speed * speed_factor
            sx, sy = drone.start_position
            keyframes: list[tuple[float, float, float, bool]] = [(0.0, sx, sy, False)]
            t = 0.0
            prev = drone.start_position
            path_pts = zone_result.path or []
            n_pts = len(path_pts)
            pi = 0
            while pi < n_pts:
                pt = path_pts[pi]
                # пропускаем трёхэлементные маркеры
                if isinstance(pt, tuple) and len(pt) == 3:
                    pi += 1
                    continue
                x, y = pt
                if math.isnan(x):
                    if (
                        pi + 3 < n_pts
                        and not _is_nan_point(path_pts[pi + 1])
                        and math.hypot(
                            path_pts[pi + 1][0] - sx, path_pts[pi + 1][1] - sy
                        )
                        < 1e-3
                        and _is_nan_point(path_pts[pi + 2])
                        and not _is_nan_point(path_pts[pi + 3])
                    ):
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
                        px, py = prev
                        keyframes.append((t, px, py, False))
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
        drone_rth_trails = []
        trail_data = [{"x": [], "y": []} for _ in self.drones]
        rth_trail_data = [{"x": [], "y": []} for _ in self.drones]
        legend_handles = []

        for i, drone in enumerate(self.drones):
            color = _DRONE_COLORS[i % len(_DRONE_COLORS)]
            (marker,) = ax.plot(
                [],
                [],
                "^",
                color=color,
                markersize=12,
                markeredgecolor="black",
                markeredgewidth=0.5,
                zorder=6,
            )
            (trail,) = ax.plot(
                [], [], "-", color=color, linewidth=2.0, alpha=0.92, zorder=6,
                solid_capstyle="round", solid_joinstyle="round",
            )
            (rth_trail,) = ax.plot(
                [], [], color=color, linewidth=1.7, alpha=0.72, zorder=5,
                linestyle=_RTH_DASH,
            )
            drone_markers.append(marker)
            drone_trails.append(trail)
            drone_rth_trails.append(rth_trail)
            legend_handles.append(
                mpatches.Patch(color=color, label=f"Drone {drone.id}")
            )

        ax.legend(handles=legend_handles, loc="upper right", fontsize=9)
        time_text = ax.text(
            0.02,
            0.97,
            "t = 0.0 s",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
        )

        def _pos_at(keyframes, t):
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
            for idx, (marker, trail, kf) in enumerate(
                zip(drone_markers, drone_trails, drone_keyframes, strict=False)
            ):
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

                active = rth_trail_data[idx] if is_rth else trail_data[idx]
                passive = trail_data[idx] if is_rth else rth_trail_data[idx]

                if active["x"]:
                    lx, ly = active["x"][-1], active["y"][-1]
                    if math.hypot(px - lx, py - ly) < 1e-6:
                        active["x"].append(float("nan"))
                        active["y"].append(float("nan"))

                if passive["x"] and not math.isnan(passive["x"][-1]):
                    passive["x"].append(float("nan"))
                    passive["y"].append(float("nan"))

                active["x"].append(px)
                active["y"].append(py)
                trail.set_data(trail_data[idx]["x"], trail_data[idx]["y"])
                drone_rth_trails[idx].set_data(
                    rth_trail_data[idx]["x"], rth_trail_data[idx]["y"]
                )
            return drone_markers + drone_trails + drone_rth_trails + [time_text]

        anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=True)

        if save_path:
            print(f"[video] Video {n_frames} frames → {save_path}")

            def _progress(current_frame, total_frames):
                if current_frame % max(1, total_frames // 20) == 0:
                    pct = current_frame / total_frames * 100
                    bar = "#" * int(pct // 5) + "." * (20 - int(pct // 5))
                    print(
                        f"\r[video] [{bar}] {pct:5.1f}%  ({current_frame}/{total_frames})",
                        end="",
                        flush=True,
                    )

            anim.save(save_path, fps=int(fps), progress_callback=_progress)
            print(f"\r[video] [####################] 100.0%  ({n_frames}/{n_frames})")
            print(f"[video] done → {save_path}")

        plt.tight_layout()
        return anim
