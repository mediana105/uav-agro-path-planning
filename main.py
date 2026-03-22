#!/usr/bin/env python3

import argparse
import importlib

import matplotlib
import matplotlib.pyplot as plt

from src.optimization.joint_optimizer import JointOptimizer
from src.pipeline.pipeline import MissionOptimizer, calculate_portions_by_productivity
from src.visualization.visualizer import MissionVisualizer


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--field", type=str, default="default",
                   help="Field config name from fields/ directory (default: default)")
    p.add_argument("--iterations", type=int, default=100,
                   help="Number of SA iterations (default: 100)")
    p.add_argument("--save", type=str, default=None,
                   help="Save final figure to this path (e.g. result.png)")
    p.add_argument("--video", type=str, default=None,
                   help="Save flight simulation to this path (e.g. flight.mp4)")
    p.add_argument("--speed", type=float, default=5.0,
                   help="Playback speed factor for simulation (default: 5.0)")
    return p.parse_args()


def _set_backend():
    for backend in ("TkAgg", "Qt5Agg", "MacOSX", "Agg"):
        try:
            matplotlib.use(backend)
            return
        except Exception:
            continue


def main():
    args = _parse_args()
    _set_backend()

    cfg = importlib.import_module(f"fields.{args.field}")
    FIELD     = cfg.FIELD
    DRONES    = cfg.DRONES
    CELL_SIZE = getattr(cfg, "CELL_SIZE", 5.0)
    SA_SEED   = getattr(cfg, "SA_SEED", 42)

    optimizer = MissionOptimizer(FIELD, DRONES, cell_size=CELL_SIZE)
    vis       = MissionVisualizer(FIELD, DRONES, cell_size=CELL_SIZE)

    initial_portions = calculate_portions_by_productivity(DRONES)
    initial_result   = optimizer.evaluate(initial_portions)

    print(f"Initial mission time : {initial_result.mission_time:.1f} s")
    print(f"Initial portions     : {[f'{p:.3f}' for p in initial_portions]}")
    print(f"Running SA ({args.iterations} iterations)…\n")


    plt.ion()
    fig, (ax_init, ax_sa) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("UAV Agricultural Mission Planning", fontsize=14, fontweight="bold")

    vis.draw_panel(
        ax_init, initial_result, initial_portions,
        "Initial Decomposition",
    )

    vis.draw_panel(
        ax_sa, initial_result, initial_portions,
        f"SA optimising…  iter 0\n"
        f"Mission time: {initial_result.mission_time:.1f} s",
    )

    plt.tight_layout()
    plt.pause(0.5)

    joint     = JointOptimizer(optimizer)
    prev_best = [initial_result.mission_time]

    def on_iteration(iteration, current_portions, current_value,
                     best_portions, best_value, temperature, accepted):
        new_best = best_value < prev_best[0] - 1e-9

        if new_best:
            prev_best[0] = best_value
            best_result = joint.last_result

            ax_sa.clear()
            vis.draw_panel(
                ax_sa, best_result, best_portions,
                f"SA  iter {iteration + 1}  |  T = {temperature:.3f}\n"
                f"Mission time: {best_value:.1f} s  ↓ new best",
            )
            plt.tight_layout()
            plt.pause(0.01)

            print(f"  iter {iteration + 1:>4d}  T={temperature:9.4f}  "
                  f"best={best_value:.1f} s  "
                  f"portions={[f'{p:.3f}' for p in best_portions]}")

        elif (iteration + 1) % 10 == 0:
            # Refresh title only — no DARP re-evaluation needed
            ax_sa.set_title(
                f"SA  iter {iteration + 1}  |  T = {temperature:.3f}\n"
                f"Mission time: {prev_best[0]:.1f} s",
                fontsize=10,
            )
            plt.pause(0.001)

    sa_result = joint.optimize(
        initial_portions=initial_portions,
        max_iterations=args.iterations,
        seed=SA_SEED,
        iteration_callback=on_iteration,
    )

    final_result   = sa_result["final_mission_result"]
    final_portions = sa_result["optimized_portions"]

    ax_sa.clear()
    vis.draw_panel(
        ax_sa, final_result, final_portions,
        f"SA Complete — {sa_result['iterations']} iterations\n"
        f"Mission time: {sa_result['best_time']:.1f} s",
    )
    plt.tight_layout()

    print(f"\nSA best mission time : {sa_result['best_time']:.1f} s")
    print(f"Optimised portions   : {[f'{p:.3f}' for p in final_portions]}")
    print(f"Time saved           : "
          f"{initial_result.mission_time - sa_result['best_time']:.1f} s")

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved figure to      : {args.save}")

    if args.video:
        print(f"\nRendering flight simulation (speed x{args.speed})…")
        anim = vis.simulate(final_result, interval=50, speed_factor=args.speed,
                            save_path=args.video)
        print(f"Saved video to       : {args.video}")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()