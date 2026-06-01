import argparse
import importlib
from pathlib import Path

import matplotlib.pyplot as plt

from src.visualization.visualizer import MissionVisualizer


def _parse_args():
    p = argparse.ArgumentParser(
        description="UAV Agricultural Path Planning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "field",
        nargs="?",
        default="basic/default",
        help="Field config name.",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="SA iterations.",
    )
    p.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save plan figure (default: out/<field>/plan.png)",
    )
    p.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to save flight video (default: out/<field>/flight.mp4)",
    )
    p.add_argument(
        "--algorithm",
        choices=["sa", "tabu"],
        default="sa",
        help="Optimization algorithm: sa (Simulated Annealing) or tabu (Tabu Search)",
    )
    p.add_argument("--no-video", action="store_true", help="Skip video generation")
    p.add_argument(
        "--speed", type=float, default=10.0, help="Playback speed factor for simulation"
    )
    p.add_argument("--show", action="store_true", help="Show interactive plot window")
    return p.parse_args()


def load_field(name: str):
    possible_paths = [
        f"fields.{name}",
        f"fields.basic.{name}",
        f"fields.complex.{name}",
        f"fields.complex_with_obstacles.{name}",
    ]

    last_error = None
    for module_path in possible_paths:
        try:
            module = importlib.import_module(module_path)
            field = module.FIELD
            drones = module.DRONES
            cell_size = getattr(module, "CELL_SIZE", 4.0)
            sa_iter = getattr(module, "SA_ITERATIONS", 100)
            sa_seed = getattr(module, "SA_SEED", 42)
            return field, drones, cell_size, sa_iter, sa_seed
        except ModuleNotFoundError as e:
            last_error = e
            continue
    raise ModuleNotFoundError(
        f"Field '{name}' not found in any of the expected paths. Last error: {last_error}"
    )


def main():
    args = _parse_args()

    field, drones, cell_size, sa_iter, sa_seed = load_field(args.field)

    if args.iterations is not None:
        sa_iter = args.iterations

    out_dir = Path("out") / args.field
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.save or str(out_dir / "plan.png")
    flight_path = args.video or str(out_dir / "flight.mp4")

    missionVisualizer = MissionVisualizer(
        field,
        drones,
        cell_size=cell_size,
    )

    _, sa_result = missionVisualizer.visualize(
        sa_iterations=sa_iter,
        sa_seed=sa_seed,
        algorithm=args.algorithm,
        show=args.show,
        save_path=plan_path,
    )
    print(f"[main] Plan saved in {plan_path}")

    dec_path = str(out_dir / "bcd_decomposition.png")
    missionVisualizer.save_bcd_decomposition_figure(sa_result, dec_path)
    print(f"[main] BCD decomposition figure saved in {dec_path}")

    adj_path = str(out_dir / "bcd_adjacency.png")
    missionVisualizer.save_bcd_adjacency_figure(sa_result, adj_path)
    print(f"[main] BCD adjacency graph saved in {adj_path}")

    if not args.no_video:
        print(f"[main] Generating animation (speed x{args.speed})...")
        missionVisualizer.simulate(
            sa_result,
            interval=40,
            speed_factor=args.speed,
            save_path=flight_path,
        )
        print(f"[main] Video saved in {flight_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
