"""
UAV Agricultural Path Planning — entry point.

Usage:
    python -m src.main # default field (fields/default.py)
    python -m src.main l_shape # fields/l_shape.py
    python -m src.main complex # fields/complex.py

Results are saved in the out/<field_name>/ folder:
    plan.png — static plan (initial partitioning + SA-optimization)
    flight.mp4 — flight animation
"""

import argparse
import importlib
from pathlib import Path

import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt

from src.visualization.visualizer import MissionVisualizer


def _parse_args():
    p = argparse.ArgumentParser(
        description="UAV Agricultural Path Planning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("field", nargs="?", default="default",
                   help="Field config name from fields/ (e.g. l_shape)")
    p.add_argument("--iterations", type=int, default=None,
                   help="SA iterations (overrides field config SA_ITERATIONS)")
    p.add_argument("--seed", type=int, default=None,
                   help="SA random seed (overrides field config SA_SEED)")
    p.add_argument("--save", type=str, default=None,
                   help="Path to save plan figure (default: out/<field>/plan.png)")
    p.add_argument("--video", type=str, default=None,
                   help="Path to save flight video (default: out/<field>/flight.mp4)")
    p.add_argument("--no-video", action="store_true",
                   help="Skip video generation")
    p.add_argument("--speed", type=float, default=10.0,
                   help="Playback speed factor for simulation")
    p.add_argument("--show", action="store_true",
                   help="Show interactive plot window")
    return p.parse_args()


def load_field(name: str):
    module = importlib.import_module(f"fields.{name}")
    field        = module.FIELD
    drones       = module.DRONES
    cell_size    = getattr(module, "CELL_SIZE",     4.0)
    sa_iter      = getattr(module, "SA_ITERATIONS", 100)
    sa_seed      = getattr(module, "SA_SEED",       42)
    return field, drones, cell_size, sa_iter, sa_seed


def main():
    args = _parse_args()

    print(f"[main] Uploading the field: fields/{args.field}.py")
    field, drones, cell_size, sa_iter, sa_seed = load_field(args.field)

    if args.iterations is not None:
        sa_iter = args.iterations
    if args.seed is not None:
        sa_seed = args.seed

    out_dir = Path("out") / args.field
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path   = args.save  or str(out_dir / "plan.png")
    flight_path = args.video or str(out_dir / "flight.mp4")

    vis = MissionVisualizer(field, drones, cell_size=cell_size)

    print(f"[main] Building the plan ({sa_iter} iteration SA)...")
    _, sa_result = vis.visualize(
        sa_iterations=sa_iter,
        sa_seed=sa_seed,
        show=args.show,
        save_path=plan_path,
    )
    print(f"[main] Plan saved in {plan_path}")

    if not args.no_video:
        print(f"[main] Generating animation (speed x{args.speed})...")
        vis.simulate(
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