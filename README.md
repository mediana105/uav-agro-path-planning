# UAV Agricultural Path Planning

A comprehensive mission planning system for agricultural UAV swarms. The system decomposes agricultural fields into sub-zones, generates optimal coverage paths for each drone, and optimizes the workload distribution to minimize total mission time.

## Features

- **Multi-drone mission planning** — supports heterogeneous drone fleets with different speeds, swath widths, tank volumes, and spray rates
- **Field decomposition** — divides a field polygon into equitable sub-zones using the **DARP** (Divide Areas based on Robot initial Positions) algorithm
- **Coverage path generation** — produces efficient boustrophedon coverage paths within each zone
- **BCD (Boustrophedon Cellular Decomposition)** — decomposes complex polygonal zones (with holes) into monotone cells for optimal coverage
- **Visibility-graph-based transitions** — computes shortest obstacle-avoiding paths between coverage swaths using a visibility graph
- **Flight time & resource constraint modeling** — accounts for tank volume, spray rate, max flight time, and return-to-home (RTH) refuel/recharge stops
- **Mission time optimization** — uses **Simulated Annealing (SA)** or **Tabu Search** to optimize the workload distribution (portions) across drones
- **Optimal flight angle search** — finds the sweep direction that minimizes the number of passes using the minimum-altitude algorithm
- **Visualization** — generates plan figures, BCD decomposition diagrams, adjacency graphs, and flight animation videos

## Project Structure

```
├── src/
│   ├── main.py                          # CLI entry point
│   ├── angle_finders/
│   │   ├── altitude_optimizer.py        # Minimum-altitude optimal angle search
│   │   └── brute_force.py               # Brute-force angle search by path length
│   ├── decomposition/
│   │   ├── field_decomposition.py       # Grid-based field discretization
│   │   └── darp.py                      # DARP multi-robot area division
│   ├── optimization/
│   │   ├── joint_optimizer.py           # Joint optimizer (SA / Tabu)
│   │   └── optimization_algorithms.py   # SA and Tabu Search implementations
│   ├── path_planning/
│   │   ├── bcd.py                       # Boustrophedon Cellular Decomposition
│   │   ├── coverage_path.py             # Coverage path generation & swath ordering
│   │   ├── visibility_graph.py          # Visibility graph for obstacle avoidance
│   │   └── utils.py                     # Path utilities (2-opt, turns, RTH logic)
│   ├── pipeline/
│   │   ├── pipeline.py                  # MissionPlanner — orchestrates the full pipeline
│   │   ├── drone_config.py              # DroneConfig dataclass
│   │   └── mission_result.py            # MissionResult / ZoneResult dataclasses
│   └── visualization/
│       └── visualizer.py                # Matplotlib-based visualization & animation
├── fields/
│   ├── basic/
│   │   ├── default.py                   # Simple rectangular field (3 drones)
│   │   └── concave.py                   # Concave C-shaped field (4 drones)
│   ├── complex/
│   │   ├── complex_1.py                 # Complex polygonal field (3 drones)
│   │   └── complex_2.py                 # Complex polygonal field (3 drones)
│   └── complex_with_obstacles/
│       ├── complex.py                   # Field with 2 internal obstacles (3 drones)
│       ├── l_shape.py                   # L-shaped field with 1 obstacle (3 drones)
│       └── narrow.py                    # Narrow winding field with 3 obstacles (4 drones)
├── tests/
│   ├── angle_finders/
│   │   └── test_altitude_optimizer.py   # Tests for altitude optimizer
│   └── path_planning/
│       └── test_bcd.py                  # Tests for BCD decomposition
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd uav-agro-path-planning

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate 

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- `numpy` — numerical computations
- `shapely` — geometric operations (polygons, intersections, buffers)
- `opencv-python-headless` — distance transforms and connected-component analysis (used in DARP)
- `numba` — JIT compilation for performance-critical DARP loops
- `scipy` — scientific computing utilities
- `matplotlib` — visualization and animation
- `ortools` — Google OR-Tools for optimal swath ordering (TSP solver)
- `pytest` — testing framework

## Usage

### Command-line interface

```bash
python -m src.main [field_name] [options]
```

#### Arguments

| Argument | Description | Default |
|---|---|---|
| `field_name` | Field configuration name (dot-separated path) | `basic/default` |
| `--iterations` | Number of SA/Tabu iterations | Per-field default |
| `--algorithm` | Optimization algorithm: `sa` or `tabu` | `sa` |
| `--save` | Path to save the plan figure | `out/<field>/plan.png` |
| `--video` | Path to save the flight animation | `out/<field>/flight.mp4` |
| `--no-video` | Skip video generation | `false` |
| `--speed` | Playback speed factor for simulation | `10.0` |
| `--show` | Show interactive plot window | `false` |

\
### Output

The system generates the following outputs in the `out/<field>/` directory:

- **`plan.png`** — side-by-side comparison of initial vs. optimized decomposition with coverage paths
- **`bcd_decomposition.png`** — BCD cell decomposition for each drone's zone
- **`bcd_adjacency.png`** — BCD adjacency graph showing cell connectivity
- **`flight.mp4`** — animated flight simulation

## Testing

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/path_planning/test_bcd.py
pytest tests/angle_finders/test_altitude_optimizer.py

# Run with verbose output
pytest -v
```