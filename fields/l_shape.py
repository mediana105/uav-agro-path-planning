from shapely.geometry import Polygon
from src.pipeline import DroneConfig

# Г-образное поле (~1.7 га) с прудом внутри
#
#   (0,150)──────────(120,150)
#      │                │
#      │   [пруд]       │
#      │                │
#   (0,80)         (120,80)──────(200,80)
#      │                              │
#      │                              │
#   (0, 0)──────────────────────(200, 0)
#
FIELD = Polygon(
    shell=[
        (0, 0), (200, 0), (200, 80),
        (120, 80), (120, 150), (0, 150),
    ],
    holes=[
        [(25, 95), (70, 95), (70, 135), (25, 135)],
    ],
)

DRONES = [
    # Быстрый, широкий захват
    DroneConfig(id=0, speed=7.0, swath_width=6.0, turn_time=1.5,
                start_position=(0.0, 0.0),
                substance_rate=0.35, tank_volume=250.0, max_flight_time=160.0),
    # Средний
    DroneConfig(id=1, speed=5.5, swath_width=5.0, turn_time=2.0,
                start_position=(200.0, 0.0),
                substance_rate=0.45, tank_volume=200.0, max_flight_time=140.0),
    # Точный, узкий захват
    DroneConfig(id=2, speed=4.5, swath_width=4.0, turn_time=2.5,
                start_position=(0.0, 150.0),
                substance_rate=0.55, tank_volume=160.0, max_flight_time=120.0),
]

CELL_SIZE     = 3.0
SA_ITERATIONS = 150
SA_SEED       = 7