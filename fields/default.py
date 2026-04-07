from shapely.geometry import Polygon

from src.pipeline import DroneConfig

FIELD = Polygon([
    (0, 0), (200, 0), (200, 150), (0, 150),
])

DRONES = [
    DroneConfig(id=0, speed=6.0, swath_width=5.0, turn_time=1.5, start_position=(  0,  75)),
    DroneConfig(id=1, speed=4.0, swath_width=5.0, turn_time=2.0, start_position=(200,  75)),
    DroneConfig(id=2, speed=5.0, swath_width=5.0, turn_time=2.0, start_position=(100,   0)),
]

CELL_SIZE = 5.0
SA_SEED   = 42