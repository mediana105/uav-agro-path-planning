from shapely.geometry import Polygon

from src.pipeline import DroneConfig

FIELD = Polygon(
    [
        (66.7, 0.0),
        (44.0, 0.0),
        (44.0, 26.0),
        (22.7, 26.0),
        (22.7, 0.0),
        (0.0, 0.0),
        (0.0, 16.0),
        (29.3, 33.3),
        (0.0, 50.7),
        (0.0, 66.7),
        (66.7, 66.7),
        (66.7, 38.0),
        (55.3, 33.3),
        (66.7, 28.7),
        (66.7, 0.0),
    ]
)
DRONES = [
    DroneConfig(
        id=0,
        speed=6.5,
        swath_width=2.0,
        turn_time=1.5,
        start_position=(6.7, 4),
        substance_rate=0.42,
        tank_volume=120.0,
        max_flight_time=170.0,
    ),
    DroneConfig(
        id=1,
        speed=5.5,
        swath_width=2.5,
        turn_time=1.8,
        start_position=(60.0, 4),
        substance_rate=0.45,
        tank_volume=110.0,
        max_flight_time=155.0,
    ),
    DroneConfig(
        id=2,
        speed=5.5,
        swath_width=2.7,
        turn_time=1.8,
        start_position=(33.3, 65.0),
        substance_rate=0.45,
        tank_volume=110.0,
        max_flight_time=155.0,
    ),
]
CELL_SIZE = 1.0
SA_ITERATIONS = 200
SA_SEED = 101
