from shapely.geometry import Polygon

from src.pipeline import DroneConfig


FIELD = Polygon(
    [
        (0.0, 0.0),
        (116.7, 0.0),
        (116.7, 86.7),
        (80.0, 86.7),
        (80.0, 40.0),
        (36.7, 40.0),
        (36.7, 86.7),
        (0.0, 86.7),
        (0.0, 0.0),
    ]
)
DRONES = [
    DroneConfig(
        id=0,
        speed=7.0,
        swath_width=6.0,
        turn_time=1.8,
        start_position=(29.0, 6),
        substance_rate=0.4,
        tank_volume=220.0,
        max_flight_time=160.0,
    ),
    DroneConfig(
        id=1,
        speed=7.0,
        swath_width=6.0,
        turn_time=1.8,
        start_position=(87.3, 6),
        substance_rate=0.4,
        tank_volume=220.0,
        max_flight_time=160.0,
    ),
    DroneConfig(
        id=2,
        speed=4.0,
        swath_width=5.0,
        turn_time=2.0,
        start_position=(18.3, 68.3),
        substance_rate=0.45,
        tank_volume=130.0,
        max_flight_time=140.0,
    ),
    DroneConfig(
        id=3,
        speed=4.0,
        swath_width=5.0,
        turn_time=2.0,
        start_position=(98.3, 68.3),
        substance_rate=0.45,
        tank_volume=130.0,
        max_flight_time=140.0,
    ),
]
CELL_SIZE = 2.0
SA_ITERATIONS = 180
SA_SEED = 23
