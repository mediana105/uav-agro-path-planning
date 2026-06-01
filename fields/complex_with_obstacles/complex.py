from shapely.geometry import Polygon

from src.pipeline import DroneConfig

FIELD = Polygon(
    shell=[
        (13.3, 0.0),
        (53.3, 0.0),
        (66.7, 33.3),
        (60.0, 66.7),
        (6.7, 66.7),
        (0.0, 33.3),
        (13.3, 0.0),
    ],
    holes=[
        [
            (11.7, 40.0),
            (20.0, 36.7),
            (25.0, 43.3),
            (20.0, 50.0),
            (11.7, 48.3),
            (11.7, 40.0),
        ],
        [(36.7, 38.3), (51.7, 38.3), (51.7, 51.7), (36.7, 51.7), (36.7, 38.3)],
    ],
)

DRONES = [
    DroneConfig(
        id=0,
        speed=6.5,
        swath_width=2.0,
        turn_time=1.5,
        start_position=(15, 4),
        substance_rate=0.4,
        tank_volume=110.0,
        max_flight_time=170.0,
    ),
    DroneConfig(
        id=1,
        speed=5.0,
        swath_width=2.0,
        turn_time=2.0,
        start_position=(52, 4),
        substance_rate=0.5,
        tank_volume=95.0,
        max_flight_time=145.0,
    ),
    DroneConfig(
        id=2,
        speed=5.5,
        swath_width=2.0,
        turn_time=1.8,
        start_position=(33.3, 60.0),
        substance_rate=0.42,
        tank_volume=100.0,
        max_flight_time=155.0,
    ),
]
CELL_SIZE = 1.0
SA_ITERATIONS = 180
SA_SEED = 13
