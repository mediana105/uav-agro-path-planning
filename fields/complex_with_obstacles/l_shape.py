from shapely.geometry import Polygon

from src.pipeline import DroneConfig

FIELD = Polygon(
    shell=[
        (0.0, 0.0),
        (66.7, 0.0),
        (66.7, 26.7),
        (40.0, 26.7),
        (40.0, 50.0),
        (0.0, 50.0),
        (0.0, 0.0),
    ],
    holes=[
        [(8.3, 31.7), (23.3, 31.7), (23.3, 45.0), (8.3, 45.0), (8.3, 31.7)],
    ],
)
DRONES = [
    DroneConfig(
        id=0,
        speed=7.0,
        swath_width=2.0,
        turn_time=1.5,
        start_position=(3, 3),
        substance_rate=0.35,
        tank_volume=250.0,
        max_flight_time=160.0,
    ),
    DroneConfig(
        id=1,
        speed=5.5,
        swath_width=3.0,
        turn_time=2.0,
        start_position=(63.0, 3),
        substance_rate=0.45,
        tank_volume=200.0,
        max_flight_time=140.0,
    ),
    DroneConfig(
        id=2,
        speed=4.5,
        swath_width=2.0,
        turn_time=2.5,
        start_position=(3, 46),
        substance_rate=0.55,
        tank_volume=160.0,
        max_flight_time=120.0,
    ),
]

CELL_SIZE = 1.0
SA_ITERATIONS = 150
SA_SEED = 7
