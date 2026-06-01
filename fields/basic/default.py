from shapely.geometry import Polygon

from src.pipeline import DroneConfig

FIELD = Polygon(
    [
        (0.0, 0.0),
        (180.0, 0.0),
        (180.0, 120.0),
        (0.0, 120.0),
        (0.0, 0.0),
    ]
)

DRONES = [
    DroneConfig(
        id=0,
        speed=7.0,
        swath_width=6.0,
        turn_time=1.5,
        start_position=(20.0, 90.0),
        substance_rate=0.0,
        tank_volume=float("inf"),
        max_flight_time=float("inf"),
    ),
    DroneConfig(
        id=1,
        speed=5.0,
        swath_width=4.5,
        turn_time=2.0,
        start_position=(160.0, 60.0),
        substance_rate=0.0,
        tank_volume=float("inf"),
        max_flight_time=float("inf"),
    ),
    DroneConfig(
        id=2,
        speed=4.0,
        swath_width=3.5,
        turn_time=2.3,
        start_position=(90.0, 15.0),
        substance_rate=0.0,
        tank_volume=float("inf"),
        max_flight_time=float("inf"),
    ),
]

CELL_SIZE = 1.0
SA_SEED = 42
