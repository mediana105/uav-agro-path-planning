from shapely.geometry import Polygon

from src.pipeline import DroneConfig

FIELD = Polygon(
    shell=[
        (0.0, 0.0),
        (0.0, 66.7),
        (66.7, 66.7),
        (66.7, 0.0),
        (49.3, 0.0),
        (49.3, 12.7),
        (34.0, 12.7),
        (34.0, 39.3),
        (17.3, 39.3),
        (17.3, 0.0),
        (0.0, 0.0),
    ],
)
DRONES = [
    DroneConfig(
        id=0,
        speed=6.5,
        swath_width=2.0,
        turn_time=1.5,
        start_position=(10.0, 2.7),
        substance_rate=0.42,
        tank_volume=120.0,
        max_flight_time=170.0,
    ),
    DroneConfig(
        id=1,
        speed=5.5,
        swath_width=2.0,
        turn_time=1.8,
        start_position=(56.7, 2.7),
        substance_rate=0.45,
        tank_volume=110.0,
        max_flight_time=155.0,
    ),
    DroneConfig(
        id=2,
        speed=5.0,
        swath_width=2.0,
        turn_time=2.0,
        start_position=(30, 50),
        substance_rate=0.48,
        tank_volume=95.0,
        max_flight_time=140.0,
    ),
]
CELL_SIZE = 1
SA_ITERATIONS = 200
SA_SEED = 102
