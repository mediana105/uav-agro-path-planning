from shapely.geometry import Polygon

from src.pipeline import DroneConfig


FIELD = Polygon(
    shell=[
        (0.0, 20.0),
        (20.0, 0.0),
        (53.3, 4.0),
        (90.0, 0.0),
        (130.0, 6.0),
        (163.3, 1.7),
        (170.0, 28.0),
        (153.3, 50.0),
        (113.3, 48.0),
        (73.3, 55.0),
        (30.0, 45.0),
        (5.0, 48.0),
        (0.0, 20.0),
    ],
    holes=[
        [(35.0, 18.0), (44.0, 14.0), (52.0, 28.0), (43.0, 32.0), (35.0, 18.0)],
        [(78.0, 20.0), (92.0, 16.0), (90.0, 38.0), (78.0, 20.0)],
        [
            (124.0, 22.0),
            (134.0, 18.0),
            (142.0, 24.0),
            (144.0, 36.0),
            (136.0, 42.0),
            (126.0, 38.0),
            (124.0, 22.0),
        ],
    ],
)
DRONES = [
    DroneConfig(
        id=0,
        speed=7.5,
        swath_width=5.0,
        turn_time=1.5,
        start_position=(38.0, 25.0),
        substance_rate=0.0,
        tank_volume=float("inf"),
        max_flight_time=float("inf"),
    ),
    DroneConfig(
        id=1,
        speed=6.0,
        swath_width=5.0,
        turn_time=1.8,
        start_position=(98.0, 25.0),
        substance_rate=0.4,
        tank_volume=180.0,
        max_flight_time=150.0,
    ),
    DroneConfig(
        id=2,
        speed=6.0,
        swath_width=4.5,
        turn_time=2.0,
        start_position=(148.0, 25.0),
        substance_rate=0.45,
        tank_volume=160.0,
        max_flight_time=140.0,
    ),
    DroneConfig(
        id=3,
        speed=6.0,
        swath_width=5.0,
        turn_time=2.5,
        start_position=(164.0, 25.0),
        substance_rate=0.5,
        tank_volume=130.0,
        max_flight_time=120.0,
    ),
]
CELL_SIZE = 2.0
SA_ITERATIONS = 200
SA_SEED = 17
