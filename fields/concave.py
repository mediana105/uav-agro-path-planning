from shapely.geometry import Polygon

from src.pipeline import DroneConfig


#   (0,260)──(110,260)              (240,260)──(350,260)
#      │          │                      │           │
#      │       (110,150)────────────(240,150)        │
#   [★](55,205)                               [★](295,205)
#      │                                            │
#   (0, 0)──[★](87,5)────────────[★](262,5)──(350, 0)

FIELD = Polygon([
    (0, 0), (350, 0), (350, 260),
    (240, 260), (240, 150), (110, 150), (110, 260),
    (0, 260),
])

DRONES = [
    DroneConfig(id=0, speed=7.0, swath_width=6.0, turn_time=1.8,
                start_position=(87.0, 5.0),
                substance_rate=0.40, tank_volume=220.0, max_flight_time=160.0),
    DroneConfig(id=1, speed=7.0, swath_width=6.0, turn_time=1.8,
                start_position=(262.0, 5.0),
                substance_rate=0.40, tank_volume=220.0, max_flight_time=160.0),
    DroneConfig(id=2, speed=4.0, swath_width=5.0, turn_time=2.0,
                start_position=(55.0, 205.0),
                substance_rate=0.45, tank_volume=130.0, max_flight_time=140.0),
    DroneConfig(id=3, speed=4.0, swath_width=5.0, turn_time=2.0,
                start_position=(295.0, 205.0),
                substance_rate=0.45, tank_volume=130.0, max_flight_time=140.0),
]

CELL_SIZE     = 4.0
SA_ITERATIONS = 180
SA_SEED       = 23