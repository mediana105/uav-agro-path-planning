from shapely.geometry import Polygon

from src.pipeline import DroneConfig


#    (15,62)──(90,55)──────(220,72)──────(340,58)──(460,65)
#       │                                               │
#   (0,25)                                          (510,35)
#       │                                               │
#    (60,0)──(160,12)──────(270,0)───────(390,18)──(490,5)
#
FIELD = Polygon(
    shell=[
        (0, 25),
        (60, 0), (160, 12), (270, 0), (390, 18), (490, 5),
        (510, 35),
        (460, 65), (340, 58), (220, 72), (90, 55), (15, 62),
    ],
    holes=[
        [(112, 22), (132, 16), (148, 42), (128, 48)],
        [(242, 26), (272, 20), (267, 52)],
        [(378, 32), (398, 26), (418, 34), (422, 50), (402, 58), (382, 50)],
    ],
)

DRONES = [
    DroneConfig(id=0, speed=7.5, swath_width=5.0, turn_time=1.5,
                start_position=(114.0, 35.0)),
    DroneConfig(id=1, speed=6.0, swath_width=5.0, turn_time=1.8,
                start_position=(295.0, 35.0),
                substance_rate=0.4, tank_volume=180.0, max_flight_time=150.0),
    DroneConfig(id=2, speed=6.0, swath_width=4.5, turn_time=2.0,
                start_position=(443.0, 35.0),
                substance_rate=0.45, tank_volume=160.0, max_flight_time=140.0),
    DroneConfig(id=3, speed=5.0, swath_width=4.0, turn_time=2.5,
                start_position=(490.0, 35.0),
                substance_rate=0.5, tank_volume=130.0, max_flight_time=120.0),
]

CELL_SIZE     = 4.0
SA_ITERATIONS = 200
SA_SEED       = 17