from shapely.geometry import Polygon

from src.pipeline import DroneConfig

#   (0,200)────────────────────────────────────(300,200)
#      │   │ ряд│    │ ряд│  │ │ ряд│    │ряд │    │
#      │   │  1 │    │  2 │  │ │  3 │    │ 4  │    │
#      │   │    │    │    │  │ │    │    │    │    │
#      │   │    │    │    │  │ │    │    │    │    │
#  [★](0,100)   │    │    │  │ │    │    │    │ (300,100)[★]
#      │   │    │    │    │  │ │    │    │    │    │
#      │   └────┘    └────┘  │ └────┘    └────┘    │
#      │         ← левая →   │   ← правая →        │
#   (0, 0)────────────────────────────────────(300, 0)
#
FIELD = Polygon(
    shell=[
        (0, 0), (300, 0), (300, 200), (0, 200),
    ],
    holes=[
        [(35,  20), (55,  20), (55, 180), (35, 180)],
        [(95,  20), (115, 20), (115, 180), (95, 180)],
        [(185, 20), (205, 20), (205, 180), (185, 180)],
        [(245, 20), (265, 20), (265, 180), (245, 180)],
    ],
)

DRONES = [
    DroneConfig(id=0, speed=6.0, swath_width=5.0, turn_time=1.8,
                start_position=(0.0, 100.0),
                substance_rate=0.40, tank_volume=150.0, max_flight_time=140.0),
    DroneConfig(id=1, speed=6.0, swath_width=5.0, turn_time=1.8,
                start_position=(300.0, 100.0),
                substance_rate=0.40, tank_volume=150.0, max_flight_time=140.0),
]

CELL_SIZE     = 4.0
SA_ITERATIONS = 150
SA_SEED       = 55