from shapely.geometry import Polygon

from src.pipeline import DroneConfig

# Неправильная вогнутая форма (трапеция с выемкой сверху) + два препятствия
#
#   (20,200)──────────────────────(180,200)
#      │                              │
#      │   [дерево]    [строение]     │
#      │                              │
#   (0,100)                       (200,100)
#       \                            /
#        \                          /
#      (40, 0)────────────────(160, 0)
#
FIELD = Polygon(
    shell=[
        (40, 0), (160, 0),
        (200, 100), (180, 200),
        (20, 200), (0, 100),
    ],
    holes=[
        # группа деревьев (круглая — апроксимация)
        [(35, 120), (60, 110), (75, 130), (60, 150), (35, 145)],
        # строение
        [(110, 115), (155, 115), (155, 155), (110, 155)],
    ],
)

DRONES = [
    DroneConfig(id=0, speed=6.5, swath_width=5.0, turn_time=1.5,
                start_position=(40.0, 0.0),
                substance_rate=0.4, tank_volume=110.0, max_flight_time=170.0),
    DroneConfig(id=1, speed=5.0, swath_width=4.5, turn_time=2.0,
                start_position=(160.0, 0.0),
                substance_rate=0.5, tank_volume=95.0, max_flight_time=145.0),
    DroneConfig(id=2, speed=5.5, swath_width=5.0, turn_time=1.8,
                start_position=(100.0, 200.0),
                substance_rate=0.42, tank_volume=100.0, max_flight_time=155.0),
]

CELL_SIZE     = 3.0
SA_ITERATIONS = 180
SA_SEED       = 13