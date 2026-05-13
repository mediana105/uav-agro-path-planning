import math
from dataclasses import dataclass, field


@dataclass
class DroneConfig:
    id: int
    speed: float
    swath_width: float
    turn_time: float

    start_position: tuple[float, float] = None

    bcd_coalesce_to: int | None = None

    substance_rate: float = 0.0  # spray consumption, L/m  (0 = spraying disabled)
    tank_volume: float = field(default_factory=lambda: math.inf)  # tank capacity, L
    max_flight_time: float = field(
        default_factory=lambda: math.inf
    )  # max airborne time per sortie, s

    def __post_init__(self):
        if self.speed <= 0:
            raise ValueError(f"Speed must be positive, got {self.speed}")
        if self.swath_width <= 0:
            raise ValueError(f"Swath width must be positive, got {self.swath_width}")
        if self.turn_time < 0:
            raise ValueError(f"Turn time cannot be negative, got {self.turn_time}")
        if self.start_position is None:
            raise ValueError("start_position is required")
        if self.bcd_coalesce_to is not None:
            if not isinstance(self.bcd_coalesce_to, int) or self.bcd_coalesce_to < 1:
                raise ValueError(
                    "bcd_coalesce_to must be None or a positive int, "
                    f"got {self.bcd_coalesce_to!r}"
                )
        if self.substance_rate < 0:
            raise ValueError(
                f"substance_rate cannot be negative, got {self.substance_rate}"
            )
        if self.tank_volume <= 0:
            raise ValueError(f"tank_volume must be positive, got {self.tank_volume}")
        if self.max_flight_time <= 0:
            raise ValueError(
                f"max_flight_time must be positive, got {self.max_flight_time}"
            )
