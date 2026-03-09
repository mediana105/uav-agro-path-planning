from dataclasses import dataclass


@dataclass
class DroneConfig:
    id: int
    speed: float
    swath_width: float
    turn_time: float

    start_position: tuple[float, float] = None

    def __post_init__(self):
        if self.speed <= 0:
            raise ValueError(f"Speed must be positive, got {self.speed}")
        if self.swath_width <= 0:
            raise ValueError(f"Swath width must be positive, got {self.swath_width}")
        if self.turn_time < 0:
            raise ValueError(f"Turn time cannot be negative, got {self.turn_time}")
        if self.start_position is None:
            raise ValueError("start_position is required")