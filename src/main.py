import math

from shapely.geometry import Polygon
from pipeline import MissionOptimizer, DroneConfig
from src.path_planning.boustrophedon import path_length


def create_test_scenario():
    """Create a test scenario with field and drone configurations."""
    # Create a rectangular field (100x50 meters)
    field = Polygon([(0, 0), (100, 0), (100, 50), (0, 50)])

    # Drones with different parameters
    drones = [
        DroneConfig(
            id=0,
            speed=10.0,      # m/s
            swath_width=5.0, # coverage width, meters
            turn_time=2.0,   # turn duration, seconds
            start_position=(0, 0)
        ),
        DroneConfig(
            id=1,
            speed=8.0,
            swath_width=3.0,
            turn_time=1.5,
            start_position=(50, 0)
        ),
        DroneConfig(
            id=2,
            speed=12.0,
            swath_width=4.0,
            turn_time=1.0,
            start_position=(100, 0)
        )
    ]

    return field, drones


def main():
    """Main function - complete pipeline execution."""
    print("Starting full pipeline")

    # Create test scenario
    field, drones = create_test_scenario()

    # Create mission optimizer
    optimizer = MissionOptimizer(field, drones, cell_size=2.0)

    # Evaluate mission (automatic decomposition based on productivity)
    result = optimizer.evaluate()

    # Print results
    print(f"\nMission time: {result.mission_time:.2f} seconds")

    for zone in result.zones:
        print(f"\nDrone {zone.drone_id} zone:")
        print(f"  Time: {zone.total_time:.2f} seconds")
        print(f"  Optimal angle: {math.degrees(zone.optimal_angle):.2f}°")
        print(f"  Path length: {path_length(zone.path):.2f} meters")
        print(f"  Number of waypoints: {len(zone.path)}")


if __name__ == "__main__":
    main()
