from decomposition import FieldDecomposition
from darp.darp import DARP


def decompose_field():
    field_width = float(input())
    field_height = float(input())
    cell_size = float(input())

    fd = FieldDecomposition(cell_size).from_rectangle(field_width, field_height)
    rows, cols = fd.grid_rows, fd.grid_cols

    num_robots = int(input())
    robot_positions = [int(input()) for _ in range(num_robots)]

    obstacle_positions = []

    darp = DARP(
        nx=rows,
        ny=cols,
        notEqualPortions=False,
        given_initial_positions=robot_positions,
        given_portions=None,
        obstacles_positions=obstacle_positions,
        visualization=False
    )

    success, iterations = darp.divideRegions()
    return success, iterations, darp


if __name__ == "__main__":
    success, iterations, darp = decompose_field()
    print(success)
    print(iterations)
    print(darp.A)
