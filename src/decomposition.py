import numpy as np


class FieldDecomposition:
    def __init__(self, cell_size: float):
        self.cell_size = cell_size

        self.grid_rows: int | None = None
        self.grid_cols: int | None = None

        self.field_map: np.ndarray | None = None

        self.obstacles: list[tuple[int, int]] = []

    def from_rectangle(self, field_width: float, field_height: float):
        self.grid_cols = int(field_width / self.cell_size)
        self.grid_rows = int(field_height / self.cell_size)

        self.field_map = np.ones(
            (self.grid_rows, self.grid_cols),
            dtype=bool
        )

        return self

    def from_polygon(self, polygon_vertices):
        raise NotImplementedError(
            "Polygon fields are not implemented yet"
        )

    def add_obstacle_cell(self, row: int, col: int):
        if self.field_map[row, col]:
            self.field_map[row, col] = False
            self.obstacles.append((row, col))

    def to_darp_grid(self):
        obstacle_positions = [
            r * self.grid_cols + c for r, c in self.obstacles
        ]

        return self.grid_rows, self.grid_cols, obstacle_positions
