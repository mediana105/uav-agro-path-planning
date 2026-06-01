import math

import numpy as np
from shapely import Polygon
from shapely.geometry import box


class FieldDecomposition:
    def __init__(self, cell_size: float):
        self.cell_size = cell_size

        self.grid_rows: int | None = None
        self.grid_cols: int | None = None

        self.field_map: np.ndarray | None = None

        self.obstacles: list[tuple[int, int]] = []

    def from_polygon(self, field_polygon: Polygon, holes: list[Polygon] = None):
        if holes is None:
            holes = []

        min_x, min_y, max_x, max_y = field_polygon.bounds
        self.grid_cols = math.ceil((max_x - min_x) / self.cell_size)
        self.grid_rows = math.ceil((max_y - min_y) / self.cell_size)

        self.field_map = np.zeros((self.grid_rows, self.grid_cols), dtype=bool)

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                cx = min_x + (c + 0.5) * self.cell_size
                cy = min_y + (r + 0.5) * self.cell_size
                cell_box = box(
                    min_x + c * self.cell_size,
                    min_y + r * self.cell_size,
                    min_x + (c + 1) * self.cell_size,
                    min_y + (r + 1) * self.cell_size,
                )

                if field_polygon.intersects(cell_box) and all(
                    not h.intersects(cell_box) for h in holes
                ):
                    self.field_map[r, c] = True
                else:
                    self.field_map[r, c] = False
                    self.obstacles.append((r, c))

        return self

    def add_obstacle_cell(self, row: int, col: int):
        if self.field_map[row, col]:
            self.field_map[row, col] = False
            self.obstacles.append((row, col))

    def to_darp_grid(self):
        obstacle_positions = [r * self.grid_cols + c for r, c in self.obstacles]

        return self.grid_rows, self.grid_cols, obstacle_positions
