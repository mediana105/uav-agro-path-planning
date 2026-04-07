import math
import random
from collections.abc import Callable
from typing import Any

import numpy as np


class SimulatedAnnealingOptimizer:

    def __init__(self,
                 initial_temp: float = 1000.0,
                 final_temp: float = 1e-10,
                 cooling_rate: float = 0.95,
                 step_size: float = 0.1,
                 random_seed: int = 1):

        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.step_size = step_size
        self.random = random.Random(random_seed)
        np.random.seed(random_seed)

    def optimize(self,
                 objective_function: Callable[[list[float]], float],
                 initial_portions: list[float],
                 max_iterations: int = 1000,
                 iteration_callback: Callable = None) -> dict[str, Any]:
        current_portions = initial_portions.copy()
        current_value = objective_function(current_portions)

        best_portions = current_portions.copy()
        best_value = current_value

        temperature = self.initial_temp

        iteration = 0

        while temperature > self.final_temp and iteration < max_iterations:
            neighbor_portions = self._generate_neighbor(current_portions)
            neighbor_value = objective_function(neighbor_portions)
            delta = neighbor_value - current_value

            accepted = False
            if delta < 0 or (self.random.random() < math.exp(-delta / temperature)):
                current_portions = neighbor_portions
                current_value = neighbor_value
                accepted = True

                if current_value < best_value:
                    best_portions = current_portions.copy()
                    best_value = current_value

            if iteration_callback is not None:
                iteration_callback(
                    iteration, current_portions, current_value,
                    best_portions, best_value, temperature, accepted,
                )

            temperature *= self.cooling_rate
            iteration += 1

        return {
            'optimized_portions': best_portions,
            'best_value': best_value,
            'iterations': iteration,
            'final_temperature': temperature
        }

    def _generate_neighbor(self, portions: list[float]) -> list[float]:
        neighbor = np.array(portions, dtype=float)
        n = len(neighbor)

        # Select two different drones
        i = self.random.randint(0, n - 1)
        j = self.random.randint(0, n - 1)
        while i == j:
            j = self.random.randint(0, n - 1)

        delta = self.random.uniform(0, self.step_size) * min(neighbor[i], neighbor[j])

        if self.random.random() < 0.5:
            neighbor[i] -= delta
            neighbor[j] += delta
        else:
            neighbor[i] += delta
            neighbor[j] -= delta

        neighbor = np.maximum(neighbor, 0.01)

        # Normalize
        total = np.sum(neighbor)
        neighbor = neighbor / total

        return neighbor.tolist()
