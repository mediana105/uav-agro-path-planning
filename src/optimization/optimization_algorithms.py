import math
import random
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np


def _generate_neighbor(portions: list[float], step_size: float, rng: random.Random) -> list[float]:
    neighbor = np.array(portions, dtype=float)
    n = len(neighbor)

    i = rng.randint(0, n - 1)
    j = rng.randint(0, n - 1)
    while i == j:
        j = rng.randint(0, n - 1)

    delta = rng.uniform(0, step_size) * min(neighbor[i], neighbor[j])

    if rng.random() < 0.5:
        neighbor[i] -= delta
        neighbor[j] += delta
    else:
        neighbor[i] += delta
        neighbor[j] -= delta

    neighbor = np.maximum(neighbor, 0.01)
    neighbor = neighbor / np.sum(neighbor)

    return neighbor.tolist()


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
        return _generate_neighbor(portions, self.step_size, self.random)


class TabuSearchOptimizer:
    def __init__(self,
                 tabu_tenure: int = 15,
                 num_neighbors: int = 5,
                 step_size: float = 0.05,
                 tabu_epsilon: float = 1e-3,
                 random_seed: int = 1):
        self.tabu_tenure = tabu_tenure
        self.num_neighbors = num_neighbors
        self.step_size = step_size
        self.tabu_epsilon = tabu_epsilon
        self.random = random.Random(random_seed)
        np.random.seed(random_seed)

    def optimize(self,
                 objective_function: Callable[[list[float]], float],
                 initial_portions: list[float],
                 max_iterations: int = 200,
                 iteration_callback: Callable = None) -> dict[str, Any]:
        current_portions = initial_portions.copy()
        current_value = objective_function(current_portions)

        best_portions = current_portions.copy()
        best_value = current_value

        tabu_list: deque[list[float]] = deque(maxlen=self.tabu_tenure)
        tabu_list.append(current_portions)

        for iteration in range(max_iterations):
            candidates = [
                self._generate_neighbor(current_portions)
                for _ in range(self.num_neighbors)
            ]

            # Evaluate all candidates; build (value, portions) sorted list
            evaluated = sorted(
                ((objective_function(c), c) for c in candidates),
                key=lambda x: x[0],
            )

            chosen_value, chosen_portions = None, None
            accepted = False

            for value, candidate in evaluated:
                if not self._is_tabu(candidate, tabu_list):
                    chosen_value, chosen_portions = value, candidate
                    accepted = True
                    break
                if value < best_value:
                    # Aspiration: accept tabu solution if it beats global best
                    chosen_value, chosen_portions = value, candidate
                    accepted = True
                    break

            if chosen_portions is None:
                # All candidates are tabu and none beats best — take the best anyway
                chosen_value, chosen_portions = evaluated[0]
                accepted = True

            current_portions = chosen_portions
            current_value = chosen_value
            tabu_list.append(current_portions)

            if current_value < best_value:
                best_portions = current_portions.copy()
                best_value = current_value

            if iteration_callback is not None:
                iteration_callback(
                    iteration, current_portions, current_value,
                    best_portions, best_value, 0.0, accepted,
                )

        return {
            'optimized_portions': best_portions,
            'best_value': best_value,
            'iterations': max_iterations,
            'final_temperature': 0.0,
        }

    def _generate_neighbor(self, portions: list[float]) -> list[float]:
        return _generate_neighbor(portions, self.step_size, self.random)

    def _is_tabu(self, candidate: list[float], tabu_list: deque) -> bool:
        c = np.array(candidate)
        return any(
            np.max(np.abs(c - np.array(t))) < self.tabu_epsilon
            for t in tabu_list
        )
