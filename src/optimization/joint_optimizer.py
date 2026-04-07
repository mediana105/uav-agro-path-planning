from typing import Any

from ..pipeline.mission_result import MissionResult
from .optimization_algorithms import SimulatedAnnealingOptimizer


class JointOptimizer:
    def __init__(self, mission_optimizer):
        self.mission_optimizer = mission_optimizer
        self.n_drones = mission_optimizer.num_drones
        self._last_result: MissionResult | None = None
        self._last_portions: list[float] | None = None

    def objective(self, portions: list[float]) -> float:
        result = self.mission_optimizer.evaluate(portions)
        self._last_result = result
        self._last_portions = portions
        return result.mission_time

    def optimize(
            self,
            initial_portions: list[float] = None,
            initial_temp: float = 1000.0,
            final_temp: float = 1e-10,
            cooling_rate: float = 0.95,
            step_size: float = 0.1,
            max_iterations: int = 1000,
            seed: int | None = None,
            iteration_callback: Any | None = None,
    ) -> dict[str, Any]:
        sa_optimizer = SimulatedAnnealingOptimizer(
            initial_temp=initial_temp,
            final_temp=final_temp,
            cooling_rate=cooling_rate,
            step_size=step_size,
            random_seed=seed
        )

        opt_result = sa_optimizer.optimize(
            objective_function=self.objective,
            initial_portions=initial_portions,
            max_iterations=max_iterations,
            iteration_callback=iteration_callback,
        )

        self._last_portions = opt_result['optimized_portions']
        self._last_result = self.mission_optimizer.evaluate(self._last_portions)

        return {
            'optimized_portions': opt_result['optimized_portions'],
            'best_time': opt_result['best_value'],
            'iterations': opt_result['iterations'],
            'final_temperature': opt_result['final_temperature'],
            'final_mission_result': self._last_result
        }

    def get_last_portions(self) -> list[float] | None:
        return self._last_portions

    def get_last_mission_result(self) -> MissionResult | None:
        return self._last_result

    @property
    def last_result(self):
        return self._last_result
