import numpy as np
from typing import Callable, List, Tuple

class Particle:
    def __init__(self, dimensions: int, bounds: List[Tuple[float, float]]):
        self.position = np.array([np.random.uniform(low, high) for low, high in bounds])
        self.velocity = np.zeros(dimensions)
        self.best_position = self.position.copy()
        self.best_score = float('inf')

    def update_velocity(self, global_best_position: np.ndarray, w: float, c1: float, c2: float):
        r1, r2 = np.random.rand(2)
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self, bounds: List[Tuple[float, float]]):
        self.position += self.velocity
        self.position = np.clip(self.position, [b[0] for b in bounds], [b[1] for b in bounds])

    def evaluate(self, objective_function: Callable):
        score = objective_function(self.position)
        if score < self.best_score:
            self.best_score = score
            self.best_position = self.position.copy()
        return score

class Swarm:
    def __init__(self, num_particles: int, dimensions: int, bounds: List[Tuple[float, float]], 
                 objective_function: Callable, w: float = 0.7, c1: float = 1.4, c2: float = 1.4):
        self.particles = [Particle(dimensions, bounds) for _ in range(num_particles)]
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.objective_function = objective_function
        self.bounds = bounds
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def optimize(self, max_iterations: int):
        for _ in range(max_iterations):
            for particle in self.particles:
                score = particle.evaluate(self.objective_function)
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = particle.position.copy()

            for particle in self.particles:
                particle.update_velocity(self.global_best_position, self.w, self.c1, self.c2)
                particle.update_position(self.bounds)

        return self.global_best_position, self.global_best_score

# Example usage:
def objective_function(x):
    return np.sum(x**2)  # Simple sphere function

dimensions = 3  # Can be 1, 2, or 3
bounds = [(-5, 5)] * dimensions
num_particles = 30
max_iterations = 100

swarm = Swarm(num_particles, dimensions, bounds, objective_function)
best_position, best_score = swarm.optimize(max_iterations)

print(f"Best position: {best_position}")
print(f"Best score: {best_score}")
