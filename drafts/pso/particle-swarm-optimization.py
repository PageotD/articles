import numpy as np
from typing import Callable, List, Tuple

import numpy as np

class Particle:
    """
    A particle is an object with a position and a velocity and which evolves in an arbitrary
    n-dimensional space.

    Attributes
    ----------
    position : np.ndarray
        The position of the particle in the search space
    velocity : np.ndarray
        The velocity of the particle in the search space
    best_position : np.ndarray
        The best position of the particle in the search space
    best_score : float
        The best score of the particle

    Methods
    --------
    update_velocity(self, gbest_position, inertia, cognitive, social):
        Update the velocity of the particle
    update_position(self, bounds: List[Tuple[float, float]]):
        Update the position of the particle
    evaluate(self, fitness_function: Callable):
        Evaluate the fitness of the particle
    """

    def __init__(self, bounds: List[Tuple[float, float]]) -> None:
        """
        Initialize the position, velocity, score, best position and best score of the particle

        Parameters
        ----------
        bounds : List[Tuple[float, float]]
            The bounds of the search space
        """

        # Initialize position of the particle in the search space and velocity to 0
        self.position = np.array([np.random.uniform(low, high) for low, high in bounds])
        self.velocity = np.zeros(len(bounds), dtype=np.float32)
        
        # Initialize score and best score to infinity
        self.score = float('inf')
        self.pbest_score = float('inf')

        # Initialize best position of the particle to its current position
        self.pbest_position = self.position.copy()
    
    def evaluate(self, fitness_function: Callable) -> None:
        """
        Evaluate the score of the particle and update the best position and score if the score is
        lower than the current best score

        Parameters
        ----------
        fitness_function : Callable
            The objective function to optimize
        """
        self.score = fitness_function(self.position)
        if self.score < self.pbest_score:
            self.pbest_score = self.score
            self.pbest_position = self.position.copy()

    def update_velocity(self, gbest_position, inertia, cognitive, social) -> None:
        """
        Update the velocity of the particle

        Parameters
        ----------
        gbest_position : np.ndarray
            The global best position of the swarm
        inertia : float
            The inertia weight
        cognitive : float
            The cognitive constant
        social : float
            The social constant
        """
        r1, r2 = np.random.rand(), np.random.rand()
        cognitive = cognitive * r1 * (self.pbest_position - self.position)
        social = social * r2 * (gbest_position - self.position)
        self.velocity = inertia * self.velocity + cognitive + social

    def update_position(self, bounds: List[Tuple[float, float]]) -> None:
        """
        Update the position of the particle

        Parameters
        ----------
        bounds : List[Tuple[float, float]]
            The bounds of the search space
        """
        self.position += self.velocity
        self.position = np.clip(self.position, [b[0] for b in bounds], [b[1] for b in bounds])

class Swarm:
    """
    A swarm is a collection of particles and which evolves in an arbitrary n-dimensional space.

    Attributes
    ----------
    particles : List[Particle]
        The particles of the swarm
    gbest_position : np.ndarray
        The global best position of the swarm
    gbest_value : float
        The global best score of the swarm
    fitness_function : Callable
        The objective function to optimize
    bounds : List[Tuple[float, float]]
        The bounds of the search space

    Methods
    --------
    optimize(self, num_iterations, inertia=0.7, cognitive=2.1, social=2.1):
        Optimize the objective function
    """

    def __init__(self, num_particles: int, bounds: List[Tuple[float, float]], fitness_function: Callable) -> None:
        """
        Initialize the swarm

        Parameters
        ----------
        num_particles : int
            The number of particles in the swarm
        bounds : List[Tuple[float, float]]
            The bounds of the search space
        fitness_function : Callable
            The objective function to optimize
        """
        self.particles = [Particle(bounds) for _ in range(num_particles)]
        self.gbest_position = None
        self.gbest_score = float('inf')
        self.fitness_function = fitness_function
        self.bounds = bounds

    def optimize(self, num_iterations: int, inertia: float=0.7, cognitive: float=2.1, social: float=2.1) -> Tuple[np.ndarray, float]:
        """
        Optimize the objective function

        Parameters
        ----------
        num_iterations : int
            The number of iterations
        inertia : float
            The inertia weight
        cognitive : float
            The cognitive constant
        social : float
            The social constant

        Returns
        -------
        Tuple[np.ndarray, float]
            The global best position and the global best score
        """
        for particle in self.particles:  # Initial evaluation of particles
            particle.evaluate(self.fitness_function)
            if particle.score < self.gbest_score:
                self.gbest_score = particle.score
                self.gbest_position = particle.position.copy()
            
        print(particle.score, self.gbest_score)
        
        for iter in range(num_iterations): # Start optimization
            print("ITER::", iter, self.gbest_score)
            for particle in self.particles:
                particle.update_velocity(self.gbest_position, inertia, cognitive, social)
                particle.update_position(self.bounds)
                particle.evaluate(self.fitness_function)
                print("EVALUATE::", particle.score)
                print(particle.score < self.gbest_score)
                if particle.score < self.gbest_score:
                    self.gbest_score = particle.score
                    self.gbest_position = particle.position.copy()

        return self.gbest_position, self.gbest_score

# Example usage:
def objective_function(x):
    """
    Simple sphere function
    
    Parameters
    ----------
    x : np.ndarray
        The position of the particle in the search space

    Returns
    -------
    float
        The fitness of the particle
    """
    return np.sum(x**2)  # Simple sphere function

dimensions = 3  # Can be 1, 2, or 3
bounds = [(-5, 5)] * dimensions
num_particles = 30
max_iterations = 100

swarm = Swarm(num_particles, bounds, objective_function)
best_position, best_score = swarm.optimize(max_iterations)

print(f"Best position: {best_position}")
print(f"Best score: {best_score}")
