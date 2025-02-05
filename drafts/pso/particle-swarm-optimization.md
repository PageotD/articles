> **Article in progress**

# Particle Swarm Optimization and It's implementation in Python

_An easy entry point to global optimization._

---

I've been working in DevOps for a few years now. But that wasn't always the case. In a “previous life”, I worked in geophysics, more specifically in seismic imaging. 

Despite my new job, geophysics is a subject that has never left me, and to which I return regularly. Whether it's implementing an eikonal solver, a 2D finite-difference wave propagation engine or using local or global optimization.

Recently, I thought it would be nice to document and share some of these topics and make an effort to make them accessible to everyone. 

So I've decided to write small articles that I'll share regularly (as much as possible), starting with a subject that seems relatively simple to me: Particle Swarm Optimization and its implementation in Python.

Heard of particle swarm optimization (PSO) but don't know where to start? In this article, we'll explore how PSO works, in its most basic form, and implement it in Python.

## 1. What is Particle Swarm Optimization?

Particle Swarm Optimization (PSO), proposed by Eberhart and Kennedy in 1995 [1], is a global optimization algorithm designed to simulate the behavoir of flocking birds or school of fish.

PSO is used to solve optimization problems in many scientific and engineering domains, including [2]:
- antenna design
- biological, medical and pharmaceutical applications 
- design and optimisation of communication networks
- neural network training
- robotics  
- and much more.

## 2. Why PSO is useful?
Unlike local optimization techniques (like Gradient Descent), PSO does not require derivatives, making it suitable for messy, real-world functions with multiple peaks and valleys.

PSO is a great entry point into optimization because of its intuitive mechanics and broad applicability. 

It's a simple algorithm that is easy to implement and understand, making it a good choice for beginners and anyone who wants to try out optimization algorithms.

Another advantage is that PSO can be easily parallelized, making it a great choice for distributed computing environments.

Compare to local optimization algorithms, PSO does not rely on the gradient of the objective function, making it a good choice for problems where the gradient is not available.

What sets PSO apart from other global optimization methods (such as Genetic Algorithms or Simulated Annealing) is its swarm-based approach. Instead of relying on operations like mutation or crossover, PSO updates solutions by mimicking how individuals in a group learn from their own experiences and the success of others. This results in fast convergence and robust performance, making it a go-to choice for engineering, machine learning, and more.

| Feature | PSO | Other global optimization (GA, SA) | Local Optimization (Gradient Descent) |
| :---: | :---: | :---: | :---: |
| Derivative-Free?	|  Yes |  Yes |  No |
| Handles Local Minima?	|  Yes	| Yes | No (can get stuck) |
| Speed	| Fast	| Varies |Very fast |


## 3. How does PSO work?

The principle of PSO is quite simple and consists in changing the position and velocity of each particle in the swarm to improve the overall best position. In its simple form, PSO consists of the two following equations:

$$x_{i}^{k} = x_{i}^{k-1} + v_{i}^{k}\ ,$$
$$v_{i}^{k} = \omega v_{i}^{k} + c_{1} r_{1} (p_{g}^{k-1} - x_{i}^{k-1}) + c_{2} r_{2} (p_{i}^{k-1} - x_{i}^{k-1})\ ,$$

where:
- $x_{i}^{k}$ is the position of particle $i$ at iteration $k$
- $v_{i}^{k}$ is the velocity of particle $i$ at iteration $k$
- $p_{g}^{k-1}$ is the best global position reached by the swarm
- $p_{i}^{k-1}$ is the best position reached by particle $i$
- $r_{1}$ and $r_{2}$ are random numbers between 0 and 1
- $\omega$ is the inertia weight
- $c_{1}$ and $c_{2}$ are the social and cognitive constants respectively

The velocity formula can be split into three terms:
- **Inertia term**: $\omega v_{i}^{k}$ <br>
    which controls how much the particle's velocity is influenced by its own previous velocity
- **Social term**: $c_{1} r_{1} (p_{g}^{k-1} - x_{i}^{k-1})$ <br>
    which controls how much the particle is influenced by the swarm's best position
- **Cognitive term**: $c_{2} r_{2} (p_{i}^{k-1} - x_{i}^{k-1})$ <br>
    which controls how much the particle is influenced by its own best position

The basic workflow can be described in few steps:
1. Initialize the swarm with random positions and velocities
2. Evaluate the fitness of each particle
3. Update the best position and velocity of each particle
4. Update the swarm best position
5. Repeat steps 2-4 until a stopping criterion is met

## 4. How to implement PSO in Python?

Implementing PSO in Python takes litterally few lines of code. Let's start by defining the needs for the Particle class. To define a particle, we need:
- its position in an arbitrary n-dimensional space
- its velocity in the same space
- its best position
- its best score

To handle arbitrary n-dimensional space, we'll use a list of tuples which represents the bounds of each dimension of the search space: `bounds: List[Tuple[float, float]]`. Consequently, the current position, the best position and the velocity of each particle will be of size `n=len(bounds)`.

```python
class Particle:

    def __init__(self, bounds: List[Tuple[float, float]]):

        # First we initialize the position of the particle in the search space
        self.position = np.array([np.random.uniform(low, high) for low, high in bounds])

        # Then we initialize the velocity of the particle to 0
        self.velocity = np.zeros(len(bounds))

        # Then we initialize the best position of the particle to its current position
        self.best_position = self.position.copy()

        # Then we initialize the best score of the particle to infinity
        self.best_score = float('inf')
```

Then we need to define a method to update the velocity of each particle. This method will take as input the global best position, the inertia weight, the cognitive and social constants. To avoid particles to be out of bounds of the search space, we'll use the `np.clip` function.

```python
def update_velocity(self, global_best, w=0.7, c1=1.5, c2=1.5):
    r1, r2 = np.random.rand(), np.random.rand()
    cognitive = c1 * r1 * (self.best_position - self.position)
    social = c2 * r2 * (global_best - self.position)
    self.velocity = w * self.velocity + cognitive + social
    # np.clip for each bound of each dimension
    self.velocity = np.clip(self.velocity, [b[0] for b in bounds], [b[1] for b in bounds])
```

Let's start with implementing the `Particle` class:

```python
import numpy as np

class Particle:

    def __init__(self, dim, bounds):
        self.position = np.random.uniform(xbounds[0], xbounds[1], dim)
        self.velocity = np.zeros(dim)
        self.best_position = self.position.copy()
        self.best_score = rastrigin(self.position)

    def update_velocity(particle, global_best, w=0.7, c1=1.5, c2=1.5):
        r1, r2 = np.random.rand(), np.random.rand()
        cognitive = c1 * r1 * (particle.best_position - particle.position)
        social = c2 * r2 * (global_best - particle.position)
        particle.velocity = w * particle.velocity + cognitive + social

# This part is optional and can be used to plot the resulting swarm in a 2D space bounded by -5.12 and 5.12 in each dimension.
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    particle = Particle(20, (-5.12, 5.12))
    plt.scatter(particle.position[0], particle.position[1], color="black", s=50)
    plt.show()
```

In this class, we describe the particle's position, velocity, best position and best score. We also define a function to update the particle's velocity based on the global best position, the cognitive and social constants and the inertia weight.

Now we can implement the `Swarm` class where we can pass a function to optimize:

```python
class Swarm:
    def __init__(self, num_particles, dim, bounds, max_iter, optimize_function):
        xbounds = bounds[0]
        ybounds = bounds[1]
        self.particles = [Particle(dim, bounds) for _ in range(num_particles)]
        self.global_best = min(self.particles, key=lambda p: p.best_score).best_position
        self.optimize_function = optimize_function
        self.max_iter = max_iter

    def optimize(self):
        for _ in range(self.max_iter):
            for particle in self.particles:
                self.update_velocity(particle, self.global_best)
                particle.position += particle.velocity
                score = self.optimize_function(particle.position)

                if score < particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position.copy()

            self.global_best = min(self.particles, key=lambda p: p.best_score).best_position

        return self.global_best
```

In this class, we describe the swarm's particles, the global best position and the function to optimize. We also define a function to update the swarm's best position and the particles' velocity.

Now we can choose a 2D benchmark function and optimize it with PSO:

```python
def rastrigin(x):
    return 10 * len(x) + sum(x**2 - 10 * np.cos(2 * np.pi * x))

bounds = (-5.12, 5.12)
swarm = Swarm(num_particles=30, dim=2, bounds=bounds, max_iter=100, optimize_function=rastrigin)
best_solution = swarm.optimize()
print("Meilleure solution trouvée :", best_solution)
```

Congrats!You have successfully implemented PSO in Python and optimized a 2D benchmark function. :smile:

## References

[1] Eberhart, R. H. and Kennedy, J. C. (1995). Particle swarm optimization. IEEE Transactions on Evolutionary Computation, 3(2), 182-197.

[2] POLI, Riccardo. An analysis of publications on particle swarm optimization applications. Essex, UK: Department of Computer Science, University of Essex, 2007.
