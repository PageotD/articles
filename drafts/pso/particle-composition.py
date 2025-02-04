import numpy as np
import matplotlib.pyplot as plt

# I want to draw vectors of the composition of the particles in particle swarm optimization
# In 2D for simplicity
# I want to draw vectors of the composition of the particles in particle swarm optimization
# In 2D for simplicity

# particle position
x = (0, 0)
# particle velocity
v = (1, 0.5)
# particle best position
pbest = (1.5, 2)
# swarm best position
gbest = (2, -1)
# inertia weight

inertia = (-0.5, 0.5)

# updated position
xupd = (x[0] + inertia[0] + pbest[0] + gbest[0], x[1] + inertia[1] + pbest[1] + gbest[1])

# Draw the particle first
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.scatter(x[0], x[1], color="black", s=50)

# Draw the velocity vector
plt.quiver(x[0], x[1], v[0], v[1], color="blue", angles="xy", scale_units="xy", scale=1)

# Draw the pbest vector
plt.quiver(x[0], x[1], pbest[0], pbest[1], color="green", angles="xy", scale_units="xy", scale=1)

# Draw the gbest vector
plt.quiver(x[0], x[1], gbest[0], gbest[1], color="red", angles="xy", scale_units="xy", scale=1)

# Draw the inertia vector
plt.quiver(x[0], x[1], inertia[0], inertia[1], color="yellow", angles="xy", scale_units="xy", scale=1)

# plot scatter empty xupd
plt.scatter(xupd[0], xupd[1], color="black", s=50) 

plt.show()