import numpy as np
import matplotlib.pyplot as plt

w = 0.7
c1 = 1.0
c2 = 1.0

ppos = [0, 0]
pbest = [1, 2]
gbest = [2, 1]
pvel = [-1, 0]

futur_pos_x = ppos[0] + w*pvel[0] + c1*(pbest[0] - ppos[0]) + c2*(gbest[0] - ppos[0])
futur_pos_y = ppos[1] + w*pvel[1] + c1*(pbest[1] - ppos[1]) + c2*(gbest[1] - ppos[1])
futur_pos = [futur_pos_x, futur_pos_y]
plt.subplot(131)
plt.xlim(-2, 3)
plt.ylim(-1, 4)
plt.quiver(ppos[0], ppos[1], pvel[0], pvel[1], color="grey", angles="xy", scale_units="xy", scale=1)
plt.quiver(ppos[0], ppos[1], pbest[0], pbest[1], color="green", angles="xy", scale_units="xy", scale=1)
plt.quiver(ppos[0], ppos[1], gbest[0], gbest[1], color="red", angles="xy", scale_units="xy", scale=1)
plt.scatter(futur_pos[0], futur_pos[1], color="blue", s=50)
plt.scatter(ppos[0], ppos[1], color="black", s=50)


pbest = [1.25, 2.5]
futur_pos_x = ppos[0] + w*pvel[0] + c1*(pbest[0] - ppos[0]) + c2*(gbest[0] - ppos[0])
futur_pos_y = ppos[1] + w*pvel[1] + c1*(pbest[1] - ppos[1]) + c2*(gbest[1] - ppos[1])
futur_pos = [futur_pos_x, futur_pos_y]
plt.subplot(132)
plt.xlim(-2, 3)
plt.ylim(-1, 4)
plt.quiver(ppos[0], ppos[1], pvel[0], pvel[1], color="grey", angles="xy", scale_units="xy", scale=1)
plt.quiver(ppos[0], ppos[1], pbest[0], pbest[1], color="green", angles="xy", scale_units="xy", scale=1)
plt.quiver(ppos[0], ppos[1], gbest[0], gbest[1], color="red", angles="xy", scale_units="xy", scale=1)
plt.scatter(futur_pos[0], futur_pos[1], color="blue", s=50)
plt.scatter(ppos[0], ppos[1], color="black", s=50)


plt.show()

# # I want to draw vectors of the composition of the particles in particle swarm optimization
# # In 2D for simplicity
# # I want to draw vectors of the composition of the particles in particle swarm optimization
# # In 2D for simplicity

# # particle position
# x = (0, 0)
# # particle velocity
# v = (1, 0.5)
# # particle best position
# pbest = (1.5, 2)
# # swarm best position
# gbest = (2, -1)
# # inertia weight

# inertia = (-0.5, 0.5)

# # updated position
# xupd = (x[0] + inertia[0] + pbest[0] + gbest[0], x[1] + inertia[1] + pbest[1] + gbest[1])

# # Draw the particle first
# plt.xlim(-3, 3)
# plt.ylim(-3, 3)
# plt.scatter(x[0], x[1], color="black", s=50)

# # Draw the velocity vector
# plt.quiver(x[0], x[1], v[0], v[1], color="blue", angles="xy", scale_units="xy", scale=1)

# # Draw the pbest vector
# plt.quiver(x[0], x[1], pbest[0], pbest[1], color="green", angles="xy", scale_units="xy", scale=1)

# # Draw the gbest vector
# plt.quiver(x[0], x[1], gbest[0], gbest[1], color="red", angles="xy", scale_units="xy", scale=1)

# # Draw the inertia vector
# plt.quiver(x[0], x[1], inertia[0], inertia[1], color="yellow", angles="xy", scale_units="xy", scale=1)

# # plot scatter empty xupd
# plt.scatter(xupd[0], xupd[1], color="black", s=50) 

# #plt.show()
# plt.savefig("particle_composition.png")