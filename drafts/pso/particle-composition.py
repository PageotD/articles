import matplotlib.pyplot as plt
import numpy as np

def plot_particle_movement(ax, inertia, social, cognitive):
    # Initial position
    initial_pos = np.array([0, 0])
    
    # Velocity components
    inertia_vel = np.array([inertia * 0.5, inertia * -0.25])
    social_vel = np.array([social * 0.5, social * 0.5])
    cognitive_vel = np.array([cognitive * -0.75, cognitive * -0.25])
    
    # New position
    new_pos = initial_pos + inertia_vel + social_vel + cognitive_vel
    
    # Plotting
    ax.quiver(*initial_pos, *inertia_vel, angles='xy', scale_units='xy', scale=1, color='r', label='Inertia')
    ax.quiver(*initial_pos, *social_vel, angles='xy', scale_units='xy', scale=1, color='g', label='Social')
    ax.quiver(*initial_pos, *cognitive_vel, angles='xy', scale_units='xy', scale=1, color='b', label='Cognitive')
    ax.quiver(*initial_pos, *(new_pos - initial_pos), angles='xy', scale_units='xy', scale=1, color='k', label='Resultant')
    
    ax.plot(*initial_pos, 'ko', markersize=10, label='Initial Position')
    ax.plot(*new_pos, 'ko', markersize=10, fillstyle='none', label='New Position')
    
    ax.set_xlim(-1, 1.5)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend(fontsize='small')
    ax.set_title(f"Inertia: {inertia}, Social: {social}, Cognitive: {cognitive}")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")

# Create one figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Plot three different scenarios
plot_particle_movement(ax1, inertia=1.0, social=1.0, cognitive=0.5)
plot_particle_movement(ax2, inertia=1.0, social=0.5, cognitive=1.0)
plot_particle_movement(ax3, inertia=0.5, social=1.0, cognitive=1.0)

plt.tight_layout()
plt.show()
