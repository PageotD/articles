import matplotlib.pyplot as plt
import numpy as np

def plot_particle_movement(inertia, social, cognitive):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Initial position
    initial_pos = np.array([0, 0])
    
    # Velocity components
    inertia_vel = np.array([inertia * 0.5, inertia * 0.])
    social_vel = np.array([social * 0.5, social * 0.5])
    cognitive_vel = np.array([cognitive * -0.25, cognitive * -0.5])
    
    # New position
    new_pos = initial_pos + inertia_vel + social_vel + cognitive_vel
    
    # Plotting
    ax.quiver(*initial_pos, *inertia_vel, angles='xy', scale_units='xy', scale=1, color='r', label='Inertia')
    ax.quiver(*initial_pos, *social_vel, angles='xy', scale_units='xy', scale=1, color='g', label='Social')
    ax.quiver(*initial_pos, *cognitive_vel, angles='xy', scale_units='xy', scale=1, color='b', label='Cognitive')
    ax.quiver(*initial_pos, *(new_pos - initial_pos), angles='xy', scale_units='xy', scale=1, color='k', label='Resultant')
    
    ax.plot(*initial_pos, 'ko', markersize=10, label='Initial Position')
    ax.plot(*new_pos, 'ko', markersize=10, fillstyle='none', label='New Position')
    
    ax.set_xlim(-1, 3)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    ax.set_title(f"Particle Movement (Inertia: {inertia}, Social: {social}, Cognitive: {cognitive})")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    
    plt.tight_layout()
    #plt.show()

# Example usage
plot_particle_movement(inertia=1.0, social=1.0, cognitive=0.5)
plot_particle_movement(inertia=1.0, social=0.5, cognitive=1.0)
plot_particle_movement(inertia=0.15, social=1.0, cognitive=1.0)
plt.show()
