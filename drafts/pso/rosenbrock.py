import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the Rosenbrock function
def rosenbrock_2d(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2

# Create a grid of (x, y) values
x_vals = np.linspace(-2, 2, 1001)
y_vals = np.linspace(-1, 3, 1001)
X, Y = np.meshgrid(x_vals, y_vals)
Z = rosenbrock_2d(X, Y)

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': None})

# 2D Contour Plot
ax1 = axes[0]
contour = ax1.contourf(X, Y, Z, levels=500, cmap='rainbow') 
ax1.contour(X, Y, Z, levels=25, colors='black', linewidths=0.5)
fig.colorbar(contour, ax=ax1, label="Function Value")
ax1.set_title("Rosenbrock Function - Contour Plot")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# 3D Surface Plot
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X, Y, Z, cmap='rainbow', edgecolor='none', alpha=0.9)
ax2.set_title("Rosenbrock Function - 3D Surface")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("Function Value")

# Show the plots
plt.tight_layout()
plt.show()

# Save the plots as images
fig.savefig("rosenbrock.png")