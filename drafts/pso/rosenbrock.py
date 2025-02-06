import numpy as np
import matplotlib.pyplot as plt

def rosenbrock_2d(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2

# Create a grid of (x, y) values
x_vals = np.linspace(-2, 2, 400)
y_vals = np.linspace(-1, 3, 400)  # Adjusted for better visualization
X, Y = np.meshgrid(x_vals, y_vals)

# Compute Rosenbrock function over the grid
Z = rosenbrock_2d(X, Y)

# Plot contour
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(contour, label="Function Value")
plt.contour(X, Y, Z, levels=10, colors='black', linewidths=0.5)  # Add contour lines

# Labels and title
plt.title("Rosenbrock Function Contour Plot")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
