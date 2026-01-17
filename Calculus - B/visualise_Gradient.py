import numpy as np
import matplotlib.pyplot as plt

# Generate grid
X, Y = np.meshgrid(np.linspace(-3, 3, 10), np.linspace(-3, 3, 10))

# Compute gradients
U = 2 * X  # ∂f/∂x
V = 6 * Y  # ∂f/∂y

# Plot gradient field
plt.figure(figsize=(6,6))
plt.quiver(X, Y, U, V, color='r', angles='xy')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Field of f(x,y) = x² + 3y²')
plt.grid()
plt.show()
