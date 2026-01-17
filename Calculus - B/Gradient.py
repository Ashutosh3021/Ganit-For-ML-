import sympy as sp

# Define variables
x, y = sp.symbols('x y')

# Define function
f = x**2 + 3*y**2

# Compute gradient
grad_f = [sp.diff(f, var) for var in (x, y)]
print("Gradient:", grad_f)

# Convert symbolic expressions to functions
grad_f_func = [sp.lambdify((x, y), expr) for expr in grad_f]

# Evaluate at (1,2)
grad_value = [func(1, 2) for func in grad_f_func]
print("Gradient at (1,2):", grad_value)