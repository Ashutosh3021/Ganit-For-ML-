from scipy.linalg import lu
import numpy as np
A = np.array([
    [2, 1, 1],
    [4, -6, 0],
    [-2, 7, 2]
], dtype=float)
P, L, U = lu(A)   # scipy version returns P,L,U directly
print("Original matrix A:")
print(A)
print("\nP:")
print(np.round(P, decimals=4))
print("\nL:")
print(np.round(L, decimals=4))
print("\nU:")
print(np.round(U, decimals=4))