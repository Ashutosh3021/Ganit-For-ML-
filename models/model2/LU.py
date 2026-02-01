import numpy as np
def lu_decomposition_partial_pivot(A):
    """
    LU decomposition with partial pivoting.
    Returns P (permutation matrix), L, U
    """
    n = A.shape[0]
    A = A.astype(float).copy()
    
    # We'll store pivots and permutation
    P = np.eye(n)          # permutation matrix
    L = np.eye(n)
    U = A.copy()
    
    for k in range(n):
        # Find pivot row (largest absolute value in column k, from row k downward)
        max_row = np.argmax(np.abs(U[k:, k])) + k
        
        if max_row != k:
            # Swap rows in U
            U[[k, max_row]] = U[[max_row, k]]
            # Swap rows in P
            P[[k, max_row]] = P[[max_row, k]]
            # For rows already processed, swap in L (below diagonal)
            if k > 0:
                L[[k, max_row], :k] = L[[max_row, k], :k]
        
        pivot = U[k, k]
        if abs(pivot) < 1e-12:
            raise ValueError("Matrix is singular or nearly singular")
        
        for i in range(k+1, n):
            factor = U[i, k] / pivot
            L[i, k] = factor
            U[i, k:] = U[i, k:] - factor * U[k, k:]
    
    return P, L, U


# ────────────────────────────────────────────────
# Test with a matrix that needs pivoting
# ────────────────────────────────────────────────

A2 = np.array([
    [0, 2, 3],
    [4, 5, 6],
    [7, 8, 10]
], dtype=float)

print("\nMatrix that needs pivoting:")
print(A2)

P, L, U = lu_decomposition_partial_pivot(A2)

print("\nP:")
print(P)
print("\nL:")
print(np.round(L, 4))
print("\nU:")
print(np.round(U, 4))

# Verify: P @ A = L @ U
print("\nP @ A:")
print(np.round(P @ A2, 4))
print("\nL @ U:")
print(np.round(L @ U, 4))