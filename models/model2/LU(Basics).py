import numpy as np

def lu_decomposition_no_pivot(A):
    """
    Performs LU decomposition without pivoting.
    Returns L (lower triangular with 1's on diagonal) and U (upper triangular)
    """
    n = A.shape[0]
    # Make a copy so we don't modify the original matrix
    U = A.astype(float).copy()
    L = np.eye(n)  # Identity matrix → will become lower triangular
    
    for k in range(n):
        # For each pivot position k
        for i in range(k+1, n):          # rows below pivot
            if U[k, k] == 0:
                raise ValueError("Zero pivot encountered — pivoting is needed!")
                
            factor = U[i, k] / U[k, k]   # multiplier (this goes into L)
            L[i, k] = factor
            
            # Row operation: subtract factor * pivot row from current row
            U[i, k:] = U[i, k:] - factor * U[k, k:]
    
    return L, U


# ────────────────────────────────────────────────
# Example usage
# ────────────────────────────────────────────────

A = np.array([
    [2, 1, 1],
    [4, -6, 0],
    [-2, 7, 2]
], dtype=float)

print("Original matrix A:")
print(A)

try:
    L, U = lu_decomposition_no_pivot(A)
    print("\nL:")
    print(np.round(L, decimals=4))
    print("\nU:")
    print(np.round(U, decimals=4))
    
    # Verify: L @ U should ≈ A
    print("\nL @ U (should ≈ A):")
    print(np.round(L @ U, decimals=4))
    
except ValueError as e:
    print("Error:", e)