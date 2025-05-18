def matmul(A, B):
    """
    Performs matrix multiplication of A * B from scratch.
    
    Parameters:
    A (list of lists): The first matrix
    B (list of lists): The second matrix
    
    Returns:
    list of lists: The result of A * B
    
    Raises:
    ValueError: If the matrices have incompatible dimensions
    """
    # Check if A and B are valid matrices
    if not A or not all(isinstance(row, list) for row in A):
        raise ValueError("Matrix A must be a non-empty list of lists")
    if not B or not all(isinstance(row, list) for row in B):
        raise ValueError("Matrix B must be a non-empty list of lists")
    
    # Get the dimensions of the matrices
    m = len(A)       # Number of rows in A
    n = len(A[0])    # Number of columns in A (should equal rows in B)
    
    # Check if all rows in A have the same length
    if not all(len(row) == n for row in A):
        raise ValueError("Matrix A has rows of inconsistent length")
    
    # Check if B is compatible with A for multiplication
    if len(B) != n:
        raise ValueError(f"Matrix B must have {n} rows to multiply with A")
    
    # Get number of columns in B
    p = len(B[0])
    
    # Check if all rows in B have the same length
    if not all(len(row) == p for row in B):
        raise ValueError("Matrix B has rows of inconsistent length")
    
    # Initialize the result matrix with zeros
    result = [[0 for _ in range(p)] for _ in range(m)]
    
    # Perform matrix multiplication
    for i in range(m):      # For each row in A
        for j in range(p):  # For each column in B
            # Calculate the dot product of row i from A and column j from B
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]
    
    return result


def print_matrix(matrix):
    """
    Print a matrix in a visually appealing format.
    
    Parameters:
    matrix (list of lists): The matrix to print
    """
    for row in matrix:
        print("[", end=" ")
        for i, val in enumerate(row):
            if i < len(row) - 1:
                print(f"{val:5.2f},", end=" ")
            else:
                print(f"{val:5.2f}", end=" ")
        print("]")


# Example usage
if __name__ == "__main__":
    # Example 1: 2x3 * 3x2 matrices
    A = [
        [1, 2, 3],
        [4, 5, 6]
    ]
    
    B = [
        [7, 8],
        [9, 10],
        [11, 12]
    ]
    
    print("Matrix A:")
    print_matrix(A)
    print("\nMatrix B:")
    print_matrix(B)
    
    C = matmul(A, B)
    print("\nResult of A * B:")
    print_matrix(C)
    
    # Example 2: 3x3 * 3x1 matrices (vector multiplication)
    D = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    
    E = [
        [2],
        [3],
        [4]
    ]
    
    print("\nMatrix D:")
    print_matrix(D)
    print("\nMatrix E:")
    print_matrix(E)
    
    F = matmul(D, E)
    print("\nResult of D * E:")
    print_matrix(F)
    
    # Example 3: Identity matrix multiplication
    I = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
    
    G = [
        [5, 6, 7],
        [8, 9, 10],
        [11, 12, 13]
    ]
    
    print("\nIdentity Matrix I:")
    print_matrix(I)
    print("\nMatrix G:")
    print_matrix(G)
    
    H = matmul(I, G)
    print("\nResult of I * G (should be G):")
    print_matrix(H)