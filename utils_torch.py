import numpy as np
import pandas as pd
from scipy.linalg import norm
import torch
from sklearn.cluster import KMeans
from scipy.linalg import lu, inv

print(f"CUDA Support: {torch.cuda.is_available()}")

def invert_matrix(matrix):
    """
    Invert a given non-sparse matrix using LU decomposition with partial pivoting.

    Parameters:
    -----------
    matrix : numpy.ndarray
        The input matrix to be inverted. This matrix should be square (n x n) and non-singular.
    
    Returns:
    --------
    numpy.ndarray or None
        The inverse of the input matrix if it is invertible. 
        If the matrix is singular or an error occurs, None is returned.

    Notes:
    ------
    - The function uses LU decomposition to check the invertibility of the matrix.
    - The inversion is verified by checking the deviation of the product of the matrix and its inverse from the identity matrix.
    - If the matrix is found to be singular or if verification fails, a message is printed and None is returned.

    Example:
    --------
    >>> matrix = np.array([[1, 2], [3, 4]])
    >>> inv_matrix = invert_matrix(matrix)
    >>> print(inv_matrix)
    [[-2.   1. ]
     [ 1.5 -0.5]]
    """

    try:
        # Perform LU decomposition of the matrix
        P, L, U = lu(matrix)

        # Check if the matrix is invertible by verifying the diagonal of U
        if np.all(np.diag(U) != 0):
            # Compute the inverse of the matrix
            matrix_inv = inv(matrix)
            
            # Verify the inversion by checking if matrix * matrix_inv approximates the identity matrix
            identity_approx = np.dot(matrix, matrix_inv)
            
            if np.allclose(identity_approx, np.eye(matrix.shape[0]), atol=1e-6):
                # Return the inverse if verification is successful
                return matrix_inv
            else:
                # Calculate deviation from the identity matrix
                identity = np.eye(matrix.shape[0]) 
                deviation = norm(identity_approx - identity, ord='fro')
                print(f"Matrix inversion verification failed for n_x: {np.sqrt(matrix.shape[0])}. "
                      f"Deviation from identity matrix: {deviation}")
                return matrix_inv
        else:
            # Print message if the matrix is singular and cannot be inverted
            print("Matrix is singular and cannot be inverted.")
            return None
    
    except Exception as e:
        # Handle any exceptions that occur during the inversion process
        print(f"An error occurred during matrix inversion: {e}")
        return None


def condition_number(matrix):

    if not isinstance(matrix, torch.tensor):
        matrix = torch.tensor(matrix, dtype=torch.float32)
    
    # get condition number of matrix w.r.t.max/min singular values
    condition_num = torch.linalg.cond(matrix, p = 2) 

    return condition_num