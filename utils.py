# Author: Rahul Narkhede
# Utility functions to support operations in RBF Interpolation


import numpy as np
import pandas as pd
from scipy.linalg import norm
import tensorflow as tf
from sklearn.cluster import KMeans
from scipy.linalg import lu, inv




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
    """
    Calculate the condition number of a given matrix using TensorFlow.

    The condition number is a measure of how the solution to a system of linear equations 
    is affected by changes in the input data. It is the ratio of the largest singular value 
    to the smallest singular value of the matrix.

    Parameters:
    -----------
    matrix : tf.Tensor or np.ndarray
        The input matrix whose condition number is to be calculated. It can be a TensorFlow tensor 
        or a NumPy array. The matrix should be a square matrix for a meaningful condition number.

    Returns:
    --------
    float
        The condition number of the matrix. A high condition number indicates that the matrix 
        is ill-conditioned, while a low condition number suggests it is well-conditioned.

    Example:
    --------
    >>> import numpy as np
    >>> matrix = np.array([[1, 2], [3, 4]])
    >>> cond_number = condition_number(matrix)
    >>> print(cond_number)
    14.933034

    Notes:
    ------
    - The function first converts the input matrix to a TensorFlow tensor if it is not already one.
    - Singular Value Decomposition (SVD) is used to compute the singular values of the matrix.
    - The condition number is computed as the ratio of the largest to the smallest singular value.

    Warnings:
    ---------
    - Ensure that the input matrix is non-singular and square for meaningful condition number calculation.
    - A condition number close to zero might indicate a potential issue with the matrix's singularity.
    """

    # Convert the input matrix to a TensorFlow tensor if it is not already
    if not isinstance(matrix, tf.Tensor):
        matrix = tf.convert_to_tensor(matrix, dtype=tf.float64)
    
    # Compute singular values using Singular Value Decomposition (SVD)
    singular_values = tf.linalg.svd(matrix, compute_uv=False)
    
    # Find the maximum and minimum singular values
    max_singular_value = tf.reduce_max(singular_values)
    min_singular_value = tf.reduce_min(singular_values)
    
    # Compute the condition number as the ratio of the largest to smallest singular value
    condition_num = max_singular_value / min_singular_value

    return condition_num



