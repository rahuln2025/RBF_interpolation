# Author: Rahul Narkhede
# Functions to perform RBF interpolation using Gaussian kernel with the closed form solution

import numpy as np
import pandas as pd
from scipy.linalg import norm
import tensorflow as tf
from sklearn.cluster import KMeans
from scipy.linalg import lu, inv



from utils import *

def select_points_grid(points, nx, ny):
    """
    Select a subset of points from a given set using grid sampling.

    This function samples points from a grid that covers the area defined by the 
    minimum and maximum x and y coordinates of the input points. The grid is created 
    with `nx` points along the x-direction, and the number of points along the y-direction
    is determined based on the aspect ratio of the point set.

    Parameters:
    -----------
    points : numpy.ndarray
        An array of shape (n, 2) where `n` is the number of original points. Each row represents a point
        with x and y coordinates.

    nx : int
        The number of grid points along the x-direction.

    Returns:
    --------
    numpy.ndarray
        An array of shape (nx * ny, 2) where `ny` is calculated based on the aspect ratio of the point set. 
        This array contains the selected points closest to the grid points.

    Notes:
    ------
    - The function calculates the aspect ratio of the input point set to determine the number of points
      along the y-direction (`ny`).
    - A grid is created with `nx` points in the x-direction and `ny` points in the y-direction.
    - For each grid point, the closest point from the original set is selected.

    Example:
    --------
    >>> points = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
    >>> selected_points = select_points_grid(points, nx=5)
    >>> print(selected_points)
    [[0. 0.]
     [1. 1.]
     [2. 2.]
     [3. 3.]
     [4. 4.]]
    """
    
    # Calculate aspect ratio from data
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    aspect_ratio = (x_max - x_min) / (y_max - y_min)

    # Calculate the number of points along the y-direction based on aspect ratio
    #ny = int(nx / aspect_ratio)
    #ny = len(points[:, 1])
    
    # Create a grid of points
    x_grid = np.linspace(x_min, x_max, nx)
    y_grid = np.linspace(y_min, y_max, ny)
    xv, yv = np.meshgrid(x_grid, y_grid)
    
    grid_points = np.vstack([xv.ravel(), yv.ravel()]).T

    # Find the closest point in the original set for each grid point
    selected_points = []
    for gp in grid_points:
        # Compute the Euclidean distance from the grid point to all original points
        distances = np.sum((points - gp) ** 2, axis=1)
        # Find the index of the closest point
        closest_point = points[np.argmin(distances)]
        selected_points.append(closest_point)

    return np.array(selected_points)



def select_points_kmeans(points, nx, ny):
    """
    Select a subset of points using K-Means clustering.

    This function uses K-Means clustering to select a subset of points from the original set. 
    The number of clusters is determined by the number of points along the x-direction (`nx`) 
    and the aspect ratio of the data. The resulting cluster centers are used as the selected points.

    Parameters:
    -----------
    points : numpy.ndarray
        An array of shape (n, 2) where `n` is the number of original points. Each row represents a point
        with x and y coordinates.

    nx : int
        The number of grid points along the x-direction. The total number of selected points will be
        calculated based on this value and the aspect ratio of the data.

    Returns:
    --------
    numpy.ndarray
        An array of shape (nx * ny, 2) where `ny` is calculated based on the aspect ratio of the point set. 
        This array contains the selected points as the cluster centers.

    Notes:
    ------
    - The aspect ratio of the data is used to determine the number of points along the y-direction (`ny`).
    - K-Means clustering is performed with `num_points` clusters, where `num_points = nx * ny`.
    - The cluster centers from the K-Means algorithm are used as the selected points.

    Example:
    --------
    >>> points = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
    >>> selected_points = select_points_kmeans(points, nx=5)
    >>> print(selected_points)
    [[0. 0.]
     [4. 4.]
     [2. 2.]]
    """
    
    # Calculate aspect ratio from data
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    aspect_ratio = (x_max - x_min) / (y_max - y_min)

    # Calculate the number of points along the y-direction based on aspect ratio
    #ny = int(nx / aspect_ratio)
    #ny = len(points[:, 1])
    
    # Total number of points to select
    num_points = nx * ny

    # Perform K-Means clustering to select the points
    kmeans = KMeans(n_clusters=num_points, random_state=42, n_init=3)
    kmeans.fit(points)
    selected_points = kmeans.cluster_centers_

    return selected_points



def homogenous_centers(data, nx, ny, plot=False):
    """
    Generate a grid of evenly spaced centers within the bounds of the provided data.

    This function creates a grid of points (centers) uniformly distributed across the x and y 
    dimensions of the data. The number of grid points along the x-direction (`nx`) is specified,
    and the number of grid points along the y-direction is calculated based on the aspect ratio of 
    the data's bounding box. Optionally, the function can plot the grid of centers.

    Parameters:
    -----------
    data : numpy.ndarray or pd.DataFrame
        The input data containing x and y coordinates. If a NumPy array, it should be of shape 
        (n, 2). If a DataFrame, it should contain columns 'x' and 'y'.

    nx : int
        The number of grid points along the x-direction.

    plot : bool, optional
        If True, a plot of the grid centers will be displayed. Default is False.

    Returns:
    --------
    numpy.ndarray
        An array of shape (nx * ny, 2) containing the coordinates of the grid centers.

    Notes:
    ------
    - The number of grid points along the y-direction (`ny`) is calculated based on the aspect ratio
      of the bounding box of the data.
    - The grid centers are created using `np.meshgrid` and then reshaped into the desired format.

    Example:
    --------
    >>> data = np.array([[1, 2], [3, 4], [5, 6]])
    >>> centers = homogenous_centers(data, nx=10, plot=True)
    >>> print(centers)
    [[1. 1.]
     [1. 2.]
     [1. 3.]
     ...
     [10. 10.]]
    """
    
    # Determine the bounds of the data
    if isinstance(data, np.ndarray):
        xmin, xmax = np.min(data[:, 0]), np.max(data[:, 0])
        ymin, ymax = np.min(data[:, 1]), np.max(data[:, 1])
    elif isinstance(data, pd.DataFrame):
        xmin, xmax = data.x.min(), data.x.max()
        ymin, ymax = data.y.min(), data.y.max()
    else:
        raise TypeError("Input data must be a numpy.ndarray or pandas.DataFrame.")
    
    # Calculate the aspect ratio of the data bounds
    lenx = xmax - xmin
    leny = ymax - ymin
    aspect_ratio = lenx / leny
    
    # Calculate the number of grid points along the y-direction
    #ny = int(nx / aspect_ratio)
    #ny = len(points[:, 1])
    
    # Generate grid points along x and y
    cx = np.linspace(xmin, xmax, nx, endpoint=True)
    cy = np.linspace(ymin, ymax, ny, endpoint=True)
    Cx, Cy = np.meshgrid(cx, cy)
    
    # Create a grid of centers
    centres = np.zeros((nx, ny, 2))
    centres[:, :, 0] = Cy
    centres[:, :, 1] = Cx
    
    # Optionally plot the grid of centers
    if plot:
        plt.figure(figsize=(12, 8))
        plt.plot(Cx, Cy, marker='o', color='k', lw=0.2, markersize=0.5, linestyle='none')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Centers in XY Space')
        plt.axis('square')
        plt.show()
    
    return centres.reshape(-1, 2)



def interpolate(x_vec, centers, lambdas, sigma):
    """
    Perform Radial Basis Function (RBF) interpolation for a set of points based on given centers and lambda values.

    This function calculates interpolated values at specified points using a Gaussian RBF kernel.

    Parameters:
    -----------
    x_vec : numpy.ndarray or tf.Tensor
        A 2D array or tensor of shape (n_points, n_dimensions) representing the coordinates at which to interpolate.
    
    centers : numpy.ndarray or tf.Tensor
        A 2D array or tensor of shape (n_centers, n_dimensions) representing the coordinates of the RBF centers.
    
    lambdas : numpy.ndarray or tf.Tensor
        A 1D array or tensor of shape (n_centers,) containing the weights (lambdas) associated with each RBF center.
    
    sigma : float or tf.Tensor
        The standard deviation of the Gaussian kernel used in the RBF interpolation.

    Returns:
    --------
    numpy.ndarray
        An array of interpolated values at the coordinates specified by `x_vec`.

    Notes:
    ------
    - If `centers` is a 1D tensor, it is treated as having a single center and the distance is computed accordingly.
    - The function handles both single and multiple RBF centers by adjusting the computation of distances and kernel values.
    - The result is converted to a NumPy array before being returned.

    Example:
    --------
    >>> x_vec = np.array([[0.5, 0.5], [1.0, 1.0]])
    >>> centers = np.array([[0, 0], [1, 1]])
    >>> lambdas = np.array([1.0, -1.0])
    >>> sigma = 0.1
    >>> interpolate(x_vec, centers, lambdas, sigma)
    array([0.9, -0.1])
    """

    # Convert input data to TensorFlow tensors
    xy = tf.constant(x_vec, dtype=tf.float64)
    c = tf.constant(centers, dtype=tf.float64)
    sigma = tf.constant(sigma, dtype=tf.float64)

    # Get dimensions
    n_points = len(x_vec)
    n_centres = len(centers)
    
    # Compute the Gaussian kernel based matrix
    if len(c.shape) == 1:
        # Single center case
        diff = xy - c
        dist_sq = tf.reduce_sum(tf.square(diff), axis=1)
        rbf = tf.exp(-dist_sq / (2 * tf.square(sigma))) * (1 / (2 * np.pi * tf.square(sigma)))
    else:
        # Multiple centers case
        diff = xy[:, tf.newaxis, :] - c[tf.newaxis, :, :]
        dist_sq = tf.reduce_sum(tf.square(diff), axis=2)
        rbf = tf.exp(-dist_sq / (2 * tf.square(sigma))) * (1 / (2 * np.pi * tf.square(sigma)))
    
    # Compute interpolated values
    s_xy = tf.reduce_sum(rbf * tf.transpose(lambdas), axis=1)
    
    return s_xy.numpy()



def rbf_interpolation_tf(data, centers, sigma):
    """
    Perform Radial Basis Function (RBF) interpolation using TensorFlow.

    This function performs RBF interpolation on the input data to estimate values at specified
    points based on a set of centers. It computes the interpolation weights (lambdas) using the
    exact solution of the least squares objective function and calculates the interpolated values,
    mean squared error (MSE), and residuals.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the input data with columns 'x', 'y', and 'z'. The 'x' and 'y' columns
        represent the coordinates, while the 'z' column contains the values to be interpolated.

    centers : numpy.ndarray
        Array of shape (m, 2) representing the centers at which interpolation is performed.

    sigma : float
        The standard deviation of the Gaussian kernel used in the RBF interpolation.

    Returns:
    --------
    tuple
        A tuple containing:
        - numpy.ndarray: Interpolated values at the data points.
        - numpy.ndarray: Computed interpolation weights (lambdas).
        - float: Condition number of the matrix used in the interpolation.
        - float: Mean squared error (MSE) of the interpolation.
        - numpy.ndarray: Residuals of the interpolation (difference between actual and interpolated values).

    Notes:
    ------
    - The function handles both the case of a single center and multiple centers.
    - For multiple centers, a Gaussian kernel matrix is computed and used in the interpolation.
    - The condition number of the matrix helps in diagnosing potential numerical stability issues.
    - MSE provides a measure of the interpolation error.

    Example:
    --------
    >>> data = pd.DataFrame({
    ...     'x': [1, 2, 3],
    ...     'y': [4, 5, 6],
    ...     'z': [7, 8, 9]
    ... })
    >>> centers = np.array([[1, 4], [2, 5], [3, 6]])
    >>> sigma = 1.0
    >>> s_xy, lambdas, cond_mat, mse, residual = rbf_interpolation_tf(data, centers, sigma)
    >>> print(s_xy)
    [7. 8. 9.]
    """
    
    # Convert input data to TensorFlow tensors
    xy = tf.constant(data[['x', 'y']].to_numpy(), dtype=tf.float64)
    c = tf.constant(centers, dtype=tf.float64)
    z = tf.constant(data['z'].to_numpy(), dtype=tf.float64)
    sigma = tf.constant(sigma, dtype=tf.float64)

    # Get dimensions
    n_points = len(data)
    n_centres = len(centers)
    
    # Compute Gaussian kernel based matrix
    if len(c.shape) == 1:
        # Single center case
        diff = xy - c
        dist_sq = tf.reduce_sum(tf.square(diff), axis=1)
        rbf = tf.exp(-dist_sq / (2 * tf.square(sigma))) * (1 / (2 * np.pi * tf.square(sigma)))
    else:
        # Multiple centers case
        diff = xy[:, tf.newaxis, :] - c[tf.newaxis, :, :]
        dist_sq = tf.reduce_sum(tf.square(diff), axis=2)
        rbf = tf.exp(-dist_sq / (2 * tf.square(sigma))) * (1 / (2 * np.pi * tf.square(sigma)))
    
    # Compute lambda values using least squares
    rbf_T = tf.transpose(rbf)
    rbf_T_rbf = tf.matmul(rbf_T, rbf)
    cond_mat = condition_number(rbf)  # Condition number of matrix to be inverted
    rbf_T_rbf_inv = tf.linalg.inv(tf.reshape(rbf_T_rbf, [1, n_centres, n_centres]))  # Inverse of matrix
    rbf_T_z = tf.matmul(rbf_T, tf.expand_dims(z, axis=1))

    lambdas = tf.matmul(rbf_T_rbf_inv, rbf_T_z)
    lambdas = tf.squeeze(lambdas)
    
    # Interpolate
    s_xy = tf.reduce_sum(rbf * tf.transpose(lambdas), axis=1)
    surface_pts = tf.concat([xy, tf.expand_dims(s_xy, axis=1)], axis=1)
    
    # Compute mean squared error (MSE)
    mse = tf.keras.losses.MSE(z, s_xy)
    residual = z - s_xy
    
    return s_xy.numpy(), lambdas.numpy(), cond_mat.numpy(), mse.numpy(), residual.numpy()


def rbf_interpolation(data, centers, sigma):
    """
    Perform Radial Basis Function (RBF) interpolation.

    This function performs RBF interpolation on the input data to estimate values at specified
    points based on a set of centers. It computes the interpolation weights (lambdas) using the
    exact solution of the least squares objective function and calculates the interpolated values,
    mean squared error (MSE), and residuals.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the input data with columns 'x', 'y', and 'z'. The 'x' and 'y' columns
        represent the coordinates, while the 'z' column contains the values to be interpolated.

    centers : numpy.ndarray
        Array of shape (m, 2) representing the centers at which interpolation is performed.

    sigma : float
        The standard deviation of the Gaussian kernel used in the RBF interpolation.

    Returns:
    --------
    tuple
        A tuple containing:
        - numpy.ndarray: Interpolated values at the data points.
        - numpy.ndarray: Computed interpolation weights (lambdas).
        - float: Condition number of the matrix used in the interpolation.
        - float: Mean squared error (MSE) of the interpolation.
        - numpy.ndarray: Residuals of the interpolation (difference between actual and interpolated values).

    Notes:
    ------
    - The function handles both the case of a single center and multiple centers.
    - For multiple centers, a Gaussian kernel matrix is computed and used in the interpolation.
    - The condition number of the matrix helps in diagnosing potential numerical stability issues.
    - MSE provides a measure of the interpolation error.

    Example:
    --------
    >>> data = pd.DataFrame({
    ...     'x': [1, 2, 3],
    ...     'y': [4, 5, 6],
    ...     'z': [7, 8, 9]
    ... })
    >>> centers = np.array([[1, 4], [2, 5], [3, 6]])
    >>> sigma = 1.0
    >>> s_xy, lambdas, cond_mat, mse, residual = rbf_interpolation(data, centers, sigma)
    >>> print(s_xy)
    [7. 8. 9.]
    """
    
    # Get data
    xy = data[['x', 'y']].to_numpy()
    c = centers
    z = data['z'].to_numpy()
    
    # Get dimensions
    n_points = len(data)
    n_centres = len(centers)
    
    # Compute Gaussian kernel based matrix
    if len(c.shape) == 1:
        # Single center case
        diff = xy - c
        dist_sq = norm(diff, ord=2, axis=1)**2
        rbf = np.exp(-dist_sq / (2 * sigma**2)) * (1 / (2 * np.pi * sigma**2))
    else:
        # Multiple centers case
        diff = xy[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist_sq = norm(diff, ord=2, axis=2)**2
        rbf = np.exp(-dist_sq / (2 * sigma**2)) * (1 / (2 * np.pi * sigma**2))
    
    # Compute lambdas using least squares
    rbf_T_rbf = np.dot(rbf.T, rbf)
    rbf_T_rbf_inv = inv(rbf_T_rbf)  # Matrix inversion
    lambdas = np.dot(rbf_T_rbf_inv, np.dot(rbf.T, z))
    
    # Interpolate
    s_xy = np.sum(rbf * lambdas.reshape(1, -1), axis=1)
    
    # Compute MSE and residuals
    mse = np.mean(np.square(z - s_xy))
    residual = z - s_xy
    
    # Compute condition number of the matrix used in interpolation
    cond_mat = np.linalg.cond(rbf_T_rbf)
    
    return s_xy, lambdas, cond_mat, mse, residual





