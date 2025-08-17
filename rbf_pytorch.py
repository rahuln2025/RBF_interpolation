import numpy as np
import pandas as pd
from scipy.linalg import norm
import torch
from sklearn.cluster import KMeans
from scipy.linalg import lu, inv
import matplotlib.pyplot as plt

from utils_torch import *

def select_points_kmeans(points, nx, ny, plot = False):
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
        
    plot : bool, optional
        If True, plots the original points and selected centers. Default is False.

    Returns:
    --------
    numpy.ndarray
        An array of shape (nx * ny, 2) where `ny` is calculated based on the aspect ratio of the point set. 
        This array contains the selected points as the cluster centers.
    """
    
    # Calculate aspect ratio from data
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    aspect_ratio = (x_max - x_min) / (y_max - y_min)

    # Total number of points to select
    num_points = nx * ny
    # Perform K-Means clustering using GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    points_tensor = torch.FloatTensor(points).to(device)
    
    # Initialize centers randomly
    idx = torch.randperm(len(points_tensor))[:num_points]
    centers = points_tensor[idx].clone()
    
    # K-means iterations
    max_iters = 100
    for _ in range(max_iters):
        # Calculate distances
        distances = torch.cdist(points_tensor, centers)
        # Assign points to nearest center
        labels = torch.argmin(distances, dim=1)
        # Update centers
        new_centers = torch.zeros_like(centers)
        for k in range(num_points):
            mask = labels == k
            if mask.any():
                new_centers[k] = points_tensor[mask].mean(0)
            else:
                new_centers[k] = centers[k]
        # Check convergence
        if torch.all(centers == new_centers):
            break
        centers = new_centers
    
    selected_points = centers.cpu().numpy()

    # Plot if requested
    if plot:
        fig_width = 10  # Base width
        fig_height = fig_width / aspect_ratio  # Height adjusted by aspect ratio
        plt.figure(figsize=(fig_width, fig_height))
        plt.scatter(points[:, 0], points[:, 1], c='blue', alpha=0.8, s = 20, label='Original points')
        plt.scatter(selected_points[:, 0], selected_points[:, 1], c='red', marker='x', s=100, label='Centers')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'K-Means Centers (n={num_points})')
        plt.legend()
        plt.savefig("./centers.png")

    return selected_points


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


def rbf_interpolation_torch(data, centers, sigma):

    # convert data to torch tensors
    xy = torch.tensor(data[["x", "y"]].to_numpy(), dtype = torch.float32)
    c = torch.tensor(centers, dtype = torch.float32)
    z = torch.tensor(data["z"].to_numpy(), dtype = torch.float32)
    sigma = torch.tensor(sigma, dtype = torch.float32)

    # get dimensions
    n_points = len(data)
    n_centers = len(centers)

    # compute Gaussian kernels based matrix
    if len(c.shape) == 1: # single center
        diff = xy - c
        dist_sq = torch.sum(torch.square(diff), dim = 1)
        rbf = torch.exp(-dist_sq / (2 * torch.square(sigma))) * (1 / (2 * np.pi * torch.square(sigma)))
    else: # multiple centers case
        diff = xy.unsqueeze(1) - c.unsqueeze(0)
        dist_sq = torch.sum(torch.square(diff), dim=2)  # [n_points, n_centres]
        rbf = torch.exp(-dist_sq / (2 * torch.square(sigma))) * (1 / (2 * np.pi * torch.square(sigma)))


    # compute lambda values using lstsq
    rbf_T_rbf = torch.matmul(rbf.T, rbf).reshape(1, n_centers, n_centers)
    cond_mat = torch.linalg.cond(rbf_T_rbf, p = 2) #condition_number(rbf_T_rbf)

    # rbf_T_z = torch.matmul(rbf.T, z.unsqueeze(1))
    lambdas = torch.linalg.lstsq(rbf, z.unsqueeze(1))[0]
    lambdas = torch.squeeze(lambdas)

    # interpolate
    s_xy = torch.sum(rbf * lambdas, dim = 1)
    #surface_pts = torch.concat([xy, 
    #                            s_xy.unsqueeze(1)], dim = 1)

    # compute errors
    residual = z - s_xy
    mse = torch.mean(torch.square(z - s_xy))

    return s_xy.numpy(), lambdas.numpy(), cond_mat.numpy()[0], mse.numpy(), residual.numpy()

    
def interpolate_torch(x_vec, centers, lambdas, sigma):

    # convert data to torch tensors
    xy = torch.tensor(x_vec, dtype = torch.float32)
    c = torch.tensor(centers, dtype = torch.float32)
    sigma = torch.tensor(sigma, dtype = torch.float32)

    # Get dimensions
    n_points = len(x_vec)
    n_centres = len(centers)

    # compute Gaussian kernels based matrix
    if len(c.shape) == 1: # single center
        diff = xy - c
        dist_sq = torch.sum(torch.square(diff), dim = 1)
        rbf = torch.exp(-dist_sq / (2 * torch.square(sigma))) * (1 / (2 * np.pi * torch.square(sigma)))
    else: # multiple centers case
        diff = xy.unsqueeze(1) - c.unsqueeze(0)
        dist_sq = torch.sum(torch.square(diff), dim=2)  # [n_points, n_centres]
        rbf = torch.exp(-dist_sq / (2 * torch.square(sigma))) * (1 / (2 * np.pi * torch.square(sigma))) 

    s_xy = torch.sum(rbf * lambdas, dim = 1)

    return s_xy.numpy()      