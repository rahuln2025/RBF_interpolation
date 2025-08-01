# RBF Interpolation Dashboard

A Dash-based web application for visualizing data and performing surface interpolation using Radial Basis Function (RBF) interpolation with Gaussian kernels.

## Author
- **Rahul Narkhede**

## Description

This project provides an interactive web interface for:
- Loading and visualizing 3D data from CSV/TXT files
- Performing RBF interpolation with automatic parameter optimization
- Displaying 3D surface plots with original data scatter points
- Showing projection plots along X and Y axes

## Features

- Interactive column selection for X, Y, and Z coordinates
- Optional data scaling between [0, 1]
- Automatic parameter optimization using cross-validation
- 3D surface visualization with Plotly
- Projection plots for better data understanding
- Responsive web interface using Dash and Bootstrap

## Files

- `dash_app.py` - Main Dash application
- `rbf.py` - RBF interpolation functions
- `utils.py` - Utility functions
- `job_script.sh` - Job submission script for cluster environments
- `surf1.txt`, `surf2.txt` - Sample data files

## Requirements

- Python 3.x
- dash
- dash-bootstrap-components
- plotly
- pandas
- numpy
- scikit-learn

## Usage

1. Install required packages:
   ```bash
   pip install dash dash-bootstrap-components plotly pandas numpy scikit-learn
   ```

2. Run the application:
   ```bash
   python dash_app.py
   ```

3. Open your browser and navigate to `http://localhost:8051`

4. Select your data columns and click "Run Interpolation" to generate the visualization

## Notes

- The application uses K-means clustering to select RBF centers for irregular data
- Parameters (nx, ny, sigma) are automatically optimized using validation data
- The app is configured to run on port 8051 by default
- Designed to work in university cluster environments with port forwarding

## Reference

Refer to the RBFInterpolationApp_Helper.ipynb notebook to understand the working of the surface interpolation method.
