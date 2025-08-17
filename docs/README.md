# RBF Interpolation with PyTorch

A PyTorch-based implementation for Radial Basis Function (RBF) interpolation with Gaussian kernels, featuring both standalone scripts and a minimal Dash web interface.

## Description

This project provides GPU-accelerated RBF interpolation for 3D surface reconstruction:
- Standalone surface interpolation with automatic parameter optimization
- Interactive web interface for real-time visualization
- Sampling point interpolation from custom coordinate files
- 3D surface plots with projection views

## Interactive Demo

[View Interactive 3D Surface Plot](surface_plot.html)

*Click the link above to see a live example of the 3D surface interpolation with original data points and sampling locations.*


## Key Features

- **PyTorch Backend**: GPU acceleration for faster computation
- **Parameter Optimization**: Automatic cross-validation for nx, ny, sigma parameters
- **Sampling Points**: Interpolate custom sampling coordinates and display as markers
- **3D Visualization**: Interactive surface plots with original data and sampling points
- **Minimal Interface**: Simple web UI

## Project Structure

```
├── interpolate_surface_torch.py    # Main interpolation script
├── dash_app.py                     # Minimal web interface  
├── rbf_pytorch.py                  # PyTorch RBF functions
├── data/
│   ├── surf1.txt                   # Sample surface data
│   └── sampling_points.txt         # Custom sampling coordinates
├── results/                        # Generated plots and outputs
└── rbf_interpolation_env.yml       # Conda environment
```

## Quick Start

1. **Setup Environment**:
   ```bash
   conda env create -f rbf_interpolation_env.yml
   conda activate rbf_interpolation_env
   ```

2. **Run Interpolation**:
   ```bash
   python interpolate_surface_torch.py
   ```

3. **Launch Web Interface**:
   ```bash
   python dash_app.py
   ```
   Navigate to `http://localhost:8051`

## Requirements

- Python 3.8+
- PyTorch
- Dash, Plotly
- NumPy, Pandas
- Scikit-learn

## Data Format

Input files should be comma-separated with columns: `x,y,z`
Sampling points file: `x,y` coordinates for interpolation
