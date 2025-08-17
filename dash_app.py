# -------------------
# Author: Rahul Narkhede 
# Minimal Dash-based web-app that visualizes data from a .txt file and interpolates a surface using RBF interpolation with PyTorch

import sys
import numpy as np
import pandas as pd
import base64
import torch
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from dash_bootstrap_components import Container, Row, Col, Card
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import time
import itertools
from collections import defaultdict
# Custom scripts for RBF Interpolation
from rbf_pytorch import select_points_kmeans, interpolate_torch, rbf_interpolation_torch


# Read TXT file path from command line arguments
# The .TXT file path is provided in the Bash Command to launch the app
# if len(sys.argv) != 2:
#     print("Usage: python app.py <txt_file_path>")
#     sys.exit(1)

txt_file_path = "./surf1.txt" #sys.argv[1]

# Load the dataset
# For TXT files with comma-separated values, use read_csv with appropriate parameters
data = pd.read_csv(txt_file_path, delimiter=',', header=None, names=["x", "y", "z"])  # TXT files typically don't have headers


# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# ------------------------------------------ App Layout ---------------------------------------
app.layout = Container([
    html.H1("Data Visualization & Surface Interpolation", className="text-center my-4"),
    
    Row([
        Col([
            html.Label("Select X Column:"),
            dcc.Dropdown(
                id='x-column',
                options=[{'label': col, 'value': col} for col in data.columns],
                value=data.columns[0],  # Default to the first column
                className='mb-3'
            ),
        ], width=4),
        
        Col([
            html.Label("Select Y Column:"),
            dcc.Dropdown(
                id='y-column',
                options=[{'label': col, 'value': col} for col in data.columns],
                value=data.columns[1],  # Default to the second column
                className='mb-3'
            ),
        ], width=4),
        
        Col([
            html.Label("Select Z Column:"),
            dcc.Dropdown(
                id='z-column',
                options=[{'label': col, 'value': col} for col in data.columns],
                value=data.columns[2],  # Default to the third column
                className='mb-3'
            ),
        ], width=4),
    ], className="mb-4"),
    
    Row([
        Col([
            html.Label("Scale Data:"),
            dcc.Checklist(
                id='scale-data',
                options=[{'label': 'Yes', 'value': 'scale'}],
                value=[],
                inline=True,
                className='mb-3'
            ),
        ], width=4),
        
        Col([
            html.Button('Run Interpolation', id='submit-button', n_clicks=0, className='btn btn-primary'),
        ], width=4),
    ], className="mb-4"),
    
    dcc.Loading(
        id="loading",
        type="default",
        children=[dcc.Graph(id='interpolation-plot'), html.Div(id='projection-plots')]
    )
], fluid=True)

# ----------------------- App data-processing -------------------------------------


def encode_image(image_path):
    '''
    Helper function to encode images for display in Dash
    '''
    with open(image_path, 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode('utf-8')
    return f'data:image/png;base64,{encoded_image}'

def points_to_xvec(points_txt: str, scaler_x, scaler_y):
    '''
    Convert sampling points from text file to scaled x_vec for interpolation
    '''
    try:
        df = pd.read_csv(points_txt, delimiter=",", header=None, names=["x", "y"])
        print(f"Number of sampling points: {len(df)}")
        
        # Scale the sampling point x, y coords 
        df_scaled = pd.DataFrame(scaler_x.transform(df[["x"]]), columns=["x"])
        df_scaled["y"] = scaler_y.transform(df[["y"]])
        
        # Create input for rbf interpolation
        x_vec = df_scaled[["x", "y"]].to_numpy()

        return x_vec, df
    except FileNotFoundError:
        print(f"Warning: {points_txt} not found. Sampling points will be skipped.")
        return None, None

def interpolate_sampling_points(x_vec, centers, lambdas, sigma, df, scaler_z):
    '''
    Interpolate sampling points and descale the results
    '''
    zi = interpolate_torch(x_vec, centers, lambdas, sigma)
    zi_descaled = scaler_z.inverse_transform(zi.flatten().reshape(-1, 1)).flatten()
    df["z"] = zi_descaled
    return df, zi, zi_descaled

# Callback for the Graph
@app.callback(
    Output('interpolation-plot', 'figure'),
    Output('projection-plots', 'children'),
    Input('submit-button', 'n_clicks'),
    Input('x-column', 'value'),
    Input('y-column', 'value'),
    Input('z-column', 'value'),
    Input('scale-data', 'value'),
)
def update_graph(n_clicks, x_col, y_col, z_col, scale_data):
    '''
    Performs surface interpolation for the selected columns and data using RBF interpolation 
    with Gaussian kernel. The parameters nx, ny and sigma for the interpolation are selected
    based on validation results. The surface plot with the scatter of original data and the 
    projections along x and y axes are plotted. 

    Parameters
    ----------
    n_clicks (int): Number of 'Run Interpolation' button clicks
    x_col (str): Name of selected column for x-axis
    y_col (str): Name of selected column for y-axis
    z_col (str): Name of selected column for z-axis
    scale_data (bool): If True, data in the x and y axes is scaled between [0, 1]

    Returns
    -------
    Surface plot of interpolation with scatter of original data
    Projection plots along x and y axes
    '''

    
    # Check if the 'Run Interpolation' button has been clicked
    # if the button is not clicked at least once, then no plot is displayed
    if n_clicks is None or n_clicks == 0:
        return {}, html.Div()  # Return empty outputs if not clicked
    
    # Range of RBF interpolation parameters to try for best fit
    nx_values = [20]  # values for nx: number of centers along x axis
    ny_values = [20]  # values for ny: number of centers along y axis
    sigma_values = [0.1, 0.05, 0.01]  # values for sigma: std. deviation of a Gaussian kernel


    # Prepare data 
    # Only the columns selected by user are now indexed
    surf1 = data[[x_col, y_col, z_col]].rename(columns={x_col: "x", y_col: "y", z_col: "z"})
    
    # Scale data if necessary
    
    # Individual saclers for each axis for easy descaling
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_z = MinMaxScaler()

    if 'scale' in scale_data:
        surf1_scaled = pd.DataFrame(scaler_x.fit_transform(surf1[['x']]), columns=['x'])
        surf1_scaled['y'] = scaler_y.fit_transform(surf1[['y']])
        surf1_scaled['z'] = scaler_z.fit_transform(surf1[['z']])
    else:
        surf1_scaled = surf1.copy()  # No scaling if not selected

    
    # Split the data into training (90%) and validation (10%)
    # Validation data used to control overfitting
    train_data = surf1_scaled.sample(frac=0.9, random_state=42)  # 90% for training
    val_data = surf1_scaled.drop(train_data.index)  # Remaining 10% for validation

    # create combinations of nx, ny, sigma from lists
    combinations = np.array(list(itertools.product(nx_values, ny_values, sigma_values)))
    # filter out combinations where nx*ny > size of train_data
    # because number of centers cannot be more than the amount of data
    combinations_selected = combinations[combinations[:,0]*combinations[:, 1] <= len(train_data)]
    grouped_combos = defaultdict(list)
    for combo in combinations_selected:
        nx, ny, sigma = combo
        grouped_combos[(nx, ny)].append(sigma)
    
    # Initialize the best validation mse and parameters 
    best_mse = float('inf')
    best_params = (None, None, None)

    
    # Cross-validation loop: Try different sets of parameters (nx, ny, sigma)
    for (nx, ny), sigmas in grouped_combos.items():
        # Calculate centers once for this nx, ny combination
        centers = select_points_kmeans(train_data[['x', 'y']].to_numpy(), nx=int(nx), ny=int(ny))
        
        # Process all sigma values for this center configuration
        for sigma in sigmas:
            try:
                s, lamdas, cond, mse, residual = rbf_interpolation_torch(train_data, centers, sigma=sigma)
                
                x_vec_val = val_data[['x', 'y']].to_numpy()
                zi_val = interpolate_torch(x_vec_val, centers, lamdas, sigma=sigma)
                
                val_mse = mean_squared_error(val_data['z'], zi_val)
                
                if val_mse < best_mse:
                    best_mse = val_mse
                    best_params = (nx, ny, sigma)
                    
            except (ZeroDivisionError, np.linalg.LinAlgError):
                continue

                
    # Use the best parameters found
    final_nx, final_ny, final_sigma = best_params

    # Perform interpolation again with the best parameters
    centers = select_points_kmeans(surf1_scaled[['x', 'y']].to_numpy(), nx=int(final_nx), ny=int(final_ny))

    # Perform the final interpolation
    s, lamdas, cond, mse, residual = rbf_interpolation_torch(surf1_scaled, centers, sigma=final_sigma)

    # Create a meshgrid for interpolation
    xi = np.linspace(0, 1, 200)
    yi = np.linspace(0, 1, 200)
    xi, yi = np.meshgrid(xi, yi)
    x_vec = np.column_stack((xi.flatten(), yi.flatten()))
    
    # Interpolate
    zi = interpolate_torch(x_vec, centers, lamdas, sigma=final_sigma)
    zi = zi.reshape(200, 200)

    # Descale the interpolated values
    original_xi = scaler_x.inverse_transform(xi.flatten().reshape(-1, 1)).flatten()
    original_yi = scaler_y.inverse_transform(yi.flatten().reshape(-1, 1)).flatten()
    original_zi = scaler_z.inverse_transform(zi.flatten().reshape(-1, 1)).flatten()

    # Determine common scale for axes
    x_min = min(surf1['x'].min(), original_xi.min())
    x_max = max(surf1['x'].max(), original_xi.max())
    y_min = min(surf1['y'].min(), original_yi.min())
    y_max = max(surf1['y'].max(), original_yi.max())
    z_min = min(surf1['z'].min(), original_zi.min())
    z_max = max(surf1['z'].max(), original_zi.max())

    # Sampling points
    x_vec_sampling, sampling_df = points_to_xvec(points_txt="./sampling_points.txt", 
                                                 scaler_x=scaler_x, 
                                                 scaler_y=scaler_y)
    
    sampling_scatter = None
    if x_vec_sampling is not None and sampling_df is not None:
        sampling_df, zi_sampling, zi_descaled_sampling = interpolate_sampling_points(x_vec_sampling,
                                                                                     centers, 
                                                                                     lamdas,
                                                                                     final_sigma, 
                                                                                     sampling_df, 
                                                                                     scaler_z)
        
        # Create sampling points scatter plot
        sampling_scatter = go.Scatter3d(
            x=sampling_df["x"], 
            y=sampling_df["y"], 
            z=sampling_df["z"], 
            mode="markers", 
            marker=dict(size=4, color='black', symbol='x'), 
            name="SamplingPoints", 
            hoverinfo="text", 
            hovertemplate='Sampling Point<br>X: %{x:}<br>Y: %{y:}<br>Z: %{z:}<extra></extra>'
        )

    # Plot scatter and surface togther in one plot 
    # Create the surface plot with descaled values
    surface = go.Surface(
        x=original_xi.reshape(xi.shape),
        y=original_yi.reshape(yi.shape),
        z=original_zi.reshape(zi.shape),
        colorscale='Viridis',
        opacity=0.8,
        showscale=False,
        name='Surface',
        hoverinfo='text',
        hovertemplate='Interpolated X: %{x:}<br>Interpolated Y: %{y:}<br>Interpolated Z: %{z:}<extra></extra>',
    )
    # hovertemplate and hoverinfo controls the information displayed when cursor is moved over a certain point in the plot 

    # Create the scatter plot from the original data
    scatter = go.Scatter3d(
        x=surf1['x'],
        y=surf1['y'],
        z=surf1['z'],
        mode='markers',
        marker=dict(size=1, color=surf1['z'], colorscale='PuRd', showscale=True),
        name='Scatter',
        hoverinfo='text',
        hovertemplate='X: %{x:}<br>Y: %{y:}<br>Z: %{z:}<extra></extra>',
    )

    # Create figure with surface, scatter data, and sampling points
    plot_data = [surface, scatter]
    if sampling_scatter is not None:
        plot_data.append(sampling_scatter)
    
    fig = go.Figure(data=plot_data)

    # Define layout with common scale
    # to add MSE in title: f"<i>Validation MSE: {best_mse:.2f}</i><br>"
    fig.update_layout(
        title=(
            f"<b>3D Surface and Scatter Plot</b><br>"
            f"Parameters: (nx, ny, sigma): {int(final_nx)}, {int(final_ny)}, {final_sigma}"
        ),
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col,
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[y_min, y_max]),
            zaxis=dict(range=[z_min, z_max])
        ),
        autosize=False,
        font=dict(size=15),
        width=1000,
        height=1000,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
        paper_bgcolor="White",
    )


    # Projections along x and y axis are also displayed for ease of visualization 
    
    # Create Plotly projection plots for projection along x
    # original data
    projection_x = go.Figure()
    projection_x.add_trace(go.Scatter(
        x=surf1['x'],
        y=surf1['z'],
        mode='markers',
        marker=dict(color='red', size=2),
        name='True'
    ))
    # interpolated data
    projection_x.add_trace(go.Scatter(
        x=original_xi.flatten(),
        y=original_zi.flatten(),
        mode='markers',
        line=dict(color='blue', width=2),  # Set line color and width
        name='Best Fit',
        opacity=0.8  # Set transparency for the line here
    ))
    projection_x.update_layout(
        title='Projection along X',
        xaxis_title=x_col,
        yaxis_title=z_col,
        showlegend=True,
        height=800,  # Set height for the plot
        width=800    # Set width for the plot
    )


    # Create Plotly projection plots for projection along y
    # original data
    projection_y = go.Figure()
    projection_y.add_trace(go.Scatter(
        x=surf1['y'],
        y=surf1['z'],
        mode='markers',
        marker=dict(color='red', size=2),
        name='True'
    ))
    # interpolated data
    projection_y.add_trace(go.Scatter(
        x=original_yi.flatten(),
        y=original_zi.flatten(),
        mode='markers',
        line=dict(color='blue', width=2),  # Set line color and width
        name='Best Fit',
        opacity=0.8  # Set transparency for the line here
    ))
    projection_y.update_layout(
        title='Projection along Y',
        xaxis_title=y_col,
        yaxis_title=z_col,
        showlegend=True,
        height=800,  # Set height for the plot
        width=800    # Set width for the plot
    )


    # Create layout for projection plots
    projection_layout = html.Div([
        Row([
            Col(dcc.Graph(figure=projection_x), width=6),  # Projection along X
            Col(dcc.Graph(figure=projection_y), width=6),  # Projection along Y
        ])
    ])

    return fig, projection_layout

if __name__ == '__main__':
    # change port number if already in use (Prefer not to kill an existing port if its not yours)
    app.run(debug=True, port=8051)
