import sys
import os
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
from rbf_pytorch import *



def points_to_xvec(points_txt:str, scaler_x, scaler_y):
    df = pd.read_csv(points_txt, delimiter=",", header=None, names=["x", "y"])
    print(f"Number of sampling points:{len(df)}")
    #scale the sampling point x, y coords 
    df_scaled = pd.DataFrame(scaler_x.fit_transform(df[["x"]]), columns=["x"])
    df_scaled["y"] = scaler_y.fit_transform(df[["y"]])
    
    # create meshgrid and input for rbf interpolation
    # xi, yi = np.meshgrid(df_scaled["x"].to_numpy(), df_scaled["y"].to_numpy())
    # x_vec = np.column_stack((xi.flatten(), yi.flatten()))
    x_vec = df_scaled[["x", "y"]].to_numpy()

    return x_vec, df

def interpolate_sampling_points(x_vec, centers, lambdas, sigma, df, scaler_z):
    zi = interpolate_torch(x_vec, centers, lambdas, sigma)
    print(f'zi shape:{zi.shape}')
    zi_descaled = scaler_z.inverse_transform(zi.flatten().reshape(-1, 1)).flatten()
    print(f'zi_descaled shape:{zi_descaled.shape}')
    df["z"] = zi_descaled
    df.to_csv("./interpolated_sampling_points.txt")
    return df, zi, zi_descaled


def main():
    txt_file_path = "./surf1.txt" #sys.argv[1]

    # Load the dataset
    # For TXT files with comma-separated values, use read_csv with appropriate parameters
    data = pd.read_csv(txt_file_path, delimiter=',', header=None, names=["x", "y", "z"])

    # Range of RBF interpolation parameters to try for best fit
    nx_values = [20]  # values for nx: number of centers along x axis
    ny_values = [20]  # values for ny: number of centers along y axis
    sigma_values = [0.1, 0.05, 0.01]  # values for sigma: std. deviation of a Gaussian kernel

    # Define variables
    x_col = "x"
    y_col = "y" 
    z_col = "z"
    scale_data = ["scale"]  # or [] if you don't want scaling


    # Prepare data 
    # Only the columns selected by user are now indexed
    surf1 = data[[x_col, y_col, z_col]].rename(columns={x_col: "x", y_col: "y", z_col: "z"})
    print(f"Number of points: {surf1.shape}")
    
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

    
    # Split the data into training (80%) and validation (20%)
    # Validation data used to control overfitting
    train_data = surf1_scaled.sample(frac=0.9, random_state=42)  # 80% for training
    val_data = surf1_scaled.drop(train_data.index)  # Remaining 20% for validation

    # # create combinations of nx, ny, sigma from lists
    combinations = np.array(list(itertools.product(nx_values, ny_values, sigma_values)))
    # # filter out combinations where nx*ny > size of train_data
    # # because number of centers cannot be more than the amount of data
    combinations_selected = combinations[combinations[:,0]*combinations[:, 1] <= len(train_data)]
    grouped_combos = defaultdict(list)
    for combo in combinations_selected:
        nx, ny, sigma = combo
        grouped_combos[(nx, ny)].append(sigma)
    
    # Initialize the best validation mse and parameters 
    best_mse = float('inf')
    best_params = (None, None, None)
    
    centers_func = select_points_kmeans

    
    # Cross-validation loop: Try different sets of parameters (nx, ny, sigma)
    print("nx ny sigma valmse cond")
    for (nx, ny), sigmas in grouped_combos.items():
        # Calculate centers once for this nx, ny combination
        centers = centers_func(train_data[['x', 'y']].to_numpy(), nx=int(nx), ny=int(ny))
        
        # Process all sigma values for this center configuration
        for sigma in sigmas:
            try:
                s, lamdas, cond, mse, residual = rbf_interpolation_torch(train_data, centers, sigma=sigma)
                
                x_vec_val = val_data[['x', 'y']].to_numpy()
                zi_val = interpolate_torch(x_vec_val, centers, lamdas, sigma=sigma)
                
                val_mse = mean_squared_error(val_data['z'], zi_val)
                print(f"{nx:3.0f} {ny:3.0f} {sigma:6.3f} {val_mse:10.4f} {cond:10.4f}")
                
                if val_mse < best_mse:
                    best_mse = val_mse
                    best_params = (nx, ny, sigma)
                    
            except (ZeroDivisionError, np.linalg.LinAlgError):
                continue

                
    # Use the best parameters found
    final_nx, final_ny, final_sigma = best_params
    print(f"Best params:")
    print(f"nx: {final_nx}, ny: {final_ny}, sigma:{final_sigma} with MSE: {best_mse}")

    # Perform interpolation again with the best parameters
    centers = centers_func(surf1_scaled[['x', 'y']].to_numpy(), nx=int(final_nx), ny=int(final_ny))

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

    # sampling points
    x_vec_sampling, sampling_df = points_to_xvec(points_txt = "./sampling_points.txt", 
                           scaler_x = scaler_x, 
                           scaler_y = scaler_y, 
                           )

    sampling_df, zi_sampling, zi_descaled_sampling = interpolate_sampling_points(x_vec_sampling,
                                                               centers, 
                                                               lamdas,
                                                               final_sigma, 
                                                               sampling_df, 
                                                               scaler_z)
    


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

    scatter_sampling = go.Scatter3d(
        x=sampling_df["x"], 
        y=sampling_df["y"], 
        z=sampling_df["z"], 
        mode="markers", 
        marker=dict(size=4, color='black', symbol='x'), 
        name="SamplingPoints", 
        hoverinfo="text", 
        hovertemplate='Sampling Point<br>X: %{x:}<br>Y: %{y:}<br>Z: %{z:}<extra></extra>'
    )

    # Create figure with both surface and scatter data
    fig = go.Figure(data=[surface, scatter_sampling])

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

    # Save the 3D surface plot as HTML
    # Create output directory if it doesn't exist
    ws_path = "./"#"/data/horse/ws/rana575h-rbf_interpolation"
    os.makedirs(ws_path, exist_ok=True)
    
    # Save the surface plot
    surface_plot_path = os.path.join(ws_path, "surface_plot.html")
    fig.write_html(surface_plot_path)
    
    # # Save projection plots as HTML files
    projection_x.write_html("projection_x.html")
    projection_y.write_html("projection_y.html")
    
    print("Files saved:")
    print("- surface_plot.html (3D surface plot)")
    print("- projection_x.html (X-axis projection)")
    print("- projection_y.html (Y-axis projection)")


    # return fig, projection_layout
if __name__ == '__main__':
    
    main()
