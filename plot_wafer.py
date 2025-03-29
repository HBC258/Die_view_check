#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_wafer.py

This script reads the FoM summary CSV (FoM_summary.csv) generated from your IDVGextraction.py
and the device status CSV (device_status.csv) exported from wafer_view.py.
It merges these files on the device ID, so that the physical (x,y) coordinates from the extraction
are combined with the user-assigned status from wafer_view.

For each FoM metric, it then produces two plots:
  (a) A wafer map showing all devices.
  (b) A wafer map showing only devices with statuses "good", "decent", or "optimal".

Custom colorbar limits are applied for each metric:
  - Gm_max @1V (µA/µm per V): 0 to 20
  - VTH (V): -2 to 2
  - Hysteresis (V): 0 to 2
  - SS_min (V/dec): 0.06 to 0.2

Usage:
    python plot_wafer.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
import numpy as np

def plot_wafer_map(df, metric, title=None, color_limits=None, grid_dims=None):
    """
    Plots a wafer map for a given FoM metric using physical (x,y) coordinates.
    If grid_dims is provided, it forces the grid extents to the given (min_x, max_x, min_y, max_y),
    so that the full wafer layout is preserved even if the data is filtered.

    Parameters:
      df : DataFrame
          Must contain columns "x", "y", and the specified metric.
      metric : str
          FoM column to visualize.
      title : str, optional
          Title for the plot.
      color_limits : tuple (vmin, vmax), optional
          Overrides automatic color scaling.
      grid_dims : tuple (min_x, max_x, min_y, max_y), optional
          If provided, these values will be used to set the grid extents.
    """
    # Compute grid dimensions: use provided grid_dims if available; otherwise compute from df.
    if grid_dims is not None:
        min_x, max_x, min_y, max_y = grid_dims
    else:
        min_x = int(df["x"].min())
        max_x = int(df["x"].max())
        min_y = int(df["y"].min())
        max_y = int(df["y"].max())
    
    width = max_x - min_x + 1
    height = max_y - min_y + 1

    # Compute offsets so that minimum coordinates become zero.
    offset_x = -min_x if min_x < 0 else 0
    offset_y = -min_y if min_y < 0 else 0

    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.axis('off')

    # Set colorbar limits.
    if color_limits is not None:
        vmin, vmax = color_limits
    else:
        valid_values = df[metric].dropna()
        if not valid_values.empty:
            vmin, vmax = valid_values.min(), valid_values.max()
        else:
            vmin, vmax = 0, 1
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.viridis

    # Draw the full wafer grid (default lightgrey).
    for i in range(height):
        for j in range(width):
            rect = patches.Rectangle((j, i), 1, 1, edgecolor="black", facecolor="lightgrey")
            ax.add_patch(rect)

    # Plot each device's FoM value using shifted coordinates.
    for _, row in df.iterrows():
        raw_x = row["x"]
        raw_y = row["y"]
        val = row[metric]
        x = int(raw_x + offset_x)
        y = int(raw_y + offset_y)
        # Skip if coordinates fall outside the grid.
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        if pd.isna(val):
            color = "grey"
            text = "NaN"
        else:
            color = cmap(norm(val))
            text = f"{val:.2f}"
        rect = patches.Rectangle((x, y), 1, 1, edgecolor="black", facecolor=color)
        ax.add_patch(rect)
        ax.text(x+0.5, y+0.5, text, ha="center", va="center", fontsize=8, color="white")

    if title:
        ax.set_title(title)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric)
    plt.show()

if __name__ == "__main__":
    # Read FoM summary from extraction.
    df_fom = pd.read_csv("FoM_summary.csv")
    # Read device status from wafer_view export.
    df_status = pd.read_csv("device_status.csv")
    # Rename wafer_view column to match FoM file; assume wafer_view exported "die_id"
    df_status.rename(columns={'die_id': 'device_id'}, inplace=True)
    # Merge on device_id.
    df_merged = pd.merge(df_fom, df_status, on="device_id", how="left")

    # Define manual colorbar limits.
    colorbar_limits = {
        "Gm_max @1V (µA/µm per V)": (0, 20),
        "VTH (V)": (-2, 2),
        "Hysteresis (V)": (0, 2),
        "SS_min (V/dec)": (0.06, 0.2)
    }

    # Compute overall grid dimensions from the full FoM summary (to keep the wafer view consistent).
    full_min_x = int(df_merged["x"].min())
    full_max_x = int(df_merged["x"].max())
    full_min_y = int(df_merged["y"].min())
    full_max_y = int(df_merged["y"].max())
    grid_dims = (full_min_x, full_max_x, full_min_y, full_max_y)

    # List of FoM metrics.
    FOM_LIST = ["SS_min (V/dec)", "VTH (V)", "Hysteresis (V)", "Gm_max @1V (µA/µm per V)"]
    valid_statuses = ["good", "decent", "optimal"]

    for metric in FOM_LIST:
        # Plot for all devices.
        plot_wafer_map(df_merged, metric, title=f"{metric} (All Devices)", 
                        color_limits=colorbar_limits.get(metric, None), grid_dims=grid_dims)
        
        # Plot for filtered devices (good/decent/optimal).
        df_filtered = df_merged[df_merged["status"].isin(valid_statuses)].copy()
        plot_wafer_map(df_filtered, metric, title=f"{metric} (Good/Decent/Optimal Only)", 
                        color_limits=colorbar_limits.get(metric, None), grid_dims=grid_dims)
