#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_wafer.py

This script reads the FoM summary CSV (e.g., FoM_summary.csv) from IDVGextraction.py
and the device status CSV (device_status.csv) exported from wafer_view.py.
It merges these files on device_id so that the physical (x,y) coordinates from the extraction
are combined with the user-assigned status from wafer_view.

For each FoM metric, it produces two plots:
  (a) A wafer map showing all devices.
  (b) A wafer map showing only devices marked as "good", "decent", or "optimal".

Unit conversion is applied as follows:
  - "SS_min (V/dec)" is converted to mV/dec (colorbar: 60–200).
  - "Hysteresis (V)" is converted to mV (colorbar: 0–2000).
  - "VTH (V)" and "Gm_max @1V (µA/µm per V)" remain unchanged.
  
The cell text shows one decimal place without the unit; the colorbar label includes the unit.
The overall wafer grid is preserved using grid_dims.
Subplot margins are adjusted with plt.subplots_adjust.

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
    ensuring the full wafer layout is preserved even if the data is filtered.

    Parameters:
      df : DataFrame
          Must contain columns "x", "y", and the specified metric.
      metric : str
          FoM column to visualize (e.g. "SS_min (V/dec)", "VTH (V)", etc.).
      title : str, optional
          Title for the plot.
      color_limits : tuple (vmin, vmax), optional
          If provided, overrides automatic color scaling (in raw units).
          (e.g. for SS_min (V/dec): (0.06, 0.2))
      grid_dims : tuple (min_x, max_x, min_y, max_y), optional
          If provided, these values will be used to set the grid extents.
    """
    # Define unit multipliers and colorbar labels for each metric.
    display_info = {
        "SS_min (V/dec)": {
            "multiplier": 1000.0,        # convert V/dec to mV/dec
            "colorbar_label": "SS_min (mV/dec)"
        },
        "Hysteresis (V)": {
            "multiplier": 1000.0,        # convert V to mV
            "colorbar_label": "Hysteresis (mV)"
        },
        "VTH (V)": {
            "multiplier": 1.0,
            "colorbar_label": "VTH (V)"
        },
        "Gm_max @1V (µA/µm per V)": {
            "multiplier": 1.0,
            "colorbar_label": "Gm_max @1V (µA/µm per V)"
        }
    }
    
    if metric in display_info:
        mult = display_info[metric]["multiplier"]
        cbar_label = display_info[metric]["colorbar_label"]
    else:
        mult = 1.0
        cbar_label = metric

    # Determine grid dimensions.
    if grid_dims is not None:
        min_x, max_x, min_y, max_y = grid_dims
    else:
        min_x = int(df["x"].min())
        max_x = int(df["x"].max())
        min_y = int(df["y"].min())
        max_y = int(df["y"].max())
    
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    offset_x = -min_x if min_x < 0 else 0
    offset_y = -min_y if min_y < 0 else 0

    # Create figure with adjusted subplot margins and DPI if desired.
    fig, ax = plt.subplots(figsize=(width, height))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.axis('off')

    # Set colorbar limits: if provided, use them multiplied by the multiplier.
    if color_limits is not None:
        raw_vmin, raw_vmax = color_limits
        vmin, vmax = raw_vmin * mult, raw_vmax * mult
    else:
        valid_vals = df[metric].dropna()
        if not valid_vals.empty:
            vmin, vmax = valid_vals.min() * mult, valid_vals.max() * mult
        else:
            vmin, vmax = 0, 1

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.viridis

    # Draw the full wafer grid (default lightgrey).
    for i in range(height):
        for j in range(width):
            rect = patches.Rectangle((j, i), 1, 1, edgecolor="black", facecolor="lightgrey")
            ax.add_patch(rect)

    # Plot each device using shifted coordinates.
    for _, row in df.iterrows():
        raw_x = row["x"]
        raw_y = row["y"]
        val = row[metric]
        x = int(raw_x + offset_x)
        y = int(raw_y + offset_y)
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        if pd.isna(val):
            color = "gray"
            text = "NaN"
        else:
            conv_val = val * mult
            color = cmap(norm(conv_val))
            text = f"{conv_val:.1f}"  # one decimal place
        # Note: (x, y) -> lower-left corner is (x, y) so we flip y for display.
        rect = patches.Rectangle((x, height - y - 1), 1, 1, edgecolor="black", facecolor=color)
        ax.add_patch(rect)
        ax.text(x+0.5, height - y - 0.5, text, ha="center", va="center", fontsize=16, color="white")

    if title:
        ax.set_title(title, fontsize=16)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(cbar_label, fontsize=16)
    plt.show()


if __name__ == "__main__":
    # Load FoM summary DataFrame from extraction.
    df_fom = pd.read_csv("FoM_summary.csv")  # Expected: device_id, x, y, SS_min (V/dec), VTH (V), Hysteresis (V), Gm_max @1V (µA/µm per V)
    
    # Load wafer view status CSV exported from wafer_view.py.
    df_status = pd.read_csv("device_status.csv")  # Expected: die_id, status
    # Rename "die_id" to "device_id" for merging.
    if "die_id" in df_status.columns:
        df_status.rename(columns={"die_id": "device_id"}, inplace=True)
    
    # Merge FoM summary with status on device_id.
    df_merged = pd.merge(df_fom, df_status, on="device_id", how="left")

    # Define manual colorbar limits (in raw units) for each metric.
    # For SS_min, raw values are in V/dec; we want colorbar to show 60 to 200 mV/dec.
    # For Hysteresis, raw values are in V; we want colorbar to show 0 to 2000 mV.
    colorbar_limits = {
        "Gm_max @1V (µA/µm per V)": (0, 20),
        "VTH (V)": (0, 2),
        "Hysteresis (V)": (0, 0.2),
        "SS_min (V/dec)": (0.06, 0.2)
    }

    # Compute overall grid dimensions from full FoM summary to preserve the wafer layout.
    full_min_x = int(df_merged["x"].min())
    full_max_x = int(df_merged["x"].max())
    full_min_y = int(df_merged["y"].min())
    full_max_y = int(df_merged["y"].max())
    grid_dims = (full_min_x, full_max_x, full_min_y, full_max_y)

    # List of FoM metrics.
    FOM_LIST = ["SS_min (V/dec)", "VTH (V)", "Hysteresis (V)", "Gm_max @1V (µA/µm per V)"]
    valid_statuses = ["good", "decent", "optimal"]

    for metric in FOM_LIST:
        # Plot wafer map for all devices.
        plot_wafer_map(df_merged, metric,
                       color_limits=colorbar_limits.get(metric, None), grid_dims=grid_dims,
                       )
        # Plot wafer map only for good, decent, optimal devices.
        df_filtered = df_merged[df_merged["status"].isin(valid_statuses)].copy()
        plot_wafer_map(df_filtered, metric,
                       color_limits=colorbar_limits.get(metric, None), grid_dims=grid_dims,
                       )
