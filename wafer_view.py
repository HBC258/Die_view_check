#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
wafer_view.py

Interactive wafer view for marking each die as good/bad/decent/optimal,
then exporting a CSV that maps 'die_id' to 'status'.

Usage:
1) Run this script.
2) In the displayed window, click each die to cycle through statuses.
3) Close the window (or press Ctrl+C in the console) when finished.
4) Uncomment or call wafer.export_map("device_status.csv") to save the results.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv

class WaferPlotter:
    def __init__(self, rows, cols, fontsize_id=8, fontsize_coord=8, fontsize_status=8):
        plt.ion()  # Enable interactive mode
        self.rows = rows
        self.cols = cols
        self.fontsize_id = fontsize_id
        self.fontsize_coord = fontsize_coord
        self.fontsize_status = fontsize_status

        # Define the statuses in order
        self.statuses = ["good", "bad", "decent", "optimal"]

        # Initialize every die to "good"
        self.die_status = [[self.statuses[0] for _ in range(cols)] for _ in range(rows)]

        # Define colors for each status
        self.colors = {
            "good": "green",
            "bad": "red",
            "decent": "yellow",
            "optimal": "blue"
        }

        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(cols, rows))
        self.ax.set_xlim(0, cols)
        self.ax.set_ylim(0, rows)
        self.ax.set_aspect('equal')
        plt.axis('off')

        # Dictionaries to hold our plot elements for each die
        self.rect_patches = {}
        self.id_texts = {}
        self.coord_texts = {}
        self.status_texts = {}

        # Build the initial plot
        self._init_plot()

        # Connect the click event so clicking a die updates its status
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def _compute_die_id(self, i, j):
        """
        Compute the die ID for row i, column j in a "snake" pattern:
        - Even rows (0, 2, ...) go left-to-right.
        - Odd rows (1, 3, ...) go right-to-left.

        Example for 9x11 wafer:
          Row 0: IDs 1..11 (left to right)
          Row 1: IDs 12..22 (right to left)
          Row 2: IDs 23..33 (left to right)
          etc.
        """
        if i % 2 == 0:
            # Even row: left to right
            return i * self.cols + (j + 1)
        else:
            # Odd row: right to left
            return i * self.cols + (self.cols - j)

    def _init_plot(self):
        """Draw the grid with die IDs, row-col coordinates, and status."""
        for i in range(self.rows):
            for j in range(self.cols):
                status = self.die_status[i][j]
                color = self.colors.get(status, "blue")
                # Position: x = column, y = rows - i - 1 (so row 0 is at top visually)
                x = j
                y = self.rows - i - 1

                # Draw the rectangle for the die
                rect = patches.Rectangle((x, y), 1, 1, edgecolor="black", facecolor=color)
                self.ax.add_patch(rect)
                self.rect_patches[(i, j)] = rect

                # Calculate snake-pattern die ID
                die_id = self._compute_die_id(i, j)
                # Top text: Die ID
                id_text = self.ax.text(x + 0.5, y + 0.75, f"ID={die_id}",
                                       ha="center", va="center", fontsize=self.fontsize_id)
                self.id_texts[(i, j)] = id_text

                # Middle text: (row, col)
                coord_text = self.ax.text(x + 0.5, y + 0.55, f"({i},{j})",
                                          ha="center", va="center", fontsize=self.fontsize_coord)
                self.coord_texts[(i, j)] = coord_text

                # Bottom text: Current status
                status_text = self.ax.text(x + 0.5, y + 0.35, status,
                                           ha="center", va="center", fontsize=self.fontsize_status)
                self.status_texts[(i, j)] = status_text

        self.ax.set_title("Interactive Wafer View")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_die(self, row, col, status):
        """Update the die's status, change its color and text accordingly."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.die_status[row][col] = status
            rect = self.rect_patches[(row, col)]
            rect.set_facecolor(self.colors.get(status, "blue"))
            status_text = self.status_texts[(row, col)]
            status_text.set_text(status)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        else:
            print("Die location out of range!")

    def on_click(self, event):
        """Handle mouse click events to update a die's status."""
        if event.inaxes != self.ax:
            return  # Click was outside wafer axes

        if event.xdata is None or event.ydata is None:
            return

        col = int(event.xdata)
        row = self.rows - int(event.ydata) - 1

        if 0 <= row < self.rows and 0 <= col < self.cols:
            current_status = self.die_status[row][col]
            try:
                index = self.statuses.index(current_status)
            except ValueError:
                index = 0
            next_index = (index + 1) % len(self.statuses)
            next_status = self.statuses[next_index]
            self.update_die(row, col, next_status)
        else:
            print("Clicked outside of die boundaries.")

    def export_map(self, filename="device_status.csv"):
        """
        Export the snake-pattern die ID and its status to a CSV.
        Columns: die_id, status
        """
        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["die_id", "status"])
            for i in range(self.rows):
                for j in range(self.cols):
                    die_id = self._compute_die_id(i, j)
                    status = self.die_status[i][j]
                    writer.writerow([die_id, status])
        print(f"Status map exported to {filename}")

    def show(self, title="Wafer View"):
        """Block execution until the figure window is closed."""
        self.ax.set_title(title)
        plt.show()

# --- Example Usage ---
if __name__ == "__main__":
    # Create a wafer with 9 rows and 11 columns
    wafer = WaferPlotter(9, 11, fontsize_id=10, fontsize_coord=9, fontsize_status=10)
    wafer.show("Interactive Wafer View")

    # Once you close the window, you can call:
    # wafer.export_map("device_status.csv")
    # to save the status map to a CSV.
