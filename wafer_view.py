import matplotlib.pyplot as plt
import matplotlib.patches as patches

class WaferPlotter:
    def __init__(self, rows, cols, fontsize_id=8, fontsize_coord=8, fontsize_status=8):
        plt.ion()  # Enable interactive mode for live updates

        self.rows = rows
        self.cols = cols

        # Save font sizes as instance variables
        self.fontsize_id = fontsize_id
        self.fontsize_coord = fontsize_coord
        self.fontsize_status = fontsize_status

        # Initialize die status: default is 'good'
        self.die_status = [['good' for _ in range(cols)] for _ in range(rows)]

        # Define colors for statuses. Now includes "optimal" and "decent"
        self.colors = {
            'good': 'green',
            'bad': 'red',
            'optimal': 'blue',   # Choose a color you prefer
            'decent': 'yellow'
        }

        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(cols, rows))
        self.ax.set_xlim(0, cols)
        self.ax.set_ylim(0, rows)
        self.ax.set_aspect('equal')
        plt.axis('off')

        # Dictionaries to hold patches and text annotations for each die
        self.rect_patches = {}
        self.id_texts = {}
        self.coord_texts = {}
        self.status_texts = {}

        # Initialize the wafer view
        self._init_plot()

    def _compute_die_id(self, i, j):
        """
        Compute the die ID for row i, column j in a 'snake' pattern:
        - Even rows (0, 2, ...) go left-to-right.
        - Odd rows (1, 3, ...) go right-to-left.
        """
        if i % 2 == 0:
            return i * self.cols + (j + 1)
        else:
            return i * self.cols + (self.cols - j)

    def _init_plot(self):
        """Draw the initial grid of dies with ID, row-col, and default status."""
        for i in range(self.rows):
            for j in range(self.cols):
                status = self.die_status[i][j]
                color = self.colors.get(status, 'blue')

                # Calculate position so that row 0 is at the top visually
                x = j
                y = self.rows - i - 1

                # Create a rectangle patch for the die
                rect = patches.Rectangle((x, y), 1, 1, edgecolor='black', facecolor=color)
                self.ax.add_patch(rect)
                self.rect_patches[(i, j)] = rect

                # Calculate the die ID using snake pattern
                die_id = self._compute_die_id(i, j)

                # Top: Die ID
                id_text = self.ax.text(x + 0.5, y + 0.75, f"ID={die_id}",
                                       ha='center', va='center', fontsize=self.fontsize_id)
                self.id_texts[(i, j)] = id_text

                # Middle: Row and Column (coordinate)
                coord_text = self.ax.text(x + 0.5, y + 0.55, f"({i},{j})",
                                          ha='center', va='center', fontsize=self.fontsize_coord)
                self.coord_texts[(i, j)] = coord_text

                # Bottom: Status (good, bad, optimal, decent)
                status_text = self.ax.text(x + 0.5, y + 0.35, status,
                                           ha='center', va='center', fontsize=self.fontsize_status)
                self.status_texts[(i, j)] = status_text

        self.ax.set_title("Wafer View")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_die(self, row, col, status):
        """Update the status of a die and refresh only that part of the plot."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            # Update internal status array
            self.die_status[row][col] = status

            # Update the rectangle's color
            rect = self.rect_patches[(row, col)]
            rect.set_facecolor(self.colors.get(status, 'blue'))

            # Update the status text
            status_text = self.status_texts[(row, col)]
            status_text.set_text(status)

            # Redraw updated area
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        else:
            print("Die location out of range!")

    def show(self, title="Wafer View"):
        """Optionally block execution until the figure window is closed."""
        self.ax.set_title(title)
        plt.show()

# --- Example Usage ---
if __name__ == "__main__":
    # Create a wafer object with 9 rows and 11 columns,
    # and adjust font sizes as desired.
    wafer = WaferPlotter(9, 11, fontsize_id=20, fontsize_coord=18, fontsize_status=20)

    # In an interactive session (e.g., Spyder's console), you can update a die like so:
    # wafer.update_die(2, 7, 'bad')
    # wafer.update_die(4, 3, 'optimal')
    # wafer.update_die(1, 5, 'decent')
    #
    # If you want a blocking window, call:
    # wafer.show("Final Wafer View")
