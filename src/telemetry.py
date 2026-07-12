import os
import matplotlib.pyplot as plt
import pandas as pd

class LivePlotterMixin:
    """
    A mixin class providing real-time plotting and data logging capabilities 
    for SOFA controllers.
    
    Expected attributes in the child class:
    - self.target
    - self.history_time, self.history_x, self.history_y, self.history_z
    - self.history_q_z_inner, self.history_q_z_outer
    - self.history_q_rot_inner, self.history_q_rot_outer
    - self.history_x_outer, self.history_y_outer, self.history_z_outer
    - self.history_force
    - self.fig
    - self.frame_counter
    - self.plot_update_frequency
    - self.plot_title (optional)
    - self.log_prefix (optional)
    """

    # =========================================================================
    # REAL-TIME PLOTTING METHODS
    # =========================================================================

    def setup_live_plot(self):
        """Initializes the matplotlib interactive mode and layout."""
        plt.ion()  # Turn on interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        title = getattr(self, 'plot_title', "Controller LIVE Dashboard")
        self.fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Initialize empty lines for Subplot 1 (X, Y, Z Positions)
        self.line_x, = self.ax1.plot([], [], color='red', linewidth=2.0, label="Tip X")
        self.line_y, = self.ax1.plot([], [], color='green', linewidth=2.0, label="Tip Y")
        self.line_z, = self.ax1.plot([], [], color='blue', linewidth=2.0, label="Tip Z")
        
        # Target lines
        self.ax1.axhline(y=self.target[0], color='red', linestyle='--', alpha=0.5, label="Target X")
        self.ax1.axhline(y=self.target[1], color='green', linestyle='--', alpha=0.5, label="Target Y")
        self.ax1.axhline(y=self.target[2], color='blue', linestyle='--', alpha=0.5, label="Target Z")
        
        self.ax1.set_ylabel("Position (mm)", fontsize=10)
        self.ax1.grid(True, linestyle=':', alpha=0.6)
        # Place legend neatly inside
        self.ax1.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
        
        # Initialize empty lines for Subplot 2
        self.line_uz_in, = self.ax2.plot([], [], color='teal', linewidth=1.5, label="Z Inner (mm)")
        self.line_uz_out, = self.ax2.plot([], [], color='cyan', linewidth=1.5, label="Z Outer (mm)")
        self.line_rot_in, = self.ax2.plot([], [], color='darkorange', linewidth=1.5, label="Rot Inner (deg)")
        self.line_rot_out, = self.ax2.plot([], [], color='purple', linewidth=1.5, label="Rot Outer (deg)")
        self.ax2.set_ylabel("Absolute Joint Positions", fontsize=10)
        self.ax2.set_xlabel("Simulation Timeline (seconds)", fontsize=11)
        self.ax2.grid(True, linestyle=':', alpha=0.6)
        self.ax2.legend(loc="lower right", ncol=2)

        plt.tight_layout()
        self.fig.show()
        self.fig.canvas.flush_events()


    def update_live_plot(self):
        """Injects new data into the existing plot lines without blocking SOFA."""
        if getattr(self, 'fig', None) is None or not plt.fignum_exists(self.fig.number):
            return

        # Update data dynamically
        self.line_x.set_data(self.history_time, self.history_x)
        self.line_y.set_data(self.history_time, self.history_y)
        self.line_z.set_data(self.history_time, self.history_z)
        
        self.line_uz_in.set_data(self.history_time, self.history_q_z_inner)
        self.line_uz_out.set_data(self.history_time, self.history_q_z_outer)
        self.line_rot_in.set_data(self.history_time, self.history_q_rot_inner)
        self.line_rot_out.set_data(self.history_time, self.history_q_rot_outer)

        # Rescale axes continuously to fit the new data
        for ax in [self.ax1, self.ax2]:
            ax.relim()
            ax.autoscale_view()

        # Flush the GUI event queue to render the frame
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Periodically save the CSV log while the simulation is running
        if self.frame_counter % (self.plot_update_frequency * 2) == 0:
            self.save_csv_log()


    def save_csv_log(self):
        """Saves the tracking data to a CSV for post-processing."""
        images_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        prefix = getattr(self, 'log_prefix', 'Controller')
        
        df = pd.DataFrame({
            'Time': self.history_time,
            'X_inner': self.history_x,
            'Y_inner': self.history_y,
            'Z_inner': self.history_z,
            'X_outer': self.history_x_outer,
            'Y_outer': self.history_y_outer,
            'Z_outer': self.history_z_outer,
            'X_target': [self.target[0]] * len(self.history_time),
            'Y_target': [self.target[1]] * len(self.history_time),
            'Z_target': [self.target[2]] * len(self.history_time),
            'Force': self.history_force
        })
        
        csv_path = os.path.join(images_dir, f'{prefix}_Log.csv')
        df.to_csv(csv_path, index=False)


    def finalize_plot(self):
        """Saves the final plot instead of blocking the SOFA thread."""
        if getattr(self, 'fig', None) is not None and plt.fignum_exists(self.fig.number):
            prefix = getattr(self, 'log_prefix', 'Controller')
            try:
                images_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images')
                os.makedirs(images_dir, exist_ok=True)
                save_path = os.path.join(images_dir, f'{prefix}_Final_Plot.png')
                self.fig.savefig(save_path)
                print(f"\033[92m[{prefix}] Final plot saved in {save_path}\033[0m")
                
                # Also save the final CSV log
                self.save_csv_log()
                print(f"\033[92m[{prefix}] Data logs saved to CSV successfully.\033[0m")
            except Exception as e:
                print(f"\033[91m[{prefix}] Failed to save plot or log: {e}\033[0m")
