import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import medfilt

import sys
import glob

# Configure matplotlib parameters
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 2,
    "figure.figsize": (10, 8),
    "figure.dpi": 300,
})

def main():
    # Load the CSV
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Auto-detect the newest log file
        log_files = glob.glob(os.path.join(base_dir, 'images', '*_Log.csv'))
        if not log_files:
            print("Error: No log files found in the images/ directory.")
            print("Please run the simulation and let the controller finish to generate the log.")
            return
        csv_path = max(log_files, key=os.path.getmtime)
        print(f"Auto-detected newest log file: {os.path.basename(csv_path)}")
    
    if not os.path.exists(csv_path):
        print(f"Error: Log file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Get controller name for output prefix
    prefix = os.path.basename(csv_path).split('_')[0].lower()
    
    time = df['Time']
    
    # -------------------------------------------------------------------------
    # 1. Desired vs Actual Position (X, Y, Z)
    # -------------------------------------------------------------------------
    fig1, axs1 = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    
    # X
    axs1[0].plot(time, df['X_inner'], label='Actual $X$', color='tab:red')
    axs1[0].plot(time, df['X_target'], '--', label='Target $X$', color='tab:red', alpha=0.6)
    axs1[0].set_ylabel('$X$ Position [mm]')
    axs1[0].grid(True, linestyle=':', alpha=0.7)
    axs1[0].legend(loc='best')
    
    # Y
    axs1[1].plot(time, df['Y_inner'], label='Actual $Y$', color='tab:green')
    axs1[1].plot(time, df['Y_target'], '--', label='Target $Y$', color='tab:green', alpha=0.6)
    axs1[1].set_ylabel('$Y$ Position [mm]')
    axs1[1].grid(True, linestyle=':', alpha=0.7)
    axs1[1].legend(loc='best')
    
    # Z
    axs1[2].plot(time, df['Z_inner'], label='Actual $Z$', color='tab:blue')
    axs1[2].plot(time, df['Z_target'], '--', label='Target $Z$', color='tab:blue', alpha=0.6)
    axs1[2].set_ylabel('$Z$ Position [mm]')
    axs1[2].set_xlabel('Time [s]')
    axs1[2].grid(True, linestyle=':', alpha=0.7)
    axs1[2].legend(loc='best')
    
    fig1.tight_layout()
    fig1.savefig(os.path.join(base_dir, 'images', f'{prefix}_positions.png'))
    
    # -------------------------------------------------------------------------
    # 2. Error for X, Y, Z
    # -------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    err_x = df['X_target'] - df['X_inner']
    err_y = df['Y_target'] - df['Y_inner']
    err_z = df['Z_target'] - df['Z_inner']
    err_norm = np.sqrt(err_x**2 + err_y**2 + err_z**2)
    
    ax2.plot(time, err_x, label='$e_x$', color='tab:red', linestyle='-.')
    ax2.plot(time, err_y, label='$e_y$', color='tab:green', linestyle='-.')
    ax2.plot(time, err_z, label='$e_z$', color='tab:blue', linestyle='-.')
    ax2.plot(time, err_norm, label='||$e$||', color='black', linewidth=2.5)
    
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Tracking Error [mm]')
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend(loc='upper right')
    
    fig2.tight_layout()
    fig2.savefig(os.path.join(base_dir, 'images', f'{prefix}_errors.png'))
    
    # -------------------------------------------------------------------------
    # 3. Force Norm vs Limit Force at 0.4 N
    # -------------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    
    # Use a median filter to completely eliminate unphysical collision spikes
    # Filter size must be odd. 15 provides strong despiking.
    filtered_force = medfilt(df['Force'].values, kernel_size=15)
    
    ax3.plot(time, filtered_force, label='Contact Force', color='tab:orange', linewidth=2)
    ax3.axhline(y=0.4, color='tab:red', linestyle='--', linewidth=2, label='Limit Force (0.4 N)')
    
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Force [N]')
    ax3.grid(True, linestyle=':', alpha=0.7)
    ax3.legend(loc='upper right')
    
    # Cap the Y-axis to 1.0 N to keep the meaningful physics visible
    max_visible_force = min(1.0, np.max(filtered_force) * 1.2)
    ax3.set_ylim([0, max(0.5, max_visible_force)])
    
    fig3.tight_layout()
    fig3.savefig(os.path.join(base_dir, 'images', f'{prefix}_force.png'))
    
    # -------------------------------------------------------------------------
    # 4. 3D Trajectory of the Instrument Tip
    # -------------------------------------------------------------------------
    fig4 = plt.figure(figsize=(10, 8))
    ax4 = fig4.add_subplot(111, projection='3d')
    
    # Plot the true physical trajectories directly from the CSV
    ax4.plot(df['X_inner'], df['Y_inner'], df['Z_inner'], label='Instrument Tip', color='tab:blue', linewidth=2)

    # Plot Start and Target
    ax4.scatter(df['X_inner'].iloc[0], df['Y_inner'].iloc[0], df['Z_inner'].iloc[0], 
                color='black', s=100, marker='o', label='Start')
    ax4.scatter(df['X_target'].iloc[0], df['Y_target'].iloc[0], df['Z_target'].iloc[0], 
                color='tab:red', s=150, marker='*', label='Target')
    
    ax4.set_xlabel('$X$ [mm]', labelpad=10)
    ax4.set_ylabel('$Y$ [mm]', labelpad=10)
    ax4.set_zlabel('$Z$ [mm]', labelpad=10)
    ax4.legend(loc='best')
    
    # Make axes look nice
    ax4.xaxis.pane.fill = False
    ax4.yaxis.pane.fill = False
    ax4.zaxis.pane.fill = False
    ax4.grid(True, linestyle=':', alpha=0.6)
    
    # Set equal aspect ratio
    max_range = np.array([df['X_inner'].max()-df['X_inner'].min(), 
                          df['Y_inner'].max()-df['Y_inner'].min(), 
                          df['Z_inner'].max()-df['Z_inner'].min()]).max() / 2.0
    mid_x = (df['X_inner'].max()+df['X_inner'].min()) * 0.5
    mid_y = (df['Y_inner'].max()+df['Y_inner'].min()) * 0.5
    mid_z = (df['Z_inner'].max()+df['Z_inner'].min()) * 0.5
    ax4.set_xlim(mid_x - max_range, mid_x + max_range)
    ax4.set_ylim(mid_y - max_range, mid_y + max_range)
    ax4.set_zlim(mid_z - max_range, mid_z + max_range)
    
    fig4.tight_layout()
    fig4.savefig(os.path.join(base_dir, 'images', f'{prefix}_3d_trajectory.png'))
    
    print(f"All paper plots for {prefix.upper()} have been successfully generated and saved to the 'images/' folder.")
    
    plt.show()

if __name__ == '__main__':
    main()
