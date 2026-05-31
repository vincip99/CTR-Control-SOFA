import numpy as np
import Sofa.Core
import Sofa.Helper
import matplotlib.pyplot as plt

class MRACPositionController(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.name = "MRACController"
        self.ir_controller = kwargs.get('irController')
        self.rootNode = kwargs.get('rootNode')
        
        # --- Target Input Vector r = [X, Y, Z]^T (3x1) ---
        self.r = np.array(kwargs.get('target', [4.49, 4.0, 46.5]))
        self.stop_at_z = kwargs.get('stop_at_z', 15.0)
        
        # --- Reference Model Parameters ---
        raw_Am = kwargs.get('Am', 0.4)
        raw_Bm = kwargs.get('Bm', 0.4)
        
        if isinstance(raw_Am, (int, float)):
            self.Am = np.array([float(raw_Am), float(raw_Am), float(raw_Am)])
        else:
            self.Am = np.array(raw_Am)
            
        if isinstance(raw_Bm, (int, float)):
            self.Bm = np.array([float(raw_Bm), float(raw_Bm), float(raw_Bm)])
        else:
            self.Bm = np.array(raw_Bm)
        
        # --- PI Adaptive Learning Rates ---
        self.gamma_trans_I = kwargs.get('gamma_trans_I', 0.1)  # Integral trans rate
        self.gamma_rot_I = kwargs.get('gamma_rot_I', 0.05)     # Integral rot rate
        self.gamma_trans_P = kwargs.get('gamma_trans_P', 0.05) # Proportional trans rate
        self.gamma_rot_P = kwargs.get('gamma_rot_P', 0.02)     # Proportional rot rate
        self.sigma = kwargs.get('sigma', 0.02) # Leakage factor to prevent windup
        
        # Initial adaptive tracking gains
        self.Ky_I = np.array([0.1, 0.1, 0.2]) # Integral part
        self.Ky = np.copy(self.Ky_I)          # Total gain (PI)
        
        self.active = False
        
        # --- Performance Plot Data Buffers ---
        self.history_time = []
        self.history_err_norm = []
        self.history_u_z = []
        self.history_u_rot_inner = []
        self.history_u_rot_outer = []
        self.history_gain_x = []
        self.history_gain_y = []
        self.history_gain_z = []
        self.sim_time_accumulator = 0.0
        
        # --- Real-Time Plotting Variables ---
        self.fig = None
        self.frame_counter = 0
        self.plot_update_frequency = 15  # Updates screen every 15 simulation frames to prevent lag

    def onKeypressedEvent(self, c):
        if str(c['key']).upper() == "V":
            self.active = not self.active
            if self.active:
                tip_pose = self.rootNode.CTR.DOFs.position.value[-1]
                self.x_m = np.array(tip_pose[:3])
                self.Ky_I = np.array([0.1, 0.1, 0.2])
                self.Ky = np.copy(self.Ky_I)
                
                # Reset historical arrays
                self.history_time = []
                self.history_err_norm = []
                self.history_u_z = []
                self.history_u_rot_inner = []
                self.history_u_rot_outer = []
                self.history_gain_x = []
                self.history_gain_y = []
                self.history_gain_z = []
                self.sim_time_accumulator = 0.0
                self.frame_counter = 0
                
                print(f"\033[94m[MRAC + RealTime Plot] STARTED targeting {self.r}\033[0m")
                
                # Launch the live plotting window
                self.setup_live_plot()
                
            else:
                print("\033[91m[MRAC] Loop STOPPED by user.\033[0m")
                self.finalize_plot()

    def onAnimateBeginEvent(self, event):
        if not self.active or self.x_m is None:
            return

        try:
            dt = self.rootNode.dt.value
            if dt <= 0:
                return
            
            self.sim_time_accumulator += dt
            self.frame_counter += 1
            
            # 1. Capture actual physical tip state from SOFA
            tip_pose = self.rootNode.CTR.DOFs.position.value[-1]
            x = np.array(tip_pose[:3])
            
            # 2. Update Reference Model
            dx_m = -self.Am * self.x_m + self.Bm * self.r
            self.x_m += dx_m * dt
            
            # 3. Calculate Errors
            error_to_target = self.r - x
            e_a = x - self.x_m  
            err_norm = np.linalg.norm(error_to_target)  
            
            # 4. Strategy 1: Proportional-Integral (PI) Adaptation Law with Leakage
            if err_norm > 1.5:
                # 1. Update the Integral Part (with leakage)
                self.Ky_I[0] += (self.gamma_rot_I * e_a[0] * self.x_m[0] - self.sigma * self.Ky_I[0]) * dt   
                self.Ky_I[1] += (self.gamma_rot_I * e_a[1] * self.x_m[1] - self.sigma * self.Ky_I[1]) * dt   
                self.Ky_I[2] += (self.gamma_trans_I * e_a[2] * self.x_m[2] - self.sigma * self.Ky_I[2]) * dt 
                
                self.Ky_I = np.clip(self.Ky_I, 0.01, 5.0)
                
                # 2. Add the Proportional Part directly to calculate total gain
                self.Ky[0] = self.Ky_I[0] + self.gamma_rot_P * e_a[0] * self.x_m[0]
                self.Ky[1] = self.Ky_I[1] + self.gamma_rot_P * e_a[1] * self.x_m[1]
                self.Ky[2] = self.Ky_I[2] + self.gamma_trans_P * e_a[2] * self.x_m[2]
                
                self.Ky = np.clip(self.Ky, 0.01, 5.0)
            else:
                # Slowly relax the integral gains when close to the target
                self.Ky_I -= self.sigma * self.Ky_I * dt
                self.Ky_I = np.clip(self.Ky_I, 0.01, 5.0)
                # Without the proportional boost, total gain smoothly equals the decaying integral gain
                self.Ky = np.copy(self.Ky_I)
            
            # 5. Generate Control Signal Inputs
            u_x = self.Ky[0] * error_to_target[0]
            u_y = self.Ky[1] * error_to_target[1]
            u_z = self.Ky[2] * error_to_target[2]
            
            step_z = np.clip(u_z, -0.5, 0.5)
            step_rot_x = np.clip(u_x, -np.radians(3), np.radians(3))
            step_rot_y = np.clip(u_y, -np.radians(3), np.radians(3))
            
            # 6. Apply commands
            move_outer = x[2] < self.stop_at_z
            with self.ir_controller.xtip.writeable() as xtip:
                xtip[1] += step_z  
                if move_outer:
                    xtip[0] += step_z  
                    
            with self.ir_controller.rotationInstrument.writeable() as rotation:
                rotation[1] += step_rot_x  
                rotation[0] += step_rot_y  
                
            # 7. Append running data tracking metrics
            self.history_time.append(self.sim_time_accumulator)
            self.history_err_norm.append(err_norm)
            self.history_u_z.append(step_z)
            self.history_u_rot_inner.append(np.degrees(step_rot_x)) 
            self.history_u_rot_outer.append(np.degrees(step_rot_y))
            self.history_gain_x.append(self.Ky[0])
            self.history_gain_y.append(self.Ky[1])
            self.history_gain_z.append(self.Ky[2])
                
            # --- REAL-TIME PLOT UPDATE TRIGGER ---
            if self.frame_counter % self.plot_update_frequency == 0:
                self.update_live_plot()
            
            # 8. Clean Component and Norm Logs
            print(f"MRAC | Norm: {err_norm:5.2f} mm | Components [X: {error_to_target[0]:+5.2f}, Y: {error_to_target[1]:+5.2f}, Z: {error_to_target[2]:+5.2f}] | "
                  f"Gains: [{self.Ky[0]:.3f}, {self.Ky[1]:.3f}, {self.Ky[2]:.3f}]")
            
            # --- Settling Filter Exit Check ---
            actuator_velocity_norm = np.sqrt(step_z**2 + step_rot_x**2 + step_rot_y**2)
            
            if err_norm < 0.2 and actuator_velocity_norm < 0.01:
                print("\n" + "="*70)
                Sofa.Helper.msg_info(self, "\033[1;92m[SUCCESS] Limit cycle broken! Tip settled perfectly under 0.2mm!\033[0m")
                print("="*70 + "\n")
                self.active = False
                self.update_live_plot() # Final update to ensure plot is perfectly aligned
                self.finalize_plot()
                
        except Exception as e:
            print(f"[MRAC Runtime Error] {e}")
            self.active = False

    # =========================================================================
    # REAL-TIME PLOTTING METHODS
    # =========================================================================

    def setup_live_plot(self):
        """Initializes the matplotlib interactive mode and layout."""
        plt.ion()  # Turn on interactive mode
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
        self.fig.suptitle("MRAC Adaptive Controller LIVE Dashboard", fontsize=14, fontweight='bold')
        
        # Initialize empty lines for Subplot 1
        self.line_err, = self.ax1.plot([], [], color='firebrick', linewidth=2.0, label="Tip Distance Error")
        self.ax1.axhline(y=0.2, color='forestgreen', linestyle='--', alpha=0.7, label="Threshold (0.2mm)")
        self.ax1.axhline(y=1.5, color='darkorange', linestyle=':', alpha=0.7, label="Freeze Boundary (1.5mm)")
        self.ax1.set_ylabel("Error Norm (mm)", fontsize=10)
        self.ax1.grid(True, linestyle=':', alpha=0.6)
        self.ax1.legend(loc="upper right")
        
        # Initialize empty lines for Subplot 2
        self.line_uz, = self.ax2.plot([], [], color='teal', linewidth=1.5, label="Translation (mm/frame)")
        self.line_rot_in, = self.ax2.plot([], [], color='darkorange', linewidth=1.5, label="Inner Twist (deg/frame)")
        self.line_rot_out, = self.ax2.plot([], [], color='purple', linewidth=1.5, label="Outer Twist (deg/frame)")
        self.ax2.set_ylabel("Control Actions", fontsize=10)
        self.ax2.grid(True, linestyle=':', alpha=0.6)
        self.ax2.legend(loc="lower right")

        # Initialize empty lines for Subplot 3
        self.line_gx, = self.ax3.plot([], [], color='crimson', linewidth=2.0, label="Gain X ($K_{y,x}$)")
        self.line_gy, = self.ax3.plot([], [], color='royalblue', linewidth=2.0, label="Gain Y ($K_{y,y}$)")
        self.line_gz, = self.ax3.plot([], [], color='darkgreen', linewidth=2.0, label="Gain Z ($K_{y,z}$)")
        self.ax3.set_xlabel("Simulation Timeline (seconds)", fontsize=11)
        self.ax3.set_ylabel("Adaptive Weights ($K_y$)", fontsize=10)
        self.ax3.grid(True, linestyle=':', alpha=0.6)
        self.ax3.legend(loc="upper right")

        plt.tight_layout()
        self.fig.show()
        self.fig.canvas.flush_events()

    def update_live_plot(self):
        """Injects new data into the existing plot lines without blocking SOFA."""
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            return # Prevent errors if the user manually closes the window early

        # Update data dynamically
        self.line_err.set_data(self.history_time, self.history_err_norm)
        
        self.line_uz.set_data(self.history_time, self.history_u_z)
        self.line_rot_in.set_data(self.history_time, self.history_u_rot_inner)
        self.line_rot_out.set_data(self.history_time, self.history_u_rot_outer)
        
        self.line_gx.set_data(self.history_time, self.history_gain_x)
        self.line_gy.set_data(self.history_time, self.history_gain_y)
        self.line_gz.set_data(self.history_time, self.history_gain_z)

        # Rescale axes continuously to fit the new data
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.relim()
            ax.autoscale_view()

        # Flush the GUI event queue to render the frame
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def finalize_plot(self):
        """Converts the plot back to blocking mode when the run finishes."""
        if self.fig is not None and plt.fignum_exists(self.fig.number):
            plt.ioff()  # Turn off interactive mode
            plt.show()  # Keep the final window open indefinitely


class MCSAdaptiveController(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.name = "MCSController"
        self.ir_controller = kwargs.get('irController')
        self.rootNode = kwargs.get('rootNode')
        
        # --- Target Input Vector r = [X, Y, Z]^T ---
        raw_target = kwargs.get('target', [0.0, 0.0, 20.0])
        self.r = np.array(raw_target[:3])
        self.stop_at_z = kwargs.get('stop_at_z', 30.0)
        
        # --- Reference Model (3D) ---
        raw_Am = kwargs.get('Am', -0.50)
        raw_Bm = kwargs.get('Bm', 0.50)
        
        if isinstance(raw_Am, (int, float)):
            self.Am = np.array([float(raw_Am)] * 3)
        else:
            self.Am = np.array(raw_Am)
            
        if isinstance(raw_Bm, (int, float)):
            self.Bm = np.array([float(raw_Bm)] * 3)
        else:
            self.Bm = np.array(raw_Bm)
            
        self.x_m = None  
        
        # --- Hyperparameters ---
        self.alpha = kwargs.get('alpha', 0.1)     # Integral learning rate
        self.beta = kwargs.get('beta', 0.02)      # Proportional learning rate
        self.sigma = kwargs.get('sigma', 0.05)    # Leakage factor for anti-windup
        
        # --- Standard MCS Adaptive Gains (3x3 matrix) ---
        self.K_xI = np.zeros((3, 3))  
        self.K_rI = np.zeros((3, 3))  
        
        # Ce matrix mapping 3D tip errors to 3 Control Inputs [rot_in, rot_out, z_step]
        self.Ce = np.array([
            [-1.0,  0.0,  0.0],  # u[0] (Inner Rot step) -> X error
            [ 0.0, -1.0,  0.0],  # u[1] (Outer Rot step) -> Y error
            [ 0.0,  0.0,  1.0]   # u[2] (Z step)         -> Z error
        ])
        
        self.active = False
        self.sim_time_accumulator = 0.0
        self.frame_counter = 0

        # --- Performance Plot Data Buffers ---
        self.history_time = []
        self.history_err_norm = []
        self.history_u_z = []
        self.history_u_rot_inner = []
        self.history_u_rot_outer = []
        self.history_gain_x = []
        self.history_gain_y = []
        
        # --- Real-Time Plotting Variables ---
        self.fig = None
        self.plot_update_frequency = 15

    def onKeypressedEvent(self, c):
        if str(c['key']).upper() == "N":
            self.active = not self.active
            if self.active:
                try:
                    tip_pose = self.rootNode.CTR.DOFs.position.value[-1]
                    self.x_m = np.array(tip_pose[:3])
                    
                    self.K_xI = np.zeros((3, 3))
                    self.K_rI = np.zeros((3, 3))
                    
                    self.sim_time_accumulator = 0.0
                    self.frame_counter = 0
                    
                    # Reset historical arrays
                    self.history_time = []
                    self.history_err_norm = []
                    self.history_u_z = []
                    self.history_u_rot_inner = []
                    self.history_u_rot_outer = []
                    self.history_gain_x = []
                    self.history_gain_y = []
                    
                    print(f"\033[94m[Standard-MCS] Loop STARTED targeting {self.r}\033[0m")
                    self.setup_live_plot()
                except Exception as e:
                    print(f"[MCS Init Error] {e}")
                    self.active = False
            else:
                print("\033[91m[Standard-MCS] Loop STOPPED\033[0m")
                self.finalize_plot()

    def onAnimateBeginEvent(self, event):
        if not self.active or self.x_m is None:
            return

        try:
            dt = self.rootNode.dt.value
            if dt <= 0: return
            
            self.sim_time_accumulator += dt
            self.frame_counter += 1
            
            # Use inner tube tip for overall tracking
            tip_pose = self.rootNode.CTR.DOFs.position.value[-1]
            x = np.array(tip_pose[:3])
            
            if np.any(np.isnan(x)):
                print("\033[91m[MCS ERROR] NaN detected! Disengaging.\033[0m")
                self.active = False
                return
            # --- 1. Update Reference Model ---
            dx_m = (self.Am * self.x_m) + (self.Bm * self.r)
            self.x_m += dx_m * dt
            
            # --- 2. Calculate Errors ---
            e = self.x_m - x           
            e_track = self.r - x       
            
            err_norm = np.linalg.norm(e_track)
            
            # --- 3. Basic MCS Adaptive Laws ---
            y_e = self.Ce @ e
            
            # Integral Updates WITH Leakage (sigma-modification)
            self.K_xI += (self.alpha * np.outer(y_e, x) - self.sigma * self.K_xI) * dt
            self.K_rI += (self.alpha * np.outer(y_e, self.r) - self.sigma * self.K_rI) * dt
            
            # Proportional Gains
            K_x = self.K_xI + self.beta * np.outer(y_e, x)
            K_r = self.K_rI + self.beta * np.outer(y_e, self.r)
            
            # --- 4. Generate Control Signal Inputs ---
            u = K_x @ x + K_r @ self.r
            
            # --- 5. Actuator Mapping & Clamping ---
            max_z_step = 0.5
            max_rot_step = np.radians(3) 
            
            step_rot_inner = np.clip(u[0], -max_rot_step, max_rot_step)
            step_rot_outer = np.clip(u[1], -max_rot_step, max_rot_step)
            step_z = np.clip(u[2], -max_z_step, max_z_step)
            
            # --- 6. Apply commands ---
            move_outer = x[2] < self.stop_at_z
            with self.ir_controller.xtip.writeable() as xtip:
                xtip[1] += step_z
                if move_outer:
                    xtip[0] += step_z
                    
            with self.ir_controller.rotationInstrument.writeable() as rotation:
                rotation[1] += step_rot_inner
                rotation[0] += step_rot_outer
                
            # 7. Append running data tracking metrics
            self.history_time.append(self.sim_time_accumulator)
            self.history_err_norm.append(err_norm)
            self.history_u_z.append(step_z)
            self.history_u_rot_inner.append(np.degrees(step_rot_inner)) 
            self.history_u_rot_outer.append(np.degrees(step_rot_outer))
            self.history_gain_x.append(np.max(np.abs(self.K_xI)))
            self.history_gain_y.append(np.max(np.abs(self.K_rI)))
                
            # --- REAL-TIME PLOT UPDATE TRIGGER ---
            if self.frame_counter % self.plot_update_frequency == 0:
                self.update_live_plot()
                
            # --- 8. Print Telemetry ---
            if self.frame_counter % 10 == 0:
                print(f"Std-MCS | Err: {err_norm:5.2f} mm | Z_step: {step_z:.3f} | Rot_in: {step_rot_inner:.3f} | Rot_out: {step_rot_outer:.3f}")
                print(f"    DEBUG -> e (x_m - x): {np.round(e, 3)}")
                print(f"    DEBUG -> K_xI max: {np.max(np.abs(self.K_xI)):.5f} | K_rI max: {np.max(np.abs(self.K_rI)):.5f}")
            
            actuator_velocity_norm = np.linalg.norm([step_z, step_rot_inner, step_rot_outer])
            if err_norm < 0.2 and actuator_velocity_norm < 0.01:
                print("\n" + "="*70)
                Sofa.Helper.msg_info(self, "\033[1;92m[SUCCESS] Standard MCS Stabilized!\033[0m")
                print("="*70 + "\n")
                self.active = False
                self.update_live_plot()
                self.finalize_plot()
                
        except Exception as e:
            print(f"[MCS Runtime Error] {e}")
            self.active = False

    # =========================================================================
    # REAL-TIME PLOTTING METHODS
    # =========================================================================

    def setup_live_plot(self):
        """Initializes the matplotlib interactive mode and layout."""
        plt.ion()  # Turn on interactive mode
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
        self.fig.suptitle("MCS Adaptive Controller LIVE Dashboard", fontsize=14, fontweight='bold')
        
        # Initialize empty lines for Subplot 1
        self.line_err, = self.ax1.plot([], [], color='firebrick', linewidth=2.0, label="Tip Distance Error")
        self.ax1.axhline(y=0.2, color='forestgreen', linestyle='--', alpha=0.7, label="Threshold (0.2mm)")
        self.ax1.axhline(y=1.5, color='darkorange', linestyle=':', alpha=0.7, label="Freeze Boundary (1.5mm)")
        self.ax1.set_ylabel("Error Norm (mm)", fontsize=10)
        self.ax1.grid(True, linestyle=':', alpha=0.6)
        self.ax1.legend(loc="upper right")
        
        # Initialize empty lines for Subplot 2
        self.line_uz, = self.ax2.plot([], [], color='teal', linewidth=1.5, label="Translation (mm/frame)")
        self.line_rot_in, = self.ax2.plot([], [], color='darkorange', linewidth=1.5, label="Inner Twist (deg/frame)")
        self.line_rot_out, = self.ax2.plot([], [], color='purple', linewidth=1.5, label="Outer Twist (deg/frame)")
        self.ax2.set_ylabel("Control Actions", fontsize=10)
        self.ax2.grid(True, linestyle=':', alpha=0.6)
        self.ax2.legend(loc="lower right")

        # Initialize empty lines for Subplot 3
        self.line_gx, = self.ax3.plot([], [], color='crimson', linewidth=2.0, label="Max $|K_{xI}|$")
        self.line_gy, = self.ax3.plot([], [], color='royalblue', linewidth=2.0, label="Max $|K_{rI}|$")
        self.ax3.set_xlabel("Simulation Timeline (seconds)", fontsize=11)
        self.ax3.set_ylabel("Adaptive Gains", fontsize=10)
        self.ax3.grid(True, linestyle=':', alpha=0.6)
        self.ax3.legend(loc="upper right")

        plt.tight_layout()
        self.fig.show()
        self.fig.canvas.flush_events()

    def update_live_plot(self):
        """Injects new data into the existing plot lines without blocking SOFA."""
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            return

        # Update data dynamically
        self.line_err.set_data(self.history_time, self.history_err_norm)
        
        self.line_uz.set_data(self.history_time, self.history_u_z)
        self.line_rot_in.set_data(self.history_time, self.history_u_rot_inner)
        self.line_rot_out.set_data(self.history_time, self.history_u_rot_outer)
        
        self.line_gx.set_data(self.history_time, self.history_gain_x)
        self.line_gy.set_data(self.history_time, self.history_gain_y)

        # Rescale axes continuously to fit the new data
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.relim()
            ax.autoscale_view()

        # Flush the GUI event queue to render the frame
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def finalize_plot(self):
        """Converts the plot back to blocking mode when the run finishes."""
        if self.fig is not None and plt.fignum_exists(self.fig.number):
            plt.ioff()
            plt.show()
