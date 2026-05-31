import numpy as np
import scipy.optimize as opt
import Sofa.Core
import Sofa.Helper
import matplotlib.pyplot as plt

class MPCPositionController(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.name = "MPCPositionController"
        self.ir_controller = kwargs.get('irController')
        self.rootNode = kwargs.get('rootNode')
        
        # --- Target Input Vector r = [X, Y, Z]^T ---
        self.target = np.array(kwargs.get('target', [4.49, 4.0, 46.5]))
        self.stop_at_z = kwargs.get('stop_at_z', 25.0)
        
        # --- MPC Hyperparameters ---
        self.N = kwargs.get('N', 5)  # Prediction Horizon
        
        # State cost (Q) - Penality for distance to target
        self.Q = np.diag([50.0, 50.0, 50.0])
        
        # Control cost (R) - Penalty for control effort to ensure smoothness
        # u = [step_z_outer, step_z_inner, step_rot_in, step_rot_out]
        # Make the outer tube (index 0) 5x more expensive than the inner tube.
        # This FORCES the optimizer to use the inner tube to reach the target!
        self.R = np.diag([5.0, 1.0, 10.0, 10.0])
        
        # --- Actuator limits ---
        self.max_z_step = 0.5
        self.max_rot_step = np.radians(3.0)  # ~0.052 rad
        
        # --- Adaptive Local Linear Model (Jacobian) ---
        # J maps control inputs (u_z_out, u_z_in, u_rot_in, u_rot_out) to tip velocities (vx, vy, vz)
        self.J = np.array([
            [0.0, 0.0, -10.0,   0.0],
            [0.0, 0.0,   0.0, -10.0],
            [1.0, 1.0,   0.0,   0.0]
        ])
        
        self.alpha_broyden = 0.5  # Learning rate for Jacobian update
        
        # State tracking
        self.active = False
        self.prev_x = None
        self.prev_u = np.zeros(4)
        
        # --- Performance Plot Data Buffers ---
        self.history_time = []
        self.history_err_norm = []
        self.history_u_z_outer = []
        self.history_u_z_inner = []
        self.history_u_rot_inner = []
        self.history_u_rot_outer = []
        
        # --- Real-Time Plotting Variables ---
        self.fig = None
        self.plot_update_frequency = 15
        self.sim_time_accumulator = 0.0
        self.frame_counter = 0

    def onKeypressedEvent(self, c):
        if str(c['key']).upper() == "P":
            self.active = not self.active
            if self.active:
                tip_pose = self.rootNode.CTR.DOFs.position.value[-1]
                self.prev_x = np.array(tip_pose[:3])
                self.prev_u = np.zeros(4)
                
                self.history_time = []
                self.history_err_norm = []
                self.history_u_z_outer = []
                self.history_u_z_inner = []
                self.history_u_rot_inner = []
                self.history_u_rot_outer = []
                
                self.frame_counter = 0
                self.sim_time_accumulator = 0.0
                
                print(f"\033[94m[MPC] Controller STARTED targeting {self.target}\033[0m")
                self.setup_live_plot()
            else:
                print("\033[91m[MPC] Controller STOPPED\033[0m")
                self.finalize_plot()

    def mpc_cost(self, U_flat, x_current):
        """ Cost function for the MPC optimizer """
        U = U_flat.reshape((self.N, 4))
        x_pred = np.copy(x_current)
        cost = 0.0
        
        for k in range(self.N):
            u_k = U[k]
            # Predict next state using local Jacobian
            x_pred = x_pred + self.J @ u_k
            
            # Error to target
            e = x_pred - self.target
            
            # Quadratic cost
            cost += e.T @ self.Q @ e + u_k.T @ self.R @ u_k
            
        return cost

    def force_constraint(self, U_flat, current_force):
        """ Inequality constraint: 0.4 - predicted_force >= 0 """
        U = U_flat.reshape((self.N, 4))
        K_stiff = 2.0  # Estimated stiffness: 2.0 N per mm of insertion
        f_preds = []
        f_k = current_force
        for k in range(self.N):
            # Both insertions contribute to force
            f_k += K_stiff * (U[k, 0] + U[k, 1])
            f_preds.append(0.4 - f_k)
        return np.array(f_preds)

    def onAnimateBeginEvent(self, event):
        if not self.active:
            return

        try:
            dt = self.rootNode.dt.value
            if dt <= 0: return
            
            self.sim_time_accumulator += dt
            
            # 1. Get current state
            tip_pose = self.rootNode.CTR.DOFs.position.value[-1]
            x_current = np.array(tip_pose[:3])
            
            # Error checking
            if np.any(np.isnan(x_current)):
                print("\033[91m[MPC ERROR] NaN detected! Disengaging.\033[0m")
                self.active = False
                return

            # 2. Update Adaptive Jacobian (Broyden's method)
            if self.prev_x is not None:
                dx = x_current - self.prev_x
                du = self.prev_u
                
                # Only update if we moved sufficiently
                if np.linalg.norm(du) > 1e-5:
                    error_pred = dx - (self.J @ du)
                    # Broyden update rule with learning rate alpha
                    self.J += self.alpha_broyden * np.outer(error_pred, du) / (np.dot(du, du) + 1e-8)
            
            # 2.5 Read Force Sensor
            force_value = 0.0
            try:
                solver = self.rootNode.getObject("solver")
                if solver is not None:
                    forces_vector = solver.constraintForces.value
                    if forces_vector is not None and len(forces_vector) > 0:
                        force_value = np.linalg.norm(forces_vector)
            except Exception as e:
                pass
                
            # 3. Setup MPC Optimization Problem
            # Initial guess for optimization (zeros)
            U0 = np.zeros(self.N * 4)
            
            # Dynamic bounds based on physical limits
            outer_z = self.ir_controller.xtip.value[0]
            max_z_outer = self.max_z_step if outer_z < self.stop_at_z else 0.0
            
            # Bounds for inputs [u_z_out, u_z_in, u_rot_in, u_rot_out] over horizon N
            bnds = [(-self.max_z_step, max_z_outer), 
                    (-self.max_z_step, self.max_z_step), 
                    (-self.max_rot_step, self.max_rot_step), 
                    (-self.max_rot_step, self.max_rot_step)] * self.N
            
            # Constraints
            cons = {'type': 'ineq', 'fun': self.force_constraint, 'args': (force_value,)}
            
            # Optimize
            res = opt.minimize(self.mpc_cost, U0, args=(x_current,), bounds=bnds, constraints=cons, method='SLSQP', options={'maxiter': 20, 'ftol': 1e-3})
            
            if not res.success and self.frame_counter % 10 == 0:
                print(f"\033[93m[MPC Warning] Optimizer did not converge perfectly: {res.message}\033[0m")
            
            if force_value > 0.4 and self.frame_counter % 10 == 0:
                print(f"\033[91m[MPC Warning] Force = {force_value:.2f} N! Optimizer actively limiting insertion.\033[0m")
            
            # 4. Extract optimal first control action
            U_opt = res.x.reshape((self.N, 4))
            u_optimal = U_opt[0]
            
            if np.any(np.isnan(u_optimal)):
                u_optimal = np.zeros(4)
                if self.frame_counter % 10 == 0:
                    print("\033[91m[MPC Error] Optimizer returned NaN. Sending zero control.\033[0m")
            
            step_z_outer = float(u_optimal[0])
            step_z_inner = float(u_optimal[1])
            step_rot_x = float(u_optimal[2])
            step_rot_y = float(u_optimal[3])
            
            # 5. Apply Commands Safely
            curr_outer = self.ir_controller.xtip.value[0]
            curr_inner = self.ir_controller.xtip.value[1]
            
            new_outer = np.clip(curr_outer + step_z_outer, 0.0, 45.0)
            new_inner = np.clip(curr_inner + step_z_outer + step_z_inner, 0.0, 50.0) # Match workspace_generator limits
            
            with self.ir_controller.xtip.writeable() as xtip:
                xtip[0] = float(new_outer)
                xtip[1] = float(new_inner)
                    
            curr_rot_outer = self.ir_controller.rotationInstrument.value[0]
            curr_rot_inner = self.ir_controller.rotationInstrument.value[1]
            
            with self.ir_controller.rotationInstrument.writeable() as rotation:
                rotation[0] = float(curr_rot_outer + step_rot_y)
                rotation[1] = float(curr_rot_inner + step_rot_x)  
                
            # 6. Save state for next step's Jacobian update
            self.prev_x = np.copy(x_current)
            self.prev_u = np.copy(u_optimal)
            
            # 7. Print Telemetry
            err_norm = np.linalg.norm(self.target - x_current)
            self.frame_counter += 1
            
            self.history_time.append(self.sim_time_accumulator)
            self.history_err_norm.append(err_norm)
            self.history_u_z_outer.append(step_z_outer)
            self.history_u_z_inner.append(step_z_inner)
            self.history_u_rot_inner.append(np.degrees(step_rot_x)) 
            self.history_u_rot_outer.append(np.degrees(step_rot_y))
            
            if self.frame_counter % self.plot_update_frequency == 0:
                self.update_live_plot()
                
            if self.frame_counter % 10 == 0:
                print(f"MPC | Err: {err_norm:5.2f} mm | Z_out: {step_z_outer:5.2f} | Z_in: {step_z_inner:5.2f} | rot: [{step_rot_x:5.2f}, {step_rot_y:5.2f}]")
                
            # Convergence check
            actuator_velocity_norm = np.linalg.norm(u_optimal)
            if err_norm < 0.2 and actuator_velocity_norm < 0.005:
                print("\n" + "="*70)
                Sofa.Helper.msg_info(self, "\033[1;92m[SUCCESS] MPC Target Reached!\033[0m")
                print("="*70 + "\n")
                self.active = False
                self.update_live_plot()
                self.finalize_plot()
                
        except Exception as e:
            print(f"[MPC Runtime Error] {e}")
            self.active = False

    # =========================================================================
    # REAL-TIME PLOTTING METHODS
    # =========================================================================

    def setup_live_plot(self):
        """Initializes the matplotlib interactive mode and layout."""
        plt.ion()  # Turn on interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        self.fig.suptitle("Model Predictive Control (MPC) LIVE Dashboard", fontsize=14, fontweight='bold')
        
        # Initialize empty lines for Subplot 1
        self.line_err, = self.ax1.plot([], [], color='firebrick', linewidth=2.0, label="Tip Distance Error")
        self.ax1.axhline(y=0.2, color='forestgreen', linestyle='--', alpha=0.7, label="Threshold (0.2mm)")
        self.ax1.set_ylabel("Error Norm (mm)", fontsize=10)
        self.ax1.grid(True, linestyle=':', alpha=0.6)
        self.ax1.legend(loc="upper right")
        
        # Initialize empty lines for Subplot 2
        self.line_uz_outer, = self.ax2.plot([], [], color='teal', linewidth=1.5, label="Base Z (mm)")
        self.line_uz_inner, = self.ax2.plot([], [], color='mediumturquoise', linewidth=1.5, label="Relative Inner Z (mm)")
        self.line_rot_in, = self.ax2.plot([], [], color='darkorange', linewidth=1.5, label="Inner Twist (deg)")
        self.line_rot_out, = self.ax2.plot([], [], color='purple', linewidth=1.5, label="Outer Twist (deg)")
        self.ax2.set_ylabel("Control Actions", fontsize=10)
        self.ax2.set_xlabel("Simulation Timeline (seconds)", fontsize=11)
        self.ax2.grid(True, linestyle=':', alpha=0.6)
        self.ax2.legend(loc="lower right")

        plt.tight_layout()
        self.fig.show()
        self.fig.canvas.flush_events()

    def update_live_plot(self):
        """Injects new data into the existing plot lines without blocking SOFA."""
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            return

        # Update data dynamically
        self.line_err.set_data(self.history_time, self.history_err_norm)
        
        self.line_uz_outer.set_data(self.history_time, self.history_u_z_outer)
        self.line_uz_inner.set_data(self.history_time, self.history_u_z_inner)
        self.line_rot_in.set_data(self.history_time, self.history_u_rot_inner)
        self.line_rot_out.set_data(self.history_time, self.history_u_rot_outer)

        # Rescale axes continuously to fit the new data
        for ax in [self.ax1, self.ax2]:
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
