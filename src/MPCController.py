import numpy as np
import scipy.optimize as opt
import Sofa.Core
import Sofa.Helper
import matplotlib.pyplot as plt
import os
from .setup import load_tube_parameters

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')

params = load_tube_parameters(CONFIG_PATH)

# --- Extract CTR parameters ---
Straight_length_1, Straight_length_2 = params["Straight_length"]
Curved_length_1,   Curved_length_2  = params["Curved_length"]
Tube_radius_1,     Tube_radius_2     = params["Tube_radius"]
Radius_curvature_1,Radius_curvature_2 = params["Radius_curvature"]
Tube1_young_modulus,Tube2_young_modulus = params["Young_modulus"]

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
        # u = [step_z_inner, step_z_outer, step_rot_in, step_rot_out]
        # Balance the outer and inner tube penalties so both are used!
        self.R = np.diag([1.0, 1.0, 10.0, 10.0])
        

        self.J = None
        
        # State tracking
        self.active = False
        self.prev_x = None
        self.prev_u = np.zeros(4)
        
        # --- Performance Plot Data Buffers ---
        self.history_time = []
        self.history_x = []
        self.history_y = []
        self.history_z = []
        self.history_q_z_inner = []
        self.history_q_z_outer = []
        self.history_q_rot_inner = []
        self.history_q_rot_outer = []
        
        # --- Real-Time Plotting Variables ---
        self.fig = None
        self.plot_update_frequency = 15
        self.sim_time_accumulator = 0.0
        self.frame_counter = 0

    def ctr_forward_kinematics(self, q):
        """
        Fast analytical model of a 2-tube CTR based on constant-curvature kinematics.
        q = [z_in, z_out, rot_in, rot_out]
        """
        z_in, z_out, rot_in, rot_out = q
        
        # Base lengths (Inner tube is 2, Outer tube is 1)
        L_s1 = Straight_length_1; L_c1 = Curved_length_1;  L_1 = L_s1 + L_c1
        L_s2 = Straight_length_2; L_c2 = Curved_length_2;  L_2 = L_s2 + L_c2
        
        # Curvatures
        k_c1 = 1.0 / Radius_curvature_1 if Radius_curvature_1 > 0 else 0.0
        k_c2 = 1.0 / Radius_curvature_2 if Radius_curvature_2 > 0 else 0.0
        
        # Stiffness E*I approx (Solid rods)
        EI_1 = Tube1_young_modulus * (Tube_radius_1**4)
        EI_2 = Tube2_young_modulus * (Tube_radius_2**4)
        
        # Transition points in arc length s (base is at s=0)
        s_t1 = z_out + L_s1
        s_1  = z_out + L_1
        s_t2 = z_in + L_s2
        s_2  = z_in + L_2
        
        # Collect and sort valid transition points up to the inner tip
        pts = [0.0, s_t1, s_1, s_t2, s_2]
        pts = sorted([p for p in pts if 0.0 <= p <= s_2])
        
        T = np.eye(4)
        for i in range(len(pts)-1):
            s_a = pts[i]
            s_b = pts[i+1]
            L = s_b - s_a
            if L < 1e-6: continue
                
            s_mid = (s_a + s_b) / 2.0
            
            has_t1 = (s_mid <= s_1)
            has_t2 = (s_mid <= s_2)
            k1 = k_c1 if (has_t1 and s_mid > s_t1) else 0.0
            k2 = k_c2 if (has_t2 and s_mid > s_t2) else 0.0
            
            if has_t1 and has_t2:
                kx = (EI_1 * k1 * np.cos(rot_out) + EI_2 * k2 * np.cos(rot_in)) / (EI_1 + EI_2)
                ky = (EI_1 * k1 * np.sin(rot_out) + EI_2 * k2 * np.sin(rot_in)) / (EI_1 + EI_2)
            elif has_t1:
                kx = k1 * np.cos(rot_out)
                ky = k1 * np.sin(rot_out)
            elif has_t2:
                kx = k2 * np.cos(rot_in)
                ky = k2 * np.sin(rot_in)
            else:
                kx = 0.0; ky = 0.0
                
            k_eq = np.sqrt(kx**2 + ky**2)
            phi = np.arctan2(ky, kx)
            
            if k_eq < 1e-6:
                T_arc = np.eye(4)
                T_arc[2, 3] = L
                T = T @ T_arc
            else:
                c_phi = np.cos(phi)
                s_phi = np.sin(phi)
                Rz = np.array([[c_phi, -s_phi, 0, 0], [s_phi, c_phi, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                Rz_inv = np.array([[c_phi, s_phi, 0, 0], [-s_phi, c_phi, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                ckL = np.cos(k_eq * L)
                skL = np.sin(k_eq * L)
                T_arc = np.array([
                    [ckL,  0, skL, (1 - ckL) / k_eq],
                    [0,    1, 0,   0],
                    [-skL, 0, ckL, skL / k_eq],
                    [0,    0, 0,   1]
                ])
                T = T @ (Rz @ T_arc @ Rz_inv)
                
        # Base frame aligns naturally with Z-axis in SOFA
        return T[0:3, 3]
        
    def compute_numerical_jacobian(self, q, delta_q=1e-4):
        num_joints = 4
        num_task_dims = 3
        J = np.zeros((num_task_dims, num_joints))
        
        for i in range(num_joints):
            e_n = np.zeros(num_joints)
            e_n[i] = 1.0
            
            q_plus = q + (delta_q / 2.0) * e_n
            x_plus = self.ctr_forward_kinematics(q_plus)
            
            q_minus = q - (delta_q / 2.0) * e_n
            x_minus = self.ctr_forward_kinematics(q_minus)
            
            J[:, i] = (x_plus - x_minus) / delta_q
            
        return J

    def onKeypressedEvent(self, c):
        if str(c['key']).upper() == "C":
            self.active = not self.active
            if self.active:
                tip_pose = self.rootNode.CTR.DOFs.position.value[-1]
                self.prev_x = np.array(tip_pose[:3])
                self.prev_u = np.zeros(4)
                
                self.history_time = []
                self.history_x = []
                self.history_y = []
                self.history_z = []
                self.history_q_z_inner = []
                self.history_q_z_outer = []
                self.history_q_rot_inner = []
                self.history_q_rot_outer = []
                
                self.frame_counter = 0
                self.sim_time_accumulator = 0.0
                
                print(f"\033[94m[MPC] Controller STARTED targeting {self.target}\033[0m")
                self.setup_live_plot()
            else:
                print("\033[91m[MPC] Controller STOPPED\033[0m")
                self.finalize_plot()

    def mpc_cost(self, U_flat, x_current, q_current):
        """ Cost function for the MPC optimizer """
        U = U_flat.reshape((self.N, 4))
        x_pred = np.copy(x_current)
        q_pred = np.copy(q_current)
        cost = 0.0
        
        for k in range(self.N):
            u_k = U[k]
            # Predict next state using local Jacobian
            x_pred = x_pred + self.J @ u_k
            q_pred = q_pred + u_k
            
            # Error to target
            e = x_pred - self.target
            
            # Gap penalty (Secondary Task: force outer tube to stay ~20mm behind inner)
            gap = q_pred[0] - q_pred[1]
            gap_error = gap - 20.0
            W_gap = 10.0
            
            # Quadratic cost
            cost += e.T @ self.Q @ e + u_k.T @ self.R @ u_k + W_gap * (gap_error**2)
            
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
            x_current = np.array(tip_pose[0:3])
            
            # Error checking
            if np.any(np.isnan(x_current)):
                print("\033[91m[MPC ERROR] NaN detected! Disengaging.\033[0m")
                self.active = False
                return

            # 2. Update Numerical Jacobian via Forward Kinematics
            xtip = self.ir_controller.xtip.value
            rotation = self.ir_controller.rotationInstrument.value
            # q = [z_inner, z_outer, rot_inner, rot_outer]
            q_current = np.array([xtip[1], xtip[0], rotation[1], rotation[0]])
            self.J = self.compute_numerical_jacobian(q_current)
            
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
            
            # Actuator physical limits (e.g. 10 mm/s)
            max_z_step = 10.0 * dt
            max_rot_step = np.radians(60.0) * dt
            
            # Dynamic bounds based on physical limits
            outer_z = self.ir_controller.xtip.value[0]
            max_z_outer = max_z_step if outer_z < self.stop_at_z else 0.0
            
            # Bounds for inputs [u_z_inner, u_z_outer, u_rot_in, u_rot_out] over horizon N
            bnds = [(-max_z_step, max_z_step), 
                    (-max_z_step, max_z_outer), 
                    (-max_rot_step, max_rot_step), 
                    (-max_rot_step, max_rot_step)] * self.N
            
            # Constraints
            cons = {'type': 'ineq', 'fun': self.force_constraint, 'args': (force_value,)}
            
            # Optimize
            res = opt.minimize(self.mpc_cost, U0, args=(x_current, q_current), bounds=bnds, constraints=cons, method='SLSQP', options={'maxiter': 20, 'ftol': 1e-3})
            
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
            
            step_z_inner = float(u_optimal[0])
            step_z_outer = float(u_optimal[1])
            step_rot_inner = float(u_optimal[2])
            step_rot_outer = float(u_optimal[3])
            
            # 5. Apply Commands Safely
            curr_outer = self.ir_controller.xtip.value[0]
            curr_inner = self.ir_controller.xtip.value[1]
            
            # Apply absolute steps directly as expected by the new Jacobian
            new_inner = np.clip(curr_inner + step_z_inner, 0.0, 50.0) 
            new_outer = np.clip(curr_outer + step_z_outer, 0.0, 45.0)
            
            with self.ir_controller.xtip.writeable() as xtip:
                xtip[1] = float(new_inner)
                xtip[0] = float(new_outer)
                    
            curr_rot_inner = self.ir_controller.rotationInstrument.value[1]
            curr_rot_outer = self.ir_controller.rotationInstrument.value[0]
            
            with self.ir_controller.rotationInstrument.writeable() as rotation:
                rotation[1] = float(curr_rot_inner + step_rot_inner)  
                rotation[0] = float(curr_rot_outer + step_rot_outer)
                
            # 6. Save state for next step's Jacobian update
            self.prev_x = np.copy(x_current)
            self.prev_u = np.copy(u_optimal)
            
            # 7. Print Telemetry
            err_norm = np.linalg.norm(self.target - x_current)
            self.frame_counter += 1
            
            self.history_time.append(self.sim_time_accumulator)
            self.history_x.append(x_current[0])
            self.history_y.append(x_current[1])
            self.history_z.append(x_current[2])
            self.history_q_z_inner.append(new_inner)
            self.history_q_z_outer.append(new_outer)
            self.history_q_rot_inner.append(np.degrees(curr_rot_inner + step_rot_inner)) 
            self.history_q_rot_outer.append(np.degrees(curr_rot_outer + step_rot_outer))
            
            if self.frame_counter % self.plot_update_frequency == 0:
                self.update_live_plot()
            if self.frame_counter % 10 == 0:
                print(f"MPC | Err: {err_norm:5.2f} mm | Z_out: {step_z_outer:5.2f} | Z_in: {step_z_inner:5.2f} | rot: [{step_rot_inner:5.2f}, {step_rot_outer:5.2f}]")
                
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
        if self.fig is None or not plt.fignum_exists(self.fig.number):
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

    def finalize_plot(self):
        """Saves the final plot instead of blocking the SOFA thread."""
        if self.fig is not None and plt.fignum_exists(self.fig.number):
            try:
                images_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images')
                os.makedirs(images_dir, exist_ok=True)
                save_path = os.path.join(images_dir, 'MPC_Final_Plot.png')
                self.fig.savefig(save_path)
                print(f"\033[92m[MPC] Final plot saved in {save_path}\033[0m")
            except Exception as e:
                print(f"\033[91m[MPC] Failed to save plot: {e}\033[0m")
