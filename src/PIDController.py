"""
Docstring for v25.06.00.Techical project.InstrumentsController

File defining controllers for the CTR simulation
"""
import os
import Sofa
import Sofa.Core
import Sofa.Simulation
import numpy as np
import matplotlib.pyplot as plt
from .setup import load_tube_parameters

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')

params = load_tube_parameters(CONFIG_PATH)

# --- Extract CTR parameters ---
Straight_length_1, Straight_length_2 = params["Straight_length"]
Curved_length_1,   Curved_length_2  = params["Curved_length"]
Tube_radius_1,     Tube_radius_2     = params["Tube_radius"]
Radius_curvature_1,Radius_curvature_2 = params["Radius_curvature"]
Tube1_young_modulus,Tube2_young_modulus = params["Young_modulus"]


class PIDPositionController(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        # These are needed (and the normal way to override from a python class)
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.name = "PIDController"
        print(" Python::__init__::"+str(self.name))
        self.ir_controller = kwargs.get('irController')
        self.rootNode = kwargs.get('rootNode')
        
        # Target Cartesian position [X, Y, Z]
        self.target = np.array(kwargs.get('target', [0.0, 0.0, 45.0]))
        # PID Gains
        self.Kp = kwargs.get('Kp', 1.0)
        self.Ki = kwargs.get('Ki', 0.1)
        self.Kd = kwargs.get('Kd', 0.01)
        
        # Error variables
        self.prev_error = np.zeros(3)
        self.integral = np.zeros(3)
        
        # Jacobian Matrix
        self.J = None

        # Flag to activate/deactivate the controller
        self.active = False
        
        # --- Damped Least Squares (DLS) Jacobian ---
        # Base damping factor for DLS (lambda_max)
        self.lambda_max = kwargs.get('lambda_max', 0.1)
        # Threshold for manipulability measure to activate damping
        self.w_threshold = kwargs.get('w_threshold', 0.05)
        
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

    def onKeypressedEvent(self, c):
        if str(c['key']).upper() == "P": 
            self.active = not self.active
            if self.active:
                self.prev_error = np.zeros(3)
                self.integral = np.zeros(3)
                self.J = None
                
                self.history_time = []
                self.history_x = []
                self.history_y = []
                self.history_z = []
                self.history_q_z_inner = []
                self.history_q_z_outer = []
                self.history_q_rot_inner = []
                self.history_q_rot_outer = []
                self.sim_time_accumulator = 0.0
                self.frame_counter = 0
                
                print(f"\n\033[94m==================================================\033[0m")
                print(f"\033[94m[PID Controller] ACTIVATED\033[0m")
                print(f"\033[94mTarget Position: X={self.target[0]:.2f}, Y={self.target[1]:.2f}, Z={self.target[2]:.2f}\033[0m")
                print(f"\033[94m==================================================\n\033[0m")
                self.setup_live_plot()
            else:
                print(f"\n\033[91m[PID Controller] DEACTIVATED\033[0m\n")
                self.finalize_plot()

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

    def compute_snapping_avoidance_gradient(self, q, delta_q=1e-4):
        num_joints = 4
        grad_C = np.zeros(num_joints)
        
        # Base J to calculate mu_d
        J_base = self.compute_numerical_jacobian(q, delta_q)
        mu_base = J_base @ J_base.T
        
        # Desired manipulability is a sphere (Isotropic manipulability)
        # We scale it to match the current trace so it only penalizes eccentricity, not overall size
        nu = max(1e-6, np.trace(mu_base) / 3.0)
        mu_d = nu * np.eye(3)
        det_mu_d = max(1e-12, np.linalg.det(mu_d))
        
        for i in range(num_joints):
            e_n = np.zeros(num_joints)
            e_n[i] = 1.0
            
            # Forward perturbation
            q_plus = q + (delta_q / 2.0) * e_n
            J_plus = self.compute_numerical_jacobian(q_plus, delta_q)
            mu_plus = J_plus @ J_plus.T
            det_plus_mid = max(1e-12, np.linalg.det((mu_plus + mu_d) / 2.0))
            det_plus_mu = max(1e-12, np.linalg.det(mu_plus))
            C_plus = np.log(det_plus_mid) - 0.5 * np.log(det_plus_mu * det_mu_d)
            
            # Backward perturbation
            q_minus = q - (delta_q / 2.0) * e_n
            J_minus = self.compute_numerical_jacobian(q_minus, delta_q)
            mu_minus = J_minus @ J_minus.T
            det_minus_mid = max(1e-12, np.linalg.det((mu_minus + mu_d) / 2.0))
            det_minus_mu = max(1e-12, np.linalg.det(mu_minus))
            C_minus = np.log(det_minus_mid) - 0.5 * np.log(det_minus_mu * det_mu_d)
            
            grad_C[i] = (C_plus - C_minus) / delta_q
            
        return grad_C

    def onAnimateBeginEvent(self, event):
        if not self.active:
            return

        try:
            # 1. Get current tip position
            tip_pose = self.rootNode.CTR.DOFs.position.value[-1]
            tip_position = np.array(tip_pose[0:3])
            
            dt = self.rootNode.dt.value
            
            self.sim_time_accumulator += dt
            self.frame_counter += 1
            
            # 2. Update Numerical Jacobian via Finite Differences
            xtip = self.ir_controller.xtip.value
            rotation = self.ir_controller.rotationInstrument.value
            z_inner = xtip[1]
            z_outer = xtip[0]
            rot_inner = rotation[1]
            rot_outer = rotation[0]
            
            q_current = np.array([z_inner, z_outer, rot_inner, rot_outer])
            self.J = self.compute_numerical_jacobian(q_current)

            # 3. Primary Tracking Task (PID Control)
            error = self.target - tip_position
            
            # --- Anti-Windup ---
            # Prevent the integral term from growing unbounded
            max_integral = 10.0
            self.integral = np.clip(self.integral + error * dt, -max_integral, max_integral)
            
            derivative = (error - self.prev_error) / dt if dt > 0 else np.zeros(3)
            # Cartesian velocity command from the PID
            v_cartesian = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
            
            # Convert Cartesian velocity (mm/s) to step displacement (mm/step)
            v_cartesian = v_cartesian * dt
            
            # --- Cap Cartesian Velocity ---
            # Max Cartesian displacement per step
            max_v_per_sec = 20.0 
            max_v_step = max_v_per_sec * dt 
            v_norm = np.linalg.norm(v_cartesian)
            if v_norm > max_v_step:
                v_cartesian = v_cartesian / v_norm * max_v_step
            
            # --- Force Safety Check ---
            try:
                solver = self.rootNode.getObject("solver")
                forces_vector = solver.constraintForces.value
                force_value = np.linalg.norm(forces_vector) if forces_vector is not None and forces_vector.size > 0 else 0.0
            except:
                force_value = 0.0
            
            # 4. Exact Algebraic Inverse Kinematics (Null-Space Projection)
            det = np.linalg.det(self.J @ self.J.T)
            w = np.sqrt(max(0.0, det))
            if w >= self.w_threshold:
                lambda_sq = 1e-6
            else:
                lambda_sq = (self.lambda_max * (1.0 - w / self.w_threshold))**2 + 1e-6
                
            J_dls = self.J.T @ np.linalg.inv(self.J @ self.J.T + lambda_sq * np.eye(3))
            
            # Kinematic Inversion (Primary Task)
            u_primary = J_dls @ v_cartesian
            
            # Manipulability Redundancy Vector (Secondary Task)
            
            # --- COMBINED SECONDARY TASK ---
            # 1. Translation: Telescopic Coordination (Gap Controller)
            # We enforce a mathematical spring to keep the outer tube roughly 20mm behind the inner tube.
            gap = q_current[0] - q_current[1]  # z_inner - z_outer
            push_val = np.clip((gap - 20.0) * 0.5, -1.0, 1.0)
            
            # 2. Rotation: S-Divergence Snapping Avoidance
            grad_C = self.compute_snapping_avoidance_gradient(q_current)
            
            # Combine: positive push_val for gap, NEGATIVE grad_C for minimizing snapping
            grad_combined = np.array([
                -push_val,     # z_inner (Gap)
                 push_val,     # z_outer (Gap)
                -grad_C[2],    # rot_inner (Snapping)
                -grad_C[3]     # rot_outer (Snapping)
            ])
            
            alpha_dexterity = 5.0
            u_secondary = alpha_dexterity * grad_combined * dt
            
            u_null = (np.eye(4) - J_dls @ self.J) @ u_secondary
            
            # Combined Control Law
            u_joint = u_primary + u_null
            
            # 5. Process Control Actions
            # Restore clipping as a hard safety mechanism!
            # Actuator physical limits (e.g. 10 mm/s)
            max_z_step = 10.0 * dt
            max_rot_step = np.radians(60.0) * dt
            step_z_inner = np.clip(u_joint[0], -max_z_step, max_z_step)
            step_z_outer = np.clip(u_joint[1], -max_z_step, max_z_step)
            step_rot_inner = np.clip(u_joint[2], -max_rot_step, max_rot_step)
            step_rot_outer = np.clip(u_joint[3], -max_rot_step, max_rot_step)
            
            # --- Force Safety Override ---
            force_threshold = 1.5 # 1.5 Newton limit
            if force_value > force_threshold:
                if step_z_inner > 0: step_z_inner = -0.1
                if step_z_outer > 0: step_z_outer = -0.1
                step_rot_inner = 0.0
                step_rot_outer = 0.0
            
            # 6. Apply Commands
            with self.ir_controller.xtip.writeable() as xtip:
                xtip[1] += step_z_inner
                xtip[0] += step_z_outer

            with self.ir_controller.rotationInstrument.writeable() as rotation:
                rotation[1] += step_rot_inner
                rotation[0] += step_rot_outer

            # 7. Save state for next step
            self.prev_error = np.copy(error)

            # 8. Debug Logging
            err_norm = np.linalg.norm(error)
            
            # Append data for plotting
            self.history_time.append(self.sim_time_accumulator)
            self.history_x.append(tip_position[0])
            self.history_y.append(tip_position[1])
            self.history_z.append(tip_position[2])
            self.history_q_z_inner.append(step_z_inner)
            self.history_q_z_outer.append(step_z_outer)
            self.history_q_rot_inner.append(np.degrees(step_rot_inner))
            self.history_q_rot_outer.append(np.degrees(step_rot_outer))
            
            if self.frame_counter % self.plot_update_frequency == 0:
                self.update_live_plot()
                
            if self.frame_counter % 10 == 0:    
                print(f"[PID Controller] "
                      f"Pos: [{tip_position[0]:5.2f}, {tip_position[1]:5.2f}, {tip_position[2]:5.2f}] | "
                      f"Err: \033[1;93m{err_norm:5.2f}\033[0m | "
                      f"Z_in: {step_z_inner:5.3f} | Z_out: {step_z_outer:5.3f} | "
                      f"Rot: [{np.rad2deg(step_rot_inner):4.1f}°, {np.rad2deg(step_rot_outer):4.1f}°]")

            # 9. Convergence check
            if err_norm < 0.2:
                Sofa.Helper.msg_info(self, "3D TARGET REACHED WITH DLS PID")
                self.active = False
                self.update_live_plot()
                self.finalize_plot()
                
        except Exception as e:
            print(f"[PID Controller Error] {e}")
            self.active = False

    # =========================================================================
    # REAL-TIME PLOTTING METHODS
    # =========================================================================

    def setup_live_plot(self):
        """Initializes the matplotlib interactive mode and layout."""
        plt.ion()  # Turn on interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        self.fig.suptitle("PID Controller LIVE Dashboard", fontsize=14, fontweight='bold')
        
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
                save_path = os.path.join(images_dir, 'PID_Final_Plot.png')
                self.fig.savefig(save_path)
                print(f"\033[92m[PID] Final plot saved in {save_path}\033[0m")
            except Exception as e:
                print(f"\033[91m[PID] Failed to save plot: {e}\033[0m")
