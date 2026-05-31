import os
import csv
import math
import numpy as np
import Sofa.Core
import Sofa.Helper

class WorkspaceGeneratorController(Sofa.Core.Controller):
    """
    A SOFA Feedforward Controller that systematically explores the configuration space of the CTR
    and records the steady-state tip position at each point to compute the workspace.
    """
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.name = "WorkspaceGenerator"
        self.ir_controller = kwargs.get('irController')
        self.rootNode = kwargs.get('rootNode')
        
        # Ranges for Z axis (inner and outer tubes)
        self.outer_z_min = kwargs.get('outer_z_min', 0.0)
        self.outer_z_max = kwargs.get('outer_z_max', 40.0)
        self.outer_z_step = kwargs.get('outer_z_step', 10.0)
        
        self.inner_z_min = kwargs.get('inner_z_min', 0.0)
        self.inner_z_max = kwargs.get('inner_z_max', 50.0)
        self.inner_z_step = kwargs.get('inner_z_step', 10.0)
        
        # Rotation step for innner and outer tubes
        self.rot_step = kwargs.get('rot_step', math.radians(90)) # 90 degrees
        
        self.settle_steps = kwargs.get('settle_steps', 20) # Num SOFA frames to wait per config
        
        self.active = False
        self.configurations = []
        self.current_idx = 0
        self.settle_counter = 0
        self.recorded_data = []
        
        # Save path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.workspace_dir = os.path.join(base_dir, "workspace")
        if not os.path.exists(self.workspace_dir):
            os.makedirs(self.workspace_dir)

    def generate_configurations(self):
        """Generates all valid combinations of translations and rotations."""
        configs = []
        
        outer_z_vals = np.arange(self.outer_z_min, self.outer_z_max + 1e-5, self.outer_z_step)
        rot_vals = np.arange(0.0, 2*math.pi, self.rot_step)
        
        for z0 in outer_z_vals:
            # Inner tube must be at least as inserted as outer tube, and respect its own min limit
            start_inner = max(z0, self.inner_z_min)
            if start_inner <= self.inner_z_max:
                inner_z_vals = np.arange(start_inner, self.inner_z_max + 1e-5, self.inner_z_step)
            else:
                inner_z_vals = []
                
            for z1 in inner_z_vals:
                for rot0 in rot_vals:
                    for rot1 in rot_vals:
                        configs.append((z0, z1, rot0, rot1))
                        
        return configs

    def onKeypressedEvent(self, c):
        if str(c['key']).upper() == "W" and not self.active: # Using W as hotkey
            self.configurations = self.generate_configurations()
            self.recorded_data = []
            self.current_idx = 0
            self.settle_counter = 0
            
            if len(self.configurations) > 0:
                self.active = True
                print("\n" + "="*70)
                Sofa.Helper.msg_info(self, f"\033[1;94m[Workspace Generator] STARTED. Total configs: {len(self.configurations)}\033[0m")
                print("="*70 + "\n")
            else:
                Sofa.Helper.msg_err(self, "No configurations generated. Check your limits.")

    def onAnimateBeginEvent(self, event):
        if not self.active:
            return

        # Initialize internal virtual state on first run to avoid read-back issues (like angle wrapping)
        if not hasattr(self, 'current_virtual_state'):
            xtip = self.ir_controller.xtip.value
            rot = self.ir_controller.rotationInstrument.value
            self.current_virtual_state = [xtip[0], xtip[1], rot[0], rot[1]]

        target_z0, target_z1, target_rot0, target_rot1 = self.configurations[self.current_idx]
        
        cz0, cz1, cr0, cr1 = self.current_virtual_state
            
        diff_z0 = target_z0 - cz0
        diff_z1 = target_z1 - cz1
        diff_rot0 = target_rot0 - cr0
        diff_rot1 = target_rot1 - cr1
        
        # Safe step sizes per frame to prevent physics explosion
        max_trans_step = 1.0  # mm per frame (increased slightly for speed)
        max_rot_step = math.radians(5.0)  # rad per frame
        
        is_at_target = (abs(diff_z0) < 1e-3 and abs(diff_z1) < 1e-3 and 
                        abs(diff_rot0) < 1e-3 and abs(diff_rot1) < 1e-3)

        if not is_at_target:
            # 1. Transition towards target configuration safely
            step_z0 = np.clip(diff_z0, -max_trans_step, max_trans_step)
            step_z1 = np.clip(diff_z1, -max_trans_step, max_trans_step)
            step_rot0 = np.clip(diff_rot0, -max_rot_step, max_rot_step)
            step_rot1 = np.clip(diff_rot1, -max_rot_step, max_rot_step)
            
            self.current_virtual_state[0] += step_z0
            self.current_virtual_state[1] += step_z1
            self.current_virtual_state[2] += step_rot0
            self.current_virtual_state[3] += step_rot1
            
            with self.ir_controller.xtip.writeable() as xtip:
                xtip[0] = self.current_virtual_state[0]
                xtip[1] = self.current_virtual_state[1]
            with self.ir_controller.rotationInstrument.writeable() as rot:
                rot[0] = self.current_virtual_state[2]
                rot[1] = self.current_virtual_state[3]
                
            self.settle_counter = 0 # Reset settling because we are moving
            
            # Occasional debug print to show it's actually moving
            if getattr(self, 'print_counter', 0) % 60 == 0:
                print(f"Transitioning to config {self.current_idx}: {self.current_virtual_state}")
            self.print_counter = getattr(self, 'print_counter', 0) + 1
            
        elif self.settle_counter < self.settle_steps:
            # 2. Reached target, now wait for simulation to settle
            self.settle_counter += 1
            
        else:
            # 3. Settled. Record Data and Advance
            try:
                # Get the tip position
                tip_pose = self.rootNode.CTR.DOFs.position.value[-1]
                x, y, z = tip_pose[0], tip_pose[1], tip_pose[2]
                
                self.recorded_data.append({
                    "X": x,
                    "Y": y,
                    "Z": z,
                    "q_z_outer": target_z0,
                    "q_z_inner": target_z1,
                    "q_rot_outer": math.degrees(target_rot0),
                    "q_rot_inner": math.degrees(target_rot1),
                    "Ins3": 1
                })
                
                # Print progress
                if self.current_idx % 20 == 0:
                    progress = (self.current_idx / len(self.configurations)) * 100
                    print(f"Workspace Generation: {progress:.1f}% ({self.current_idx}/{len(self.configurations)})")
                    
                self.current_idx += 1
                self.settle_counter = 0 # Reset for next config
                
                # Check completion
                if self.current_idx >= len(self.configurations):
                    self.finish_and_save()
                    
            except Exception as e:
                print(f"[Workspace Generator Error] {e}")
                self.active = False

    def finish_and_save(self):
        """Saves the recorded data to a CSV and stops the controller."""
        self.active = False
        
        filepath = os.path.join(self.workspace_dir, "workspace_nominal.csv")
        
        # Write to CSV
        fieldnames = ["X", "Y", "Z", "q_z_outer", "q_z_inner", "q_rot_outer", "q_rot_inner", "Ins3"]
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in self.recorded_data:
                    writer.writerow(row)
                    
            print("\n" + "="*70)
            Sofa.Helper.msg_info(self, f"\033[1;92m[SUCCESS] Workspace saved to {filepath}!\033[0m")
            print("="*70 + "\n")
            
        except Exception as e:
            Sofa.Helper.msg_err(self, f"Failed to save workspace: {e}")
