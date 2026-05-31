import numpy as np

class MinimumJerkPlanner:
    """Generates a smooth, 0-acceleration start/stop trajectory in 3D space."""
    def __init__(self, start_pos, target_pos, speed=2.0):
        self.p0 = np.array(start_pos, dtype=float)
        self.pf = np.array(target_pos, dtype=float)
        distance = np.linalg.norm(self.pf - self.p0)
        
        # Prevent division by zero if target is already reached
        self.T = distance / speed if speed > 0 and distance > 0 else 0.0
        self.t = 0.0

    def step(self, dt):
        if self.T <= 0.0:
            return self.pf, True

        self.t = np.clip(self.t + dt, 0.0, self.T)
        tau = self.t / self.T
        
        # Minimum Jerk Quintic Polynomial: 10t^3 - 15t^4 + 6t^5
        scale = 10 * (tau**3) - 15 * (tau**4) + 6 * (tau**5)
        
        current_pos = self.p0 + (self.pf - self.p0) * scale
        is_finished = (self.t >= self.T)
        
        return current_pos, is_finished