"""
Docstring for v25.06.00.Techical project.InstrumentsController

File defining controllers for the CTR simulation, including keyboard and haptic device controllers.
"""
# import sys
import os
import Sofa
import Sofa.Core
# import Sofa.constants.Key as Key
import Sofa.Simulation
import csv
import numpy as np
import matplotlib.pyplot as plt
from .setup import load_tube_parameters, colored_tube_number, colored

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')

params = load_tube_parameters(CONFIG_PATH)
Straight_length_1, Straight_length_2 = params["Straight_length"]
Curved_length_1,   Curved_length_2  = params["Curved_length"]
Tube_radius_1,     Tube_radius_2     = params["Tube_radius"]
Radius_curvature_1,Radius_curvature_2= params["Radius_curvature"]
Sofa.Helper.msg_info("InstrmentsController", f"\033[1;92mParameters loaded successfully\033[0;0m")


# Classe del controller da tastiera
class KeyBoardController(Sofa.Core.Controller):
    # Costruttore
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.name = 'KeyBoardController'
        self.ir_controller = kwargs.get('irController')
        self.rootNode = kwargs.get('rootNode')

        self.tube = 1    # (1 = TUBE_1, 2 = TUBE_2)
        self.translation_step = 1.0 # [mm]
        self.rotation_step = np.radians(22.5) # [rad]

        # Compare the plotted workspace when varying the curvature of the innermost and medium tube of 12% of the nominal value.
        self.variazione_percentuale = 12.0 # in valore assoluto
        # Loggig data
        self.print_interval = 1
        self.last_print_time = 0.0   
        # --- ONLINE FORCE PLOT ---
        plt.ion()  # modalità interattiva
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Force vs Time")
        self.ax.set_xlabel("Tempo [s]")
        self.ax.set_ylabel("Forza [N]")
        self.ax.grid(True)
        
        self.times = []
        self.forces = []
        
        self.line, = self.ax.plot(self.times, self.forces, '-b')
        self.fig.show()
        self.fig.canvas.draw()
 

    # Keyboard contrller	
    def onKeypressedEvent(self, c):
        key = c['key']

        if key != 'U':
            Sofa.Helper.msg_info("KeyBoardController", f"Key pressed: {key}")

        if key == "1":
            self.tube = 1
            Sofa.Helper.msg_info("KeyBoardController", f"Tubo {colored_tube_number(self.tube)} selezionato")
            self.ir_controller.controlledInstrument.value = self.tube -1
        elif key == "2":
            self.tube = 2
            Sofa.Helper.msg_info("KeyBoardController", f"Tubo {colored_tube_number(self.tube)} selezionato")
            self.ir_controller.controlledInstrument.value = self.tube -1
        elif key == "J":   
            Sofa.Helper.msg_info("KeyBoardController", f"Tubo {colored_tube_number(self.tube)}: accorciamento di {colored(self.translation_step, 'yellow')} mm")
            self.translate(self.tube, - self.translation_step)		
        elif key == "K":   
            Sofa.Helper.msg_info("KeyBoardController", f"Tubo {colored_tube_number(self.tube)}: allungamento di {colored(self.translation_step, 'yellow')} mm")
            self.translate(self.tube, self.translation_step)
        elif key == "M":    
            Sofa.Helper.msg_info("KeyBoardController", f"Tubo {colored_tube_number(self.tube)}: Rotazione oraria di {colored(np.rad2deg(self.rotation_step), 'yellow')}°")
            self.rotate(self.tube, - self.rotation_step)		
        elif key == "N":   
            Sofa.Helper.msg_info("KeyBoardController", f"Tubo {colored_tube_number(self.tube)}: Rotazione antioraria di {colored(np.rad2deg(self.rotation_step), 'yellow')}°")
            self.rotate(self.tube, self.rotation_step)

    # Utility functions per il controller
    def translate(self, tube, quantity):
        with self.ir_controller.xtip.writeable() as d: d[tube-1] = d[tube-1] + quantity

    def rotate(self, tube, quantity):
        with self.ir_controller.rotationInstrument.writeable() as d: d[tube-1] = d[tube-1] + quantity

    def update_radius_curvature(self, percentage=100.0):
        # Modify the innermost and medium tube radius curvature
        percentage /= 100.0
        # Nominal values
        r2 = Radius_curvature_2
        # Update radius
        r2 *= percentage
        # Update the sofa diameter values
        self.rootNode.TUBE_2.SpireSection.spireDiameter.value = 2*r2
        # Reset rest shape
        self.rootNode.TUBE_2.RestShape_2.reinit()

    def onAnimateEndEvent(self, event):
         self.get_force_value()
         self.get_position_value()

    def get_position_value(self):
        try:
            # 1. Access the tip position (last node of the DOFs MechanicalObject)
            # Make sure 'CTR' and 'DOFs' names match exactly your scene graph
            tip_pose = self.rootNode.CTR.DOFs.position.value[-1]
            
            # 2. Extract X, Y, Z
            x, y, z = tip_pose[0], tip_pose[1], tip_pose[2]
            
            # 3. Log the coordinates
            # Using a hardcoded string "InstrumentsController" to avoid the TypeError
            Sofa.Helper.msg_info("InstrumentsController", 
                f"Tip Position -> \033[1;32mX: {x:8.3f}\033[0m | \033[1;33mY: {y:8.3f}\033[0m | \033[1;36mZ: {z:8.3f}\033[0m")
            
            return [x, y, z]
            
        except AttributeError:
            Sofa.Helper.msg_err("InstrumentsController", "Could not find rootNode.CTR.DOFs. Check scene graph names.")
            return [0, 0, 0]
        except Exception as e:
            Sofa.Helper.msg_err("InstrumentsController", f"Error in get_position_value: {str(e)}")
            return [0, 0, 0]
            
        
    def get_force_value(self):
        t = self.rootNode.time.value    
        solver = self.rootNode.getObject("solver")
        forces_vector = solver.constraintForces.value
        if forces_vector is not None and forces_vector.size > 0 and (t - self.last_print_time) > self.print_interval:
            force_value = np.linalg.norm(forces_vector)
            # force_value = forces_vector[0]*0.0001 # Solo la componente lungo X
            Sofa.Helper.msg_info("KeyBoardController", f"Sensore di forza: \033[1;93mF = {force_value:.3f} N\033[0;0m\n")
            self.last_print_time = t

            # --- ONLINE PLOT UPDATE ---
            self.times.append(t)
            self.forces.append(force_value)

            # Aggiorna la curva
            self.line.set_xdata(self.times)
            self.line.set_ydata(self.forces)

            # Aggiorna i limiti
            self.ax.relim()
            self.ax.autoscale_view()

            # Refresh del grafico
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()  
# end class
