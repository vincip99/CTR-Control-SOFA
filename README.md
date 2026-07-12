# Concentric Tube Robot (CTR) SOFA Simulation

This repository contains a comprehensive simulation environment for a 2-instrument Concentric Tube Robot (CTR) developed using the [SOFA framework](https://www.sofa-framework.org/). It provides a realistic, physics-based simulation of the CTR interacting with soft tissue (a liver model) and features a variety of advanced control strategies.

## Project Structure

- **`2instruments.py`**: The main simulation script. It constructs the SOFA scene, configures the two concentric tubes (Catheter and Guide), sets up collision models, integrates a soft tissue liver model, and initializes the selected controller.
- **`src/`**: Contains the core logic and controllers:
  - **`Teleoperation.py`**: Implements a Keyboard Controller for manual, teleoperated manipulation of the tubes.
  - **`PIDController.py`**: A sophisticated 3D position controller utilizing a Damped Least Squares (DLS) analytical Jacobian. It employs Null-Space Projection for secondary tasks including telescopic coordination (Gap Controller) and S-Divergence Snapping Avoidance to prevent mechanical instability. Includes a real-time tracking dashboard.
  - **`MPCController.py`**: Implements a 4-DOF Model Predictive Control (MPC) using SLSQP optimization. It coordinates independent tube deployment, handles collision forces via explicit inequality constraints, and visualizes system states in a live dashboard.
  - **`AdaptiveController.py`**: Contains advanced controllers, including a 4-DOF MIMO MCS Adaptive Controller utilizing relative insertion state representations to safely decouple and control the highly nonlinear tube dynamics.
  - **`Liver.py`**: Handles the generation of a volumetric mesh of a liver from a surface mesh, represented as an elastic object for realistic interaction with the CTR.
  - **`workspace_generator.py`**: A feedforward controller that systematically explores the configuration space using physics-aware sampling to compute the CTR's reachable workspace.
  - **`setup.py`**: Utility functions to load SOFA plugins and tube parameters from configuration files.
  - **`plot.py` / `planner.py`**: Utilities for data visualization and path planning.
- **`config/`**: Contains configuration files such as `tube_parameters.json` which define physical properties (lengths, radii, curvature) for the tubes.
- **`mesh/`**: Contains 3D mesh files (e.g., `liver.obj`) used in the simulation environment.
- **`workspace/` / `Workspace images/` / `SOFA captures/` / `images/`**: Directories for generated data, plots, and visual captures from the simulation.

## Features

- **Physics-Based Simulation**: Accurately models the kinematics and mechanics of concentric tube robots using SOFA's beam models (`AdaptiveBeamForceFieldAndMass`, `Edge2QuadTopologicalMapping`, etc.).
- **Soft Tissue Interaction**: Includes a volumetric liver model with adjustable stiffness (Young's modulus) to simulate contact and interactions.
- **Advanced Control Strategies**: Evaluate and compare tracking performance using Teleoperation, PID (with snapping avoidance and null-space projection), MPC (with force constraints), MRAC, and MCS controllers.
- **Real-Time Dashboards**: Live matplotlib-based dashboards integrated into the SOFA simulation loop for monitoring target tracking, errors, and actuator inputs.
- **Workspace Generation**: Automated scripts to compute and visualize the reachable workspace of the robot configuration.

## Setup and Usage

### Prerequisites
- [SOFA Framework](https://www.sofa-framework.org/) (v25.06 or similar compatible version) with Python 3 support enabled.
- Required Python packages: `numpy`, `scipy`, `matplotlib`

### Running the Simulation

You can launch the simulation using the `runSofa` executable provided by your SOFA installation:

```bash
/path/to/SOFA/bin/runSofa 2instruments.py
```

### Controls (Keyboard Teleoperation)
When the Keyboard Controller is active, the following controls can be used to manually drive the CTR:
- **`Ctrl + 1/2`**: Select which tube to control (1 = Catheter, 2 = Guide).
- **`Ctrl + K/J`**: Translate (insert/retract) the selected tube.
- **`Ctrl + M/N`**: Rotate the selected tube clockwise/counter-clockwise.

## Configuration

Physical parameters of the tubes can be modified in `config/tube_parameters.json`. This includes:
- `Straight_length`
- `Curved_length`
- `Tube_radius`
- `Radius_curvature`

## License

*(Add your license information here)*
