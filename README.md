# Concentric Tube Robot (CTR) SOFA Simulation

This repository contains a comprehensive simulation environment for a 2-instrument Concentric Tube Robot (CTR) developed using the [SOFA framework](https://www.sofa-framework.org/). It provides a realistic, physics-based simulation of the CTR interacting with soft tissue (a liver model) and features a variety of advanced control strategies.

## Project Structure

- **`2instruments.py`**: The main simulation script. It constructs the SOFA scene, configures the two concentric tubes (Catheter and Guide), sets up collision models, integrates a soft tissue liver model, and initializes the selected controller.
- **`src/`**: Contains the core logic and controllers:
  - **`Teleoperation.py`**: Implements a Keyboard Controller for manual, teleoperated manipulation of the tubes.
  - **`PIDController.py`**: A PID position controller utilizing inverse kinematics (with Jacobian approximation) and null-space projection for target tracking.
  - **`MPCController.py`**: Implements Model Predictive Control (MPC) to handle position control while optimizing control effort and ensuring smooth movements.
  - **`AdaptiveController.py`**: Contains advanced adaptive controllers, including Model Reference Adaptive Control (MRAC) and a MIMO MCS Adaptive Controller, designed to handle nonlinearities and uncertain dynamics.
  - **`Liver.py`**: Handles the generation of a volumetric mesh of a liver from a surface mesh, represented as an elastic object for realistic interaction with the CTR.
  - **`workspace_generator.py`**: A feedforward controller that systematically explores the configuration space to sample and compute the CTR's reachable workspace.
  - **`setup.py`**: Utility functions to load SOFA plugins and tube parameters from configuration files.
  - **`plot.py` / `planner.py`**: Utilities for data visualization and path planning.
- **`config/`**: Contains configuration files such as `tube_parameters.json` which define physical properties (lengths, radii, curvature) for the tubes.
- **`mesh/`**: Contains 3D mesh files (e.g., `liver.obj`) used in the simulation environment.
- **`workspace/` / `Workspace images/` / `SOFA captures/`**: Directories for generated data, plots, and visual captures from the simulation.

## Features

- **Physics-Based Simulation**: Accurately models the kinematics and mechanics of concentric tube robots using SOFA's beam models (`AdaptiveBeamForceFieldAndMass`, `Edge2QuadTopologicalMapping`, etc.).
- **Soft Tissue Interaction**: Includes a volumetric liver model with adjustable stiffness (Young's modulus) to simulate contact and interactions.
- **Multiple Control Strategies**: Easily switch between Teleoperation, PID, MPC, MRAC, and MCS controllers to evaluate and compare tracking performance.
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
