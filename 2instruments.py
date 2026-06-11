"""
Docstring for v25.06.00.Techical project.2instruments

File defining a SOFA scene with a 2-instrument CTR, including haptic and keyboard controllers.
"""
# import sys
import os
import Sofa
from src.Teleoperation import KeyBoardController
from src.PIDController import PIDPositionController
from src.AdaptiveController import MRACPositionController, MCSAdaptiveController
from src.MPCController import MPCPositionController
from src.Liver import create_liver
from src.setup import load_plugin_list, load_tube_parameters
import numpy as np

MESH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mesh')
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')

# Simulate the interaction of the CTR with stiffer (55e6 Pa) and soft (2e5 Pa) object and plot the force values online.
Liver_young_modulus_rigid = "55"
Liver_young_modulus_soft = "0.2" # default: 2e5


# Modify the parameter of tube stiffness (tube radius and young modulus of a chosen tube) to reach a value of contact force of 1.5 N          
# defualt: 1e9

def createScene(rootNode):
    # Load plugins from json file
    required_plugins = load_plugin_list(CONFIG_PATH)
    rootNode.addObject("RequiredPlugin", pluginName=" ".join(required_plugins))
    Sofa.Helper.msg_info("createScene", f"\033[1;92mPlugins loaded successfully\033[0;0m")

    # Get tubes parameters (dictionary)
    params = load_tube_parameters(CONFIG_PATH)
    Straight_length_1, Straight_length_2 = params["Straight_length"]
    Curved_length_1,   Curved_length_2   = params["Curved_length"]
    Tube_radius_1,     Tube_radius_2     = params["Tube_radius"]
    Radius_curvature_1,Radius_curvature_2= params["Radius_curvature"]
    Tube1_young_modulus,Tube2_young_modulus= params["Young_modulus"]
    Sofa.Helper.msg_info("createScene", f"\033[1;92mParameters loaded successfully\033[0;0m")

    # Set the scene
    rootNode.addObject('VisualStyle', displayFlags='showVisualModels showBehaviorModels showCollisionModels ' \
                        'hideBoundingCollisionModels hideForceFields')
    rootNode.addObject('FreeMotionAnimationLoop')
    rootNode.addObject('DefaultVisualManagerLoop')
    rootNode.gravity = [0, 0, 0]
    rootNode.dt = 0.01

    # --- SCELTA INTERATTIVA DEL LIVER --- Se voglio fare il workspace non voglio che il liver sia di intralcio
    try:
        user_input = input("Vuoi creare il Liver? [y/n] : ").strip().lower()
        LOAD_LIVER = (user_input == 'y')
    except:
        LOAD_LIVER = True  # fallback se input non è disponibile
    if LOAD_LIVER:
        create_liver(rootNode=rootNode, ym=Liver_young_modulus_soft, 
                    path=MESH_PATH) #, fixingBox=[20, 0, 90, 30, 15, 120])
    else:
        Sofa.Helper.msg_info("createScene", "\033[1;91mLiver NON creato\033[0;0m")
    

    # Collision pipeline
    solver = rootNode.addObject('GenericConstraintSolver', name='solver', computeConstraintForces="1", 
                                tolerance='1e-6', maxIterations='1000')
    rootNode.addObject('CollisionPipeline', verbose='0')
    rootNode.addObject('ParallelBruteForceBroadPhase')
    rootNode.addObject('ParallelBVHNarrowPhase')
    rootNode.addObject('CollisionResponse', name='response', response='FrictionContactConstraint')
    rootNode.addObject('LocalMinDistance', name='proximity', alarmDistance='2', contactDistance='0.01', angleCone='0.0')


    # CATHETER
    TUBE_1 = rootNode.addChild('TUBE_1', bbox='-3 -6 -3 3 3 3')
    TUBE_1.addObject('RodStraightSection', name='StraightSection', 
                     length = Straight_length_1, radius = Tube_radius_1, 
                     youngModulus=Tube1_young_modulus, massDensity=1.55e-6, nbBeams=40, nbEdgesCollis=40, nbEdgesVisu=80)
    TUBE_1.addObject('RodSpireSection', name='SpireSection', 
                     length = Curved_length_1, spireDiameter = 2*Radius_curvature_1, spireHeight=0.0, 
                     youngModulus=Tube1_young_modulus, massDensity=1.55e-6, nbBeams=40, nbEdgesCollis=40, nbEdgesVisu=80)
    TUBE_1.addObject('WireRestShape', template='Rigid3d', name='RestShape_1', wireMaterials='@StraightSection @SpireSection')
    # Mechanical properties
    TUBE_1.addObject('EdgeSetTopologyContainer', name='meshLines_1')
    TUBE_1.addObject('EdgeSetTopologyModifier', name='Modifier')
    TUBE_1.addObject('EdgeSetGeometryAlgorithms', name='GeomAlgo', template='Rigid3d')
    TUBE_1.addObject('MechanicalObject', template='Rigid3d', name='dofTopo_1')


    # GUIDE
    TUBE_2 = rootNode.addChild('TUBE_2', bbox='-3 -6 -3 3 3 3')
    TUBE_2.addObject('RodStraightSection', name='StraightSection', 
                     length = Straight_length_2, radius = Tube_radius_2, 
                     youngModulus=Tube2_young_modulus, massDensity=1.55e-6, nbBeams=40, nbEdgesCollis=40, nbEdgesVisu=80)
    TUBE_2.addObject('RodSpireSection', name='SpireSection', 
                     length = Curved_length_2, spireDiameter = 2*Radius_curvature_2, spireHeight=0.0, 
                     youngModulus=Tube2_young_modulus, massDensity=1.55e-6, nbBeams=40, nbEdgesCollis=40, nbEdgesVisu=80)
    TUBE_2.addObject('WireRestShape', template='Rigid3d', name='RestShape_2', wireMaterials='@StraightSection @SpireSection')
    # Mechanical properties
    TUBE_2.addObject('EdgeSetTopologyContainer', name='meshLines_2')
    TUBE_2.addObject('EdgeSetTopologyModifier', name='Modifier')
    TUBE_2.addObject('EdgeSetGeometryAlgorithms', name='GeomAlgo', template='Rigid3d')
    TUBE_2.addObject('MechanicalObject', template='Rigid3d', name='dofTopo_2')


    # INSTRUMENT COMBINED
    CTR = rootNode.addChild('CTR')
    CTR.addObject('EulerImplicitSolver', rayleighStiffness='0.2', rayleighMass='0.1', printLog='false')
    CTR.addObject('BTDLinearSolver')
    CTR.addObject('RegularGridTopology', name='meshLinesCombined', 
                  nx='241', ny='1', nz='1', xmin='0.0', xmax='1.0', ymin='0', ymax='0', zmin='1', zmax='1')
    CTR.addObject('MechanicalObject', template='Rigid3d', name='DOFs', showIndices='0', ry='-90')
    CTR.addObject('WireBeamInterpolation', name='Interpol_1', WireRestShape='@../TUBE_1/RestShape_1')
    CTR.addObject('AdaptiveBeamForceFieldAndMass', name='Tube1ForceField', interpolation='@Interpol_1')
    CTR.addObject('WireBeamInterpolation', name='Interpol_2', WireRestShape='@../TUBE_2/RestShape_2')
    CTR.addObject('AdaptiveBeamForceFieldAndMass', name='Tube2ForceField', interpolation='@Interpol_2')
    CTR.addObject('InterventionalRadiologyController', template="Rigid3d", name="IRController", 
                  instruments="Interpol_1 Interpol_2 ", xtip="1 0", step="3", 
                  rotationInstrument="0 0", controlledInstrument="0", startingPos="0 0 0 0 0 0 1")
    CTR.addObject('RestShapeSpringsForceField', points="@IRController.indexFirstNode", 
                  stiffness="1e8", angularStiffness="1e8")
    CTR.addObject('FixedProjectiveConstraint', name='FixedConstraint', indices='0')
    CTR.addObject('LinearSolverConstraintCorrection', wire_optimization='true')


    # Collisions
    collis = CTR.addChild('Collis')
    collis.addObject('EdgeSetTopologyContainer', name='collisEdgeSet')
    collis.addObject('EdgeSetTopologyModifier', name='colliseEdgeModifier')
    collis.addObject('MechanicalObject', name='MechanicalObject', template='Vec3d')
    collis.addObject('MultiAdaptiveBeamMapping', controller='../IRController')
    collis.addObject('SphereCollisionModel', name='SCM', radius=Tube_radius_2)


    # Visual Properties CATHETER
    visu_1 = CTR.addChild('visu_1', activated='true')
    visu_1.addObject('MechanicalObject', name='Quads')
    visu_1.addObject('QuadSetTopologyContainer', name='Container_1')
    visu_1.addObject('QuadSetTopologyModifier', name='Modifier')
    visu_1.addObject('QuadSetGeometryAlgorithms', name='GeomAlgo', template='Vec3d')
    visu_1.addObject('Edge2QuadTopologicalMapping', nbPointsOnEachCircle='10', 
                     radius=Tube_radius_1, input='@../../TUBE_1/meshLines_1', output='@Container_1', flipNormals='true')
    visu_1.addObject('AdaptiveBeamMapping',  name='VisuMap_1', useCurvAbs='1', printLog='0', 
                     interpolation='@../Interpol_1', input='@../DOFs', output='@Quads')
    visuOgl_1 = visu_1.addChild('visuOgl_1', activated='true')
    visuOgl_1.addObject('OglModel', name='Visual', color='0.5 0.5 0.5 1', quads="@../Container_1.quads")
    visuOgl_1.addObject('IdentityMapping', input="@../Quads", output="@Visual")

    # Visual Properties GUIDE
    visu_2 = CTR.addChild('visu_2', activated='true')
    visu_2.addObject('MechanicalObject', name='Quads')
    visu_2.addObject('QuadSetTopologyContainer', name='Container_2')
    visu_2.addObject('QuadSetTopologyModifier', name='Modifier')
    visu_2.addObject('QuadSetGeometryAlgorithms', name='GeomAlgo', template='Vec3d')
    visu_2.addObject('Edge2QuadTopologicalMapping', nbPointsOnEachCircle='10', 
                     radius=Tube_radius_2, input='@../../TUBE_2/meshLines_2', output='@Container_2', flipNormals='true')
    visu_2.addObject('AdaptiveBeamMapping',  name='VisuMap_2', useCurvAbs='1', printLog='0', 
                     interpolation='@../Interpol_2', input='@../DOFs', output='@Quads')
    visuOgl_2 = visu_2.addChild('visuOgl_2', activated='true')
    visuOgl_2.addObject('OglModel', name='Visual', color='0.3 0.7 1 1', quads="@../Container_2.quads")
    visuOgl_2.addObject('IdentityMapping', input="@../Quads", output="@Visual")

    # KEYBOARD CONTROLLER
    rootNode.addObject(KeyBoardController(
        name="KeyBoardController",
        irController=CTR.getObject('IRController'),
        rootNode=rootNode
    ))

    Sofa.Helper.msg_info("createScene", """\033[93mKeyboard Controller \033[1;92mAttivo\033[0;93m
    Comandi:
    \033[1;94mCtrl + 1/2\033[0;93m per selezionare quale tubo controllare; 
    \033[1;94mCtrl + K/J\033[0;93m per traslare in avanti o indietro il tubo selezionato;
    \033[1;94mCtrl + M/N\033[0;93m per ruotare in verso orario o antiorario il tubo selezionato;\033[0m""")

    # PID CONTROLLER (IK + Null-Space)
    try:
        rootNode.addObject(PIDPositionController(
            name="PIDPositionController",
            rootNode=rootNode, 
            irController=CTR.getObject('IRController'), 
            target=[-2.0, 4.0, 45.5],
            Kp=5.0,
            Ki=1.0,
            Kd=0.0
        ))
        Sofa.Helper.msg_info("createScene", "\033[1;92mPID IK Controller initialized successfully\033[0;0m")
    except Exception as e:
        Sofa.Helper.msg_info("createScene", f"\033[1;91mPID Controller failed to initialize: {e}\033[0;0m")
    # MRAC POSITION CONTROLLER
    try:
        rootNode.addObject(MRACPositionController(
            name="MRACController",
            rootNode=rootNode, 
            irController=CTR.getObject('IRController'), 
            target=[4.49, 2.0, 46.5], #[1.49, -8.0, 46.5],
            stop_at_z=15.0,
            Am=0.5,              # Reference model tracking speed
            Bm=0.5,              # Reference model input gain
            gamma_trans=0.02,    # Adaptation rate for translation (Z)
            gamma_rot=0.01      # Adaptation rate for rotation (X,Y)
        ))
        Sofa.Helper.msg_info("createScene", "\033[1;92mMRAC Controller initialized successfully\033[0;0m")
    except Exception as e:
        Sofa.Helper.msg_info("createScene", f"\033[1;91mMRAC Controller failed to initialize: {e}\033[0;0m")

        # MIMO MCS ADAPTIVE CONTROLLER
    try:
        rootNode.addObject(MCSAdaptiveController(
            name="MCSController",
            rootNode=rootNode,
            irController=CTR.getObject('IRController'),
            target=[4.49, 2.0, 46.5],
            stop_at_z=25.0,
            alpha=0.005,         # Integral adaptation gain
            beta=0.001,          # Proportional adaptation gain
            sigma=0.05,          # Anti-windup leakage factor
            Am=-0.1,             # Slower Reference Model
            Bm=0.1
        ))
        Sofa.Helper.msg_info("createScene", "\033[1;92mMIMO-MCS Controller initialized successfully\033[0;0m")
    except Exception as e:
        Sofa.Helper.msg_info("createScene", f"\033[1;91mMIMO-MCS Controller failed to initialize: {e}\033[0;0m")
 
    # MPC CONTROLLER
    try:
        rootNode.addObject(MPCPositionController(
            name="MPCPositionController",
            rootNode=rootNode,
            irController=CTR.getObject('IRController'),
            target=[-2.0, 4.0, 45.5],
            stop_at_z=25.0,
            N=5
        ))
        Sofa.Helper.msg_info("createScene", "\033[1;92mMPC Controller initialized successfully (Press P to start)\033[0;0m")
    except Exception as e:
        Sofa.Helper.msg_info("createScene", f"\033[1;91mMPC Controller failed to initialize: {e}\033[0;0m")

    return rootNode


def main():
    import SofaRuntime
    import Sofa.Gui

    root = Sofa.Core.Node('root')
    createScene(root)
    Sofa.Simulation.init(root)
    Sofa.Gui.GUIManager.Init('myscene', 'qglviewer')
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1080, 1080)
    Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()


# Function used only if this script is called from a python environment
if __name__ == '__main__':
    main()
