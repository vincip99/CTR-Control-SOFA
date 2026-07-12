import os
import sys
import importlib.util
import Sofa.Core
import Sofa.Simulation
import Sofa.Gui

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

spec = importlib.util.spec_from_file_location("instruments2", os.path.join(project_dir, "2instruments.py"))
instruments2 = importlib.util.module_from_spec(spec)
sys.modules["instruments2"] = instruments2
spec.loader.exec_module(instruments2)

def createScene(root):
    # 1. Load your original scene normally
    instruments2.createScene(root)
    
    # 2. Upgrade the Visual Aesthetics!
    # By default, SOFA shows collision and behavior models (the ugly blue dots and red lines).
    # We turn those off here so ONLY the beautiful 3D Visual Models are rendered.
    visual_style = root.getObject("VisualStyle")
    if visual_style:
        visual_style.displayFlags = "showVisualModels hideBehaviorModels hideCollisionModels hideBoundingCollisionModels hideForceFields"
    
    # Add a clean, publication-ready background (a bit more grey as requested)
    root.addObject('BackgroundSetting', color='0.8 0.8 0.8 1')
    
    # Add a bright, 3-point lighting setup to properly illuminate the CTR and Liver from all angles
    root.addObject('LightManager', ambient="0.5 0.5 0.5 1")  # Increased ambient light so shadows aren't too dark
    root.addObject('DirectionalLight', direction="0 -1 -1", color="0.9 0.9 0.9")  # Main key light from top-front
    root.addObject('DirectionalLight', direction="1 1 0", color="0.6 0.6 0.6")    # Fill light from the right
    root.addObject('DirectionalLight', direction="-1 0 1", color="0.5 0.5 0.5")   # Back light from the left
    
    
    return root


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

if __name__ == '__main__':
    main()
