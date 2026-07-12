"""
Docstring for v25.06.00.Techical project.Liver

File to create a volumetric mesh of the liver from a surface mesh as an elastic object.
"""
import Sofa
import os

# Starting from the surface mesh assigned, create a volumetric mesh using the python script provided. 
# Generate an elastic object from the volumetric mesh and add in the simulation scene (Use fixed boxes to constrain the object to a fixed position)
def create_liver(rootNode=None, ym = 55e6, path=None,
                 pos = [0, -10, 10], rot = [45, 45, 0], scale = 10, 
                 fixingBox = [10, 0, 30, 20, 15, 40], mass = 100.0, color = [0.7, 0.3, 0.1]):

    liver = rootNode.addChild('liver')
    liver.addObject('EulerImplicitSolver', rayleighStiffness="0.1", rayleighMass="0.1")	
    liver.addObject('SparseLDLSolver')
    
    liver.addObject('MeshOBJLoader', name="liverLoader", filename=os.path.join(path, 'liver.obj'), 
                    translation=pos, rotation=rot, scale=scale)	
    liver.addObject('MeshGenerationFromPolyhedron', name="tetraGenerator", inputPoints="@liverLoader.position", 
                    inputTriangles="@liverLoader.triangles", inputQuads="@liverLoader.quads", drawTetras="0", 
                    facetSize="5", facetApproximation="1", cellRatio="2", cellSize="5")	
    liver.addObject('MechanicalObject', name="dofs", position="@tetraGenerator.outputPoints")	
    liver.addObject('TetrahedronSetTopologyContainer', name="topo", tetrahedra="@tetraGenerator.outputTetras")
    liver.addObject('TetrahedronSetGeometryAlgorithms', template="Vec3d", name="GeomAlgo")
    liver.addObject('ParallelTetrahedronFEMForceField', name="FEM", youngModulus=ym, poissonRatio="0.3", method="large") 
    liver.addObject('UniformMass', name="mass", totalMass=mass)
    liver.addObject('LinearSolverConstraintCorrection')

    liverVisu = liver.addChild('Visu')											      
    liverVisu.addObject('OglModel', name="Visual", src="@../liverLoader", color=color)
    liverVisu.addObject('BarycentricMapping', input="@..", output="@Visual")

    liverCollis = liver.addChild('Collision')								    
    liverCollis.addObject('MeshTopology', src="@../liverLoader")
    # Independent collision models
    liverCollis.addObject('MechanicalObject')	
    liverCollis.addObject('TriangleCollisionModel')
    liverCollis.addObject('LineCollisionModel')
    liverCollis.addObject('PointCollisionModel')
    # Bridge Collision and FEM Mesh
    liverCollis.addObject('BarycentricMapping', input="@..", output="@.")

    constraintNode = liver.addChild('Constraints')
    # Select vertices inside the bounding box						
    constraintNode.addObject('BoxROI', name="fixedBox", box=fixingBox, drawBoxes="1", position="@../dofs.rest_position")
    # Lock the selected vertices
    constraintNode.addObject('FixedProjectiveConstraint', indices="@fixedBox.indices")
    # Additional fixed boxes to keep the liver in position
    constraintNode.addObject('BoxROI', name="fixedBox2", box=[-30, 0, 30, -20, 15, 40], drawBoxes="1", position="@../dofs.rest_position")
    constraintNode.addObject('FixedProjectiveConstraint', indices="@fixedBox2.indices")
    constraintNode.addObject('BoxROI', name="fixedBox3", box=[0, -30, 40, -15, -15, 60], drawBoxes="1", position="@../dofs.rest_position")
    constraintNode.addObject('FixedProjectiveConstraint', indices="@fixedBox3.indices")

    liver.addObject('VTKExporter', 
                        filename=os.path.join(path, 'liver.vtu'), 
                        listening=True,
                        exportAtBegin=True,
                        edges="0", 
                        triangles="0", 
                        quads="0",
                        tetras="1", 
                        hexas="0", 
                        printLog="1")
    
    Sofa.Helper.msg_info("createScene", "\033[1;92mLiver Correctly Generated\033[0;0m")