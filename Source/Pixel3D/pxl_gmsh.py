

import os
import time
import vtk
import gmsh


def read_vtp_file(file_path):
    """Reads a .vtp file and returns vtkPolyData."""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    polydata = reader.GetOutput()
    if not polydata:
        raise ValueError("The file does not contain vtkPolyData.")
    return polydata


def vtk_to_gmsh(polydata):
    """Converts vtkPolyData to a format usable by GMSH."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # Silence output
    gmsh.model.add("VTK_to_GMSH")

    # Add points to GMSH
    node_tags = []
    for i in range(polydata.GetNumberOfPoints()):
        x, y, z = polydata.GetPoint(i)
        tag = gmsh.model.geo.addPoint(x, y, z)
        node_tags.append(tag)

    # Add curves and surfaces
    for i in range(polydata.GetNumberOfCells()):
        cell = polydata.GetCell(i)
        num_points = cell.GetNumberOfPoints()
        if num_points < 3:
            continue  # Skip cells with fewer than 3 points

        # Create closed curves
        curve_tags = []
        for j in range(num_points):
            start_point = cell.GetPointId(j) + 1  # GMSH uses 1-based indices
            end_point = cell.GetPointId((j + 1) % num_points) + 1
            curve_tags.append(gmsh.model.geo.addLine(start_point, end_point))

        # Create a closed loop
        curve_loop = gmsh.model.geo.addCurveLoop(curve_tags)

        # Create a planar surface
        gmsh.model.geo.addPlaneSurface([curve_loop])

    # Synchronize geometry with GMSH
    gmsh.model.geo.synchronize()
    return gmsh


def remesh_gmsh(target_size=0.5, algorithm=5, optimize=False):
    """Applies mesh regularization in GMSH."""
    gmsh.option.setNumber("Mesh.Algorithm", algorithm)  # Change algorithm (5 = Delaunay)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", target_size * 0.2)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", target_size)

    # gmsh.option.setNumber("Mesh.MinimumCirclePoints", 4)  # Default is 6 or higher
    # gmsh.option.setNumber("Mesh.MinimumCurvePoints", 4)   # Default is 10 or higher
    # gmsh.option.setNumber("Mesh.ElementOrder", 1)  # Linear elements
    # gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Frontal-Delaunay
    
    if optimize:
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)  # Enable optimization
    else:
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)  # Disable optimization

    start_time_1 = time.time()
    
    gmsh.model.mesh.generate(3)
    
    lapse = round(time.time() - start_time_1, 3)
    print("mgsh regularization lapse:", lapse)


def gmsh_to_vtk():
    """Converts the GMSH mesh to vtkPolyData."""
    nodes = gmsh.model.mesh.getNodes()
    elements = gmsh.model.mesh.getElementsByType(2)  # Type 2 = triangles

    # Create vtkPolyData
    points = vtk.vtkPoints()
    for i in range(0, len(nodes[1]), 3):
        points.InsertNextPoint(nodes[1][i], nodes[1][i + 1], nodes[1][i + 2])

    polys = vtk.vtkCellArray()
    for i in range(0, len(elements[1]), 3):
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, elements[1][i] - 1)  # GMSH uses 1-based indices
        triangle.GetPointIds().SetId(1, elements[1][i + 1] - 1)
        triangle.GetPointIds().SetId(2, elements[1][i + 2] - 1)
        polys.InsertNextCell(triangle)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(polys)
    
    n_cells = polydata.GetNumberOfCells()
    n_points = polydata.GetNumberOfPoints()
    print("cells, points:", n_cells, n_points)
    
    # ------------------------------

    if False:
        # Compute reduction factor
        num_cells = polydata.GetNumberOfCells()
        target_cells = 1000
        reduction = max(0.0, min(1.0, 1.0 - (target_cells / num_cells)))
        print(f"Target reduction: {reduction:.2%}")
        preserve_topology=True

        # Apply the DecimatePro filter
        decimate = vtk.vtkDecimatePro()
        decimate.SetInputData(polydata)
        decimate.SetTargetReduction(reduction)  # Fraction of cells to remove
        decimate.PreserveTopologyOn() if preserve_topology else decimate.PreserveTopologyOff()
        decimate.BoundaryVertexDeletionOff()  # Prevents deleting boundary vertices
        decimate.Update()
        
        _polydata:vtk.vtkPolyData = vtk.vtkPolyData()
        _polydata.DeepCopy(decimate.GetOutput())
        
        n_cells = _polydata.GetNumberOfCells()
        n_points = _polydata.GetNumberOfPoints()
        print("reduced cells, points:", n_cells, n_points)

        gmsh.model.remove()  # Clean the current model after processing
        return _polydata
    
    # ------------------------------    
    
    gmsh.model.remove()  # Clean the current model after processing
    return polydata


def visualize_vtk(polydata):
    """Visualizes vtkPolyData in VTK."""
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.3)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor_style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(interactor_style)
    interactor.SetRenderWindow(render_window)

    render_window.Render()
    interactor.Start()


if __name__ == "__main__":

    folder_path = r"C:\Users\monti\AppData\Local\PixelPolish3D"
    filename = "LHP_Certified_Round_Polished_800_0001_11p7_2024-08-03.vtp"
    input_vtp = os.path.join(folder_path, filename)

    # Main workflow
    # output_vtp = "output_surface.vtp"  # Output file for the regularized surface

    # Read the VTK file
    polydata = read_vtp_file(input_vtp)

    start_time_1 = time.time()

    # Convert to GMSH
    gmsh_context = vtk_to_gmsh(polydata)

    # Regularize the mesh
    remesh_gmsh(target_size=1, algorithm=5, optimize=False)

    # Convert back to VTK
    regularized_polydata = gmsh_to_vtk()

    # gmsh.model.remove()  # Clean the current model after processing

    # Save the regularized surface
    # writer = vtk.vtkXMLPolyDataWriter()
    # writer.SetFileName(output_vtp)
    # writer.SetInputData(regularized_polydata)
    # writer.Write()

    lapse = round(time.time() - start_time_1, 3)
    print("GMSH Triangulation:", lapse)

    # Visualize the result
    visualize_vtk(regularized_polydata)

    # Finalize GMSH
    gmsh.finalize()






# # Initialize GMSH
# gmsh.initialize()

# # List of files to process
# files = ["file1.vtp", "file2.vtp", "file3.vtp"]
# processed_data = []

# # Process each file
# for file in files:
#     # Process the file with GMSH
#     polydata = process_file_with_gmsh(file, target_size=0.2)
#     processed_data.append(polydata)

#     # Clean up the current model
#     gmsh.model.remove()          # Remove the current model to free resources
#     gmsh.model.add("NextModel")  # Add a new model for the next file

# # Finalize GMSH once all files are processed
# gmsh.finalize()

# # Visualize the last processed file
# if processed_data:
#     visualize_vtk(processed_data[-1])

