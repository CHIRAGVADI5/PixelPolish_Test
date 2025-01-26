
import os
import time
import pyvista as pv
import pyacvd
import vtk

import numpy as np
import pandas as pd

# import sys
# import time
# import gc # garbage collector

import numpy as np
import vtk
# import random

from sklearn.cluster import KMeans
# from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
# from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

# import matplotlib.pyplot as plt

from collections import Counter
from collections import defaultdict
from Pixel3D.pxl_tools import pxl_path_to_appdata
from Pixel3D.pxl_gmsh import vtk_to_gmsh, remesh_gmsh, gmsh_to_vtk



def get_elevation(normal):
    """
    Calculate the elevation angle of a normal vector.

    Parameters:
        normal (numpy.ndarray): The normal vector as a numpy array [x, y, z].

    Returns:
        float: The elevation angle in degrees.
    """
    # Extract the coordinates of the vector
    x, y, z = normal

    # Calculate the radial distance r
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Handle the special case where r is zero
    if r == 0:
        return 0.0

    # Calculate the polar angle (colatitude) with respect to the Z axis
    phi = np.arcsin(y / r)

    # Convert to degrees
    phi_degrees = np.degrees(phi)

    return phi_degrees


def get_face_normal(polydata, face_id):
    """
    Get the normal of a specific face in vtkPolyData as a numpy array.

    Parameters:
        polydata (vtk.vtkPolyData): The input vtkPolyData with normals assigned.
        face_id (int): The ID of the face for which the normal is required.

    Returns:
        numpy.ndarray: The normal vector of the specified face.
    """
    normals = polydata.GetCellData().GetNormals()
    if not normals:
        raise ValueError("Normals are not assigned to the faces in the vtkPolyData.")

    normal = [0.0, 0.0, 0.0]
    normals.GetTuple(face_id, normal)
    return np.array(normal)


def get_cell_area(polydata, face_id):
    """
    Calculate the area of a specific face in vtkPolyData.

    Parameters:
        polydata (vtk.vtkPolyData): The input vtkPolyData.
        face_id (int): The ID of the face for which the area is required.

    Returns:
        float: The area of the specified face.
    """
    cell = polydata.GetCell(face_id)
    points = cell.GetPoints()
    num_points = points.GetNumberOfPoints()

    # Get the normal of the polygon (assumed planar)
    normal = get_face_normal(polydata, face_id)

    # Calculate the area using the projection onto the plane defined by the normal
    area = 0.0
    origin = np.array(points.GetPoint(0))
    for i in range(1, num_points - 1):
        p1 = np.array(points.GetPoint(i)) - origin
        p2 = np.array(points.GetPoint(i + 1)) - origin
        cross = np.cross(p1, p2)
        area += np.dot(cross, normal) / 2.0

    return abs(area)


def add_group_labels_to_surface(polydata, offset=-0.2):
    """
    Add labels to the centroids of each group's facets that always face the camera.

    Parameters:
        polydata (vtk.vtkPolyData): The input vtkPolyData with grouped facets.
        num_groups (int): The number of groups.

    Returns:
        list: A list of vtkBillboardTextActor3D for the labels.
    """
    text_actors = []
    centroids = []

    
    group_array:vtk.vtkIntArray = polydata.GetCellData().GetArray("FacetGroups")
    ng = group_array.GetRange()
    num_groups = int(ng[1] + 1)

    for group_id in range(num_groups):
        # Extract cells belonging to the current group
        group_cells = [i for i in range(polydata.GetNumberOfCells())
                       if group_array.GetValue(i) == group_id]
                    #    if polydata.GetCellData().GetArray("FacetGroups").GetValue(i) == group_id]

        if not group_cells:
            continue
        
        # Calculate the centroid of the group
        centroid = np.zeros(3)
        suma = 0
        for cell_id in group_cells:
            points = vtk.vtkIdList()
            polydata.GetCellPoints(cell_id, points)
            for j in range(points.GetNumberOfIds()):
                point = [0.0, 0.0, 0.0]
                polydata.GetPoint(points.GetId(j), point)
                centroid += np.array(point)
                suma += 1

        centroid /= suma
        centroid[1] += offset  # Slight offset for better visibility
        centroids.append(centroid)

        # Create a text label for the group ID
        text_actor = vtk.vtkBillboardTextActor3D()
        text_actor.SetInput(str(group_id))
        text_actor.SetPosition(centroid)
        text_actor.GetTextProperty().SetFontSize(18)
        text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)  # White color
        text_actor.GetTextProperty().SetBackgroundColor(0.0, 0.0, 0.0)  # Black background
        text_actor.GetTextProperty().SetBackgroundOpacity(0.5)

        # Add the text actor to the list
        text_actors.append(text_actor)

    return text_actors


def regularize_mesh_with_target_points_gmsh(input_polydata:vtk.vtkPolyData, target_size=0.5):
  
    '''
    remesing using gmsh
    '''
    start_time_1 = time.time()

    # Convert to GMSH
    gmsh_context = vtk_to_gmsh(input_polydata)

    # Regularize the mesh
    remesh_gmsh(target_size=target_size, algorithm=5, optimize=False)

    # Convert back to VTK
    regularized_polydata = gmsh_to_vtk()
    
    # Compute normals
    normal_generator = vtk.vtkPolyDataNormals()
    normal_generator.SetInputData(regularized_polydata)
    # normal_generator.SetInputData(triangle_filter.GetOutput())
    # normal_generator.SetInputData(input_polydata)
    normal_generator.ComputePointNormalsOff()
    normal_generator.ComputeCellNormalsOn()
    normal_generator.Update()
    
    n_cells = regularized_polydata.GetNumberOfCells()
    n_points = regularized_polydata.GetNumberOfPoints()
    
    lapse = round(time.time() - start_time_1, 3)
    print("gmsh remeshing lapse:", lapse)
    print("cells, points:", n_cells, n_points)

    output_polydata:vtk.vtkPolyData = vtk.vtkPolyData()   
    output_polydata.DeepCopy(normal_generator.GetOutput())
    return output_polydata


def regularize_mesh_with_target_points_vtk(input_polydata:vtk.vtkPolyData, target_points=100, use_pyacvd=False):
    """
    Remeshing using vtk.
    
    Regularize a mesh by triangulating and remeshing with precise target points.
    
    Parameters:
        input_polydata (vtk.vtkPolyData): Input surface with irregular polygons.
        target_points (int): Target number of points for remeshing.

    Returns:
        vtk.vtkPolyData: Regularized surface.
    """
    start_time_1 = time.time()

    # Step 1: Triangulate the surface to ensure all cells are triangles
    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(input_polydata)
    triangle_filter.Update()
    triangulated_polydata:vtk.vtkPolyData = triangle_filter.GetOutput()
    n_original_cells = triangulated_polydata.GetNumberOfCells()

    if use_pyacvd:
        # Step 2: Remesh using PyACVD
        pv_polydata = pv.PolyData(triangulated_polydata, deep=False)
        remesher = pyacvd.Clustering(pv_polydata)
        # Avoid excessive subdivision to respect target points
        remesher.subdivide(1)  # Minimal subdivision for basic refinement
        remesher.cluster(target_points)  # Set the exact number of points
        # Calculate the target reduction
        n_original_cells = remesher.mesh.GetNumberOfCells()
    
    target_cells = 1200
    target_reduction = 1 - (target_cells / n_original_cells)
    print("remeshing output cells:", n_original_cells)

    # Create the decimation filter
    decimation_filter = vtk.vtkQuadricDecimation()
    if use_pyacvd:
        decimation_filter.SetInputData(remesher.mesh)
    else:
        decimation_filter.SetInputData(triangulated_polydata)
        
    decimation_filter.SetTargetReduction(target_reduction)  # Reduce 90% of the cells
    decimation_filter.Update()

    # Get the simplified polydata
    # simplified_polydata = decimation_filter.GetOutput()    
    
    # # Crear el objeto de remallado isotrópico
    # isotropic_remeshing = vtk.vtkIsotropicDiscreteRemeshing()
    # # Establecer los parámetros deseados
    # isotropic_remeshing.SetInputData(remesher.mesh)  # polydata es tu malla de entrada
    # isotropic_remeshing.SetNumberOfOutputTriangles(1000)  # Número deseado de celdas
    # isotropic_remeshing.SetUseAnisotropicSizingFunction(False)  # Configuración isotrópica
    # isotropic_remeshing.Update()

    # # Obtener la malla remallada
    # remeshed_polydata = isotropic_remeshing.GetOutput()

    # Compute normals
    normal_generator = vtk.vtkPolyDataNormals()
    normal_generator.SetInputData(decimation_filter.GetOutput())
    # normal_generator.SetInputData(triangle_filter.GetOutput())
    # normal_generator.SetInputData(input_polydata)
    normal_generator.ComputePointNormalsOff()
    normal_generator.ComputeCellNormalsOn()
    normal_generator.Update()
    
    n_cells = decimation_filter.GetOutput().GetNumberOfCells()
    n_points = decimation_filter.GetOutput().GetNumberOfPoints()
    
    lapse = round(time.time() - start_time_1, 3)
    print("regularization lapse:", lapse)
    print("cells, points:", n_cells, n_points)

    output_polydata:vtk.vtkPolyData = vtk.vtkPolyData()   
    output_polydata.DeepCopy(normal_generator.GetOutput())
    return output_polydata
    # Return the remeshed surface
    # return remesher.mesh


def split_polydata_by_elevation(polydata):
    """
    Split a vtkPolyData into three connected surfaces based on elevation angle.

    Parameters:
        polydata (vtk.vtkPolyData): The input vtkPolyData with normals assigned.

    Returns:
        tuple: Three vtkPolyData objects (s1, s2, s3) for negative, zero, and positive elevation surfaces.
    """
    # Get number of faces in polyData
    num_faces = polydata.GetNumberOfCells()
    cell_normals = polydata.GetCellData().GetNormals()
    
    if not cell_normals: 
        return None, None, None

    # Create three vtkPolyData objects to store the split surfaces
    s1 = vtk.vtkPolyData()
    s2 = vtk.vtkPolyData()
    s3 = vtk.vtkPolyData()

    # Copy the input polydata structure
    s1.DeepCopy(polydata)
    s2.DeepCopy(polydata)
    s3.DeepCopy(polydata)

    # Initialize cell arrays for keeping the selected cells
    s1_cells = vtk.vtkCellArray()
    s2_cells = vtk.vtkCellArray()
    s3_cells = vtk.vtkCellArray()
    
    # Initialize normals arrays for each subset
    s1_normals = vtk.vtkDoubleArray()
    s1_normals.SetNumberOfComponents(3)
    s1_normals.SetName("Normals")

    s2_normals = vtk.vtkDoubleArray()
    s2_normals.SetNumberOfComponents(3)
    s2_normals.SetName("Normals")

    s3_normals = vtk.vtkDoubleArray()
    s3_normals.SetNumberOfComponents(3)
    s3_normals.SetName("Normals")        

    n_cells = int(num_faces)
    cell_size = cell_normals.GetNumberOfComponents()
    normal = [0.0, 0.0, 0.0]
    
    # Resize and initialize the arrays
    # self.normals_array.clear()
    # self.normals_array = [[0.0] * cell_size for _ in range(n_cells)]
    # self.cell_type_array.clear()
    # self.cell_type_array = [0] * n_cells
    
    elevation_limit = 10.0 ## degrees
    
    # Fill normals_array with cellNormals
    for face_id in range(n_cells):
        # Retrieve the normal for the specified face
        cell_normals.GetTuple(face_id, normal)
        # Convert to numpy array
        # normal_array = np.array(normal)
        elevation = get_elevation(normal)                        
        if elevation < -elevation_limit:
            s1_cells.InsertNextCell(polydata.GetCell(face_id))
            s1_normals.InsertNextTuple(normal)
        elif elevation > elevation_limit:
            s3_cells.InsertNextCell(polydata.GetCell(face_id))
            s3_normals.InsertNextTuple(normal)
        else:
            s2_cells.InsertNextCell(polydata.GetCell(face_id))
            s2_normals.InsertNextTuple(normal)
            
    # Assign the selected cells to the corresponding vtkPolyData
    s1.SetPolys(s1_cells)
    s1.GetCellData().SetNormals(s1_normals)
    
    s2.SetPolys(s2_cells)
    s2.GetCellData().SetNormals(s2_normals)
    
    s3.SetPolys(s3_cells)
    s3.GetCellData().SetNormals(s3_normals)
    
    return s1, s2, s3

def detect_jump_with_max_index(group_areas:dict, max_index=4):
    """
    Detect the first significant jump in a dictionary using the value at index `max_index` divided by 2 as the threshold.
    Only considers indices up to `max_index` for detecting the jump.

    Parameters:
        group_areas (dict): Input dictionary with keys and values.
        max_index (int): Maximum index to consider for detecting the jump.

    Returns:
        int: Index of the first significant jump based on the threshold. Returns -1 if no jump is found.
    """
    values = list(group_areas.values())
    
    if len(values) <= max_index:
        return -1  # Not enough elements to apply the threshold

    old_option = False
    if old_option:
        threshold = values[max_index] / 4  # Calculate the threshold using the value at index `max_index`

        for i in range(max_index):  # Only consider indices up to `max_index`
            if abs(values[i + 1] - values[i]) > threshold:
                return i + 1  # Return the index of the jump
    else:
        threshold = 0.05  # PENDIENTE

        for i in range(len(values)):  # Only consider indices up to `max_index`
            if abs(values[i]) > threshold:
                return i  # Return the index of the jump
        
    return -1  # Return -1 if no jump is found


def calculate_group_dispersions_and_areas(polydata):
    """
    Calculate the angular dispersion, maximum angular deviation, maximum angular difference, 
    maximum projection separation, and total area for each group in a classified vtkPolyData.

    Parameters:
        polydata (vtk.vtkPolyData): The input vtkPolyData with group labels assigned as scalars.

    Returns:
        dict: A dictionary mapping labels to angular dispersion (standard deviation of angles), sorted by value.
        dict: A dictionary mapping labels to maximum angular deviation, sorted by value.
        dict: A dictionary mapping labels to maximum angular difference, sorted by value.
        dict: A dictionary mapping labels to maximum projection separation, sorted by value.
        dict: A dictionary mapping labels to total group area, sorted by value.
    """
    # Extract group labels
    labels = polydata.GetCellData().GetScalars()
    if not labels:
        raise ValueError("FacetGroups scalar array is not assigned to the polydata.")

    num_cells = polydata.GetNumberOfCells()

    # Initialize dictionaries for group properties
    group_normals = defaultdict(list)
    group_areas = defaultdict(float)
    group_points = defaultdict(list)

    # Populate group normals, areas, and points
    for i in range(num_cells):
        label = int(labels.GetValue(i))
        normal = get_face_normal(polydata, i)
        area = get_cell_area(polydata, i)
        cell = polydata.GetCell(i)
        for j in range(cell.GetNumberOfPoints()):
            group_points[label].append(np.array(cell.GetPoints().GetPoint(j)))

        group_normals[label].append(normal)
        group_areas[label] += area

    # Calculate angular dispersion, maximum angular deviation, maximum angular difference, maximum projection separation, and total area for each group
    group_dispersion = {}
    group_max_deviation = {}
    group_max_difference = {}
    group_max_projection = {}
    if False:
        for label, normals in group_normals.items():
            normals = np.array(normals)
            mean_normal = np.mean(normals, axis=0)
            mean_normal /= np.linalg.norm(mean_normal)  # Normalize mean normal

            # Calculate angles between each normal and the mean normal
            angles = [
                np.arccos(np.clip(np.dot(normal / np.linalg.norm(normal), mean_normal), -1.0, 1.0))
                for normal in normals if np.linalg.norm(normal) > 0
            ]

            # Standard deviation of angles
            group_dispersion[label] = float(round(np.std(angles), 3))

            # Maximum angular deviation
            group_max_deviation[label] = float(round(np.max(angles), 3))

            # Calculate pairwise angular differences
            max_difference = 0.0
            for i in range(len(normals)):
                if np.linalg.norm(normals[i]) == 0:
                    continue 
                for j in range(i + 1, len(normals)):
                    if np.linalg.norm(normals[j]) == 0:
                        continue 
                    angle_diff = np.arccos(np.clip(np.dot(normals[i] / np.linalg.norm(normals[i]), normals[j] / np.linalg.norm(normals[j])), -1.0, 1.0))
                    max_difference = max(max_difference, angle_diff)

            group_max_difference[label] = float(round(np.degrees(max_difference), 3))  # Convert to degrees

            # Calculate maximum projection separation
            projections = [np.dot(point, mean_normal) for point in group_points[label]]
            max_projection = max(projections) - min(projections)
            group_max_projection[label] = float(round(max_projection, 3))

        # Convert areas to float and sort dictionaries by value
        group_dispersion = dict(sorted(group_dispersion.items(), key=lambda x: x[1]))
        group_max_deviation = dict(sorted(group_max_deviation.items(), key=lambda x: x[1]))
        group_max_difference = dict(sorted(group_max_difference.items(), key=lambda x: x[1]))
        group_max_projection = dict(sorted(group_max_projection.items(), key=lambda x: x[1]))
    
    group_areas = dict(sorted({label: float(round(area, 3)) for label, area in group_areas.items()}.items(), key=lambda x: x[1]))

    return group_dispersion, group_max_deviation, group_max_difference, group_max_projection, group_areas


# ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo    
def classify_facets_by_normals(polydata, num_groups):
    """
    Classify the facets of a vtkPolyData into groups based on their normals.

    Parameters:
        polydata (vtk.vtkPolyData): The input vtkPolyData with normals assigned.
        num_groups (int): The number of groups to classify the facets into.

    Returns:
        vtkPolyData: The input vtkPolyData with a scalar array assigned to the cells, indicating the group index.
    """
    num_cells = polydata.GetNumberOfCells()
    
    # Extract normals for all cells
    normals = np.array([get_face_normal(polydata, i) for i in range(num_cells)])

    # Perform clustering on the normals
    kmeans = KMeans(n_clusters=num_groups, random_state=0).fit(normals)
    labels = kmeans.labels_

    # integer_list = labels.tolist()
    # value_counts = Counter(integer_list)

    # Create a scalar array to store the group labels
    group_array = vtk.vtkIntArray()
    group_array.SetName("FacetGroups")
    group_array.SetNumberOfComponents(1)
    group_array.SetNumberOfTuples(num_cells)

    for i in range(num_cells):
        group_array.SetValue(i, labels[i])

    # Assign the scalar array to the polydata
    polydata.GetCellData().SetScalars(group_array)

    return polydata, labels


# ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
def classify_facets_by_normals_area_weighting(polydata, num_groups):
    """
    Classify the facets of a vtkPolyData into groups by amplifying the influence of celdas with larger areas.

    Parameters:
        polydata (vtk.vtkPolyData): The input vtkPolyData with normals assigned.
        num_groups (int): The number of groups to classify the facets into.

    Returns:
        vtkPolyData: The input vtkPolyData with a scalar array assigned to the cells, indicating the group index.
    """
    num_cells = polydata.GetNumberOfCells()

    # Extract normals and areas for all cells
    augmented_normals = []
    for i in range(num_cells):
        normal = get_face_normal(polydata, i)
        area = get_cell_area(polydata, i)

        # Amplify influence by duplicating normals proportional to area
        duplication_factor = max(1, int(area * 10))  # Scale factor for duplication
        augmented_normals.extend([normal] * duplication_factor)

    augmented_normals = np.array(augmented_normals)

    # Perform clustering on the augmented normals
    kmeans = KMeans(n_clusters=num_groups, random_state=0).fit(augmented_normals)
    labels = kmeans.labels_

    # Map augmented labels back to original cells
    original_labels = np.zeros(num_cells, dtype=int)
    counter = 0
    for i in range(num_cells):
        area = get_cell_area(polydata, i)
        duplication_factor = max(1, int(area * 10))
        original_labels[i] = np.bincount(labels[counter:counter + duplication_factor]).argmax()
        counter += duplication_factor

    # Create a scalar array to store the group labels
    group_array = vtk.vtkIntArray()
    group_array.SetName("FacetGroups")
    group_array.SetNumberOfComponents(1)
    group_array.SetNumberOfTuples(num_cells)

    for i in range(num_cells):
        group_array.SetValue(i, original_labels[i])

    # Assign the scalar array to the polydata
    polydata.GetCellData().SetScalars(group_array)

    return polydata, original_labels


# oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

def classify_facets_by_normals_agglomerative(polydata, n_clusters=10):
    """
    Classify the facets of a vtkPolyData into groups based on their normals using Agglomerative Hierarchical Clustering.

    Parameters:
        polydata (vtk.vtkPolyData): The input vtkPolyData with normals assigned.
        n_clusters (int): The number of clusters to form.

    Returns:
        vtkPolyData: The input vtkPolyData with a scalar array assigned to the cells, indicating the group index.
    """
    num_cells = polydata.GetNumberOfCells()

    # Extract normals for all cells
    normals = []
    for i in range(num_cells):
        normal = get_face_normal(polydata, i)
        normals.append(normal)
    normals = np.array(normals)

    # Apply Agglomerative Clustering
    agglomerative = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='euclidean',  # Updated parameter
        linkage='ward'
    )
    labels = agglomerative.fit_predict(normals)

    # Create a scalar array to store the group labels
    group_array = vtk.vtkIntArray()
    group_array.SetName("FacetGroups")
    group_array.SetNumberOfComponents(1)
    group_array.SetNumberOfTuples(num_cells)

    for i in range(num_cells):
        group_array.SetValue(i, labels[i])

    # Assign the scalar array to the polydata
    polydata.GetCellData().SetScalars(group_array)

    return polydata, labels


# oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

def classify_facets_by_normals_dbscan(polydata, eps, min_samples):
    """
    Classify the facets of a vtkPolyData into groups based on their normals using DBSCAN.

    Parameters:
        polydata (vtk.vtkPolyData): The input vtkPolyData with normals assigned.
        eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
        min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.

    Returns:
        vtkPolyData: The input vtkPolyData with a scalar array assigned to the cells, indicating the group index.
    """
    num_cells = polydata.GetNumberOfCells()

    # Extract normals for all cells
    normals = []
    for i in range(num_cells):
        normals.append(get_face_normal(polydata, i))
    normals = np.array(normals)

    # Apply DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = db.fit_predict(normals)

    # Create a scalar array to store the group labels
    group_array = vtk.vtkIntArray()
    group_array.SetName("FacetGroups")
    group_array.SetNumberOfComponents(1)
    group_array.SetNumberOfTuples(num_cells)

    for i in range(num_cells):
        group_array.SetValue(i, labels[i])

    # Assign the scalar array to the polydata
    polydata.GetCellData().SetScalars(group_array)
    return polydata, labels





def facets_clustering_and_classification(table_surface, pavilion_surface, table_faces=33, pavilion_faces=24, selected_classifier=3):
    '''
    table_surface, pavilion_surface are the splited surfaces, also called s1 and s3.
    '''
    def facet_classification(selected_classifier=3, surface=None, num_faces=33):
        
        if selected_classifier == 1:
            classified_polydata, labels = classify_facets_by_normals(surface, num_groups=num_faces)
        elif selected_classifier == 2:
            classified_polydata, labels = classify_facets_by_normals_area_weighting(surface, num_groups=num_faces)
        elif selected_classifier == 3:
            classified_polydata, labels = classify_facets_by_normals_agglomerative(surface, n_clusters=num_faces)
        elif selected_classifier == 4:
            classified_polydata, labels = classify_facets_by_normals_dbscan(surface, eps=0.025, min_samples=5)

        # Here group_areas is the only one used 
        group_dispersion, group_max_deviation, group_max_difference, group_max_projection, group_areas \
            = calculate_group_dispersions_and_areas(classified_polydata)

        # print("group_areas_1: ", list(group_areas_1.values()))
        # print("group_areas_3: ", list(group_areas_3.values()))
        
        """
        Print all the keys and values of a dictionary in the format { key:value, ... }.

        Parameters:
            input_dict (dict): The dictionary to be printed.
        """
        print()
        print("{" + ", ".join([f"{key}:{value}" for key, value in group_areas.items()]) + "}")
        print()

        jump_index = detect_jump_with_max_index(group_areas, 7)
        return classified_polydata, jump_index, labels, group_areas

    old_option = False
    if old_option: 
        classified_table_surface, i1, table_labels, table_group_areas = \
            facet_classification(selected_classifier=selected_classifier, surface=table_surface, num_faces=table_faces)
        if i1 != -1:
            classified_table_surface, i1, table_labels, table_group_areas = \
                facet_classification(selected_classifier=selected_classifier, surface=table_surface, num_faces=table_faces+i1)
    else:
        n_jumps = 0
        n_good_faces = 0
        while n_good_faces < table_faces:
            num_faces = table_faces + n_jumps
            classified_table_surface, i1, table_labels, table_group_areas = \
                facet_classification(selected_classifier=selected_classifier, surface=table_surface, num_faces=num_faces)
            n_jumps = i1
            n_good_faces = num_faces - n_jumps
            pass
                
    if old_option:
        classified_pavilion_surface, i3, pavilion_labels, pavilion_group_areas = \
            facet_classification(selected_classifier=selected_classifier, surface=pavilion_surface, num_faces=pavilion_faces)
        if i3 != -1:
            classified_pavilion_surface, i3, pavilion_labels, pavilion_group_areas = \
                facet_classification(selected_classifier=selected_classifier, surface=pavilion_surface, num_faces=pavilion_faces+i3)
    else:
        n_jumps = 0
        n_good_faces =0
        while n_good_faces < pavilion_faces:
            num_faces = pavilion_faces + n_jumps
            classified_pavilion_surface, i3, pavilion_labels, pavilion_group_areas = \
                facet_classification(selected_classifier=selected_classifier, surface=pavilion_surface, num_faces = num_faces)
            n_jumps = i3
            n_good_faces = num_faces - n_jumps
            pass
        
    # table_group_areas, pavilion_group_areas are dictionaries of group id vs group area
    return classified_table_surface, classified_pavilion_surface, i1, i3, table_labels, pavilion_labels, table_group_areas, pavilion_group_areas



def get_facet_properties(vtk_polydata, cell_indices):
    # Extract points and normals for the selected cells in the grouping
    points = vtk_polydata.GetPoints()
    normals_array = vtk_polydata.GetCellData().GetNormals()

    cell_points = []
    cell_normals = []

    for cell_index in cell_indices:
        cell = vtk_polydata.GetCell(cell_index)
        cell_normals.append(normals_array.GetTuple(cell_index))  # Normal of the cell

        for i in range(cell.GetNumberOfPoints()):
            point_id = cell.GetPointId(i)
            cell_points.append(points.GetPoint(point_id))

    # Convert to numpy arrays for easier calculations
    cell_points = np.array(cell_points)
    cell_normals = np.array(cell_normals)

    # Calculate the average point (centroid) of the points in the grouping
    centroid = np.mean(cell_points, axis=0)

    # Calculate the average normal from the cell normals
    normal_average = np.mean(cell_normals, axis=0)
    normal_average /= np.linalg.norm(normal_average)  # Normalize

    # Calculate the normal using PCA
    covariance_matrix = np.cov(cell_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    normal_pca = eigenvectors[:, 0]  # Vector with the smallest eigenvalue
    normal_pca /= np.linalg.norm(normal_pca)  # Normalize

    # Compute dot product between normal_average and normal_pca for validation
    dot_product = np.dot(normal_average, normal_pca)

    # Ensure both normals point in the same direction
    if dot_product < 0:
        normal_pca = -normal_pca
        dot_product = -dot_product

    # Compute the angle in degrees between normal_average and normal_pca
    angle_degrees = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))

    # Calculate distances of points to the average plane
    d = -np.dot(normal_average, centroid)  # Plane distance from origin
    distances = np.dot(cell_points, normal_average) + d  # Distances to the plane

    # Calculate statistics for distances
    std_dev = np.std(distances)
    min_dist = np.min(distances)
    max_dist = np.max(distances)

    # Count points outside one standard deviation on each side, including extreme points
    outside_std_count_min = np.sum(distances < -std_dev)
    outside_std_count_max = np.sum(distances > std_dev)

    # Results
    results = {
        "angle_deg": round(float(angle_degrees),4),
        "extreme_dist": {"min": round(float(min_dist),4), "max": round(float(max_dist),4)},
        "outside_std_min": int(outside_std_count_min),
        "outside_std_max": int(outside_std_count_max),
        "std_dev": round(float(std_dev),4),
        "dot_prod": round(float(dot_product),4),
        "normal_aver": normal_average.tolist(),
        "normal_pca": normal_pca.tolist(),
        "point_on_plane": centroid.tolist()
    }
    return results



def example_get_facet_properties():
    # Example usage
    # Create a vtkPolyData and define the cells (this would be your input)
    polydata = vtk.vtkPolyData()  # Assume your vtkPolyData is already populated
    cell_indices = [0, 1, 2]  # Indices of the cells in the grouping

    # Call the function
    result = get_facet_properties(polydata, cell_indices)

    # Print results
    print("Average normal (from cells):", result["normal_average"])
    print("Normal calculated by PCA:", result["normal_pca"])
    print("Dot product (normal_average . normal_pca):", result["dot_product"])
    print("Angle between normals (degrees):", result["angle_degrees"])
    print("Point on the plane:", result["point_on_plane"])
    print("Standard deviation of distances:", result["std_dev_distances"])
    print("Extreme distances to the plane:", result["extreme_distances"])
    print("Number of points outside one standard deviation (min side):", result["outside_std_count_min"])
    print("Number of points outside one standard deviation (max side):", result["outside_std_count_max"])


def facet_data_to_excel(data:dict, prefix:str="facet_data_table_", code_id:str=None):
    '''
    # Input dictionary example
    data = {
        25: {'angle_deg': 0.0001, 'extreme_dist': {'min': -0.0, 'max': 0.0}, 'outside_std_min': 3, 'outside_std_max': 25, 'std_dev': 0.0, 'dot_prod': 1.0, 'normal_aver': [-1.7560717796239548e-06, -0.999999999998312, 5.407564734032591e-07], 'normal_pca': [1.0004047533723864e-08, -0.9999999999999978, -6.757364368059626e-08], 'point_on_plane': [0.252583793166912, 2.1019559508622296, -0.2236007246931996]},
        28: {'angle_deg': 0.1798, 'extreme_dist': {'min': -0.0065, 'max': 0.0028}, 'outside_std_min': 13, 'outside_std_max': 0, 'std_dev': 0.0028, 'dot_prod': 1.0, 'normal_aver': [-0.6607362534227169, -0.7505491780634078, -0.010175201288137252], 'normal_pca': [-0.6629634790570077, -0.7485959471192655, -0.00913966045531517], 'point_on_plane': [-3.052697952242865, 2.935328863669133, 0.19844222900228223]},
        0: {'angle_deg': 3.8648, 'extreme_dist': {'min': -0.0638, 'max': 0.0148}, 'outside_std_min': 10, 'outside_std_max': 0, 'std_dev': 0.02, 'dot_prod': 0.9977, 'normal_aver': [-0.42552290674673443, -0.9043487800918327, 0.032917165434224835], 'normal_pca': [-0.37424936644193474, -0.9245446993402856, 0.07179492070886062], 'point_on_plane': [-2.095296850204468, 2.2510599581400554, 0.28993883311748503]},
        26: {'angle_deg': 0.2192, 'extreme_dist': {'min': -0.0034, 'max': 0.0032}, 'outside_std_min': 12, 'outside_std_max': 13, 'std_dev': 0.0021, 'dot_prod': 1.0, 'normal_aver': [0.6670972822487973, -0.7448085575265413, 0.015537974498024716], 'normal_pca': [0.6693587179699027, -0.7427257664107596, 0.017814112024965637], 'point_on_plane': [2.9483811602447973, 2.9258708773237285, 0.04445848122916438]},
        9: {'angle_deg': 0.7175, 'extreme_dist': {'min': -0.0162, 'max': 0.0079}, 'outside_std_min': 19, 'outside_std_max': 25, 'std_dev': 0.0066, 'dot_prod': 0.9999, 'normal_aver': [-0.5594003014029323, -0.8211519408179986, -0.11305216884758437], 'normal_pca': [-0.5685724421099556, -0.814140078179662, -0.11790382171386532], 'point_on_plane': [-2.390991238933621, 2.4868554798039524, -0.3950634581234419]}
    }
    '''

    # Build the path
    app_data_path = pxl_path_to_appdata()
    df_path = os.path.join(app_data_path, prefix + code_id + ".xlsx")

    # Create a list of rows for the DataFrame
    rows = []
    for facet_id, values in data.items():
        # Prepare each row as a dictionary
        row = {
            'facet_id': facet_id,
            'angle_deg': values['angle_deg'],
            'extreme_dist_min': values['extreme_dist']['min'],
            'extreme_dist_max': values['extreme_dist']['max'],
            'outside_std_min': values['outside_std_min'],
            'outside_std_max': values['outside_std_max'],
            'std_dev': values['std_dev'],
            'dot_prod': values['dot_prod'],
            'normal_aver': ', '.join(map(str, values['normal_aver'])),
            'normal_pca': ', '.join(map(str, values['normal_pca'])),
            'point_on_plane': ', '.join(map(str, values['point_on_plane']))
        }
        rows.append(row)

    # Convert the list of rows to a DataFrame
    df = pd.DataFrame(rows)

    # Save the DataFrame to an Excel file
    df.to_excel(df_path, index=False)

def get_planes_from_polydata(polydata:vtk.vtkPolyData, points:vtk.vtkPoints=None, normals:vtk.vtkDoubleArray=None):

    # Crear los objetos para almacenar las normales y puntos
    if normals == None:
        normals = vtk.vtkDoubleArray()
        normals.SetNumberOfComponents(3)  # Cada normal tiene 3 componentes (x, y, z)
        normals.SetName("FacetNormals")

    if points == None:
        points = vtk.vtkPoints()

    # Obtener las normales de las facetas del vtkPolyData
    facet_normals = polydata.GetCellData().GetNormals()
    if not facet_normals:
        raise RuntimeError("No normals found in the vtkPolyData.")

    # Iterar sobre cada celda del vtkPolyData
    for cell_id in range(polydata.GetNumberOfCells()):
        # Extraer una normal a la faceta
        normal = facet_normals.GetTuple(cell_id)
        normals.InsertNextTuple(normal)

        # Extraer un punto de la faceta (usamos el primer punto de la celda)
        cell = polydata.GetCell(cell_id)
        point_id = cell.GetPointId(0)  # ID del primer punto de la celda
        point = polydata.GetPoint(point_id)
        points.InsertNextPoint(point)

    return points, normals
    # Normales y puntos extraídos están ahora en `normals` y `points`
    # print(f"Extracted {normals.GetNumberOfTuples()} normals.")
    # print(f"Extracted {points.GetNumberOfPoints()} points.")


def get_planes_from_facet_data(data:dict, points:vtk.vtkPoints=None, normals:vtk.vtkDoubleArray=None):
    '''
    Data input example
    data = {
        25: {'angle_deg': 0.0001, 'extreme_dist': {'min': -0.0, 'max': 0.0}, 'outside_std_min': 3, 'outside_std_max': 25, 'std_dev': 0.0, 'dot_prod': 1.0, 'normal_aver': [-1.7560717796239548e-06, -0.999999999998312, 5.407564734032591e-07], 'normal_pca': [1.0004047533723864e-08, -0.9999999999999978, -6.757364368059626e-08], 'point_on_plane': [0.252583793166912, 2.1019559508622296, -0.2236007246931996]},
        28: {'angle_deg': 0.1798, 'extreme_dist': {'min': -0.0065, 'max': 0.0028}, 'outside_std_min': 13, 'outside_std_max': 0, 'std_dev': 0.0028, 'dot_prod': 1.0, 'normal_aver': [-0.6607362534227169, -0.7505491780634078, -0.010175201288137252], 'normal_pca': [-0.6629634790570077, -0.7485959471192655, -0.00913966045531517], 'point_on_plane': [-3.052697952242865, 2.935328863669133, 0.19844222900228223]},
        0: {'angle_deg': 3.8648, 'extreme_dist': {'min': -0.0638, 'max': 0.0148}, 'outside_std_min': 10, 'outside_std_max': 0, 'std_dev': 0.02, 'dot_prod': 0.9977, 'normal_aver': [-0.42552290674673443, -0.9043487800918327, 0.032917165434224835], 'normal_pca': [-0.37424936644193474, -0.9245446993402856, 0.07179492070886062], 'point_on_plane': [-2.095296850204468, 2.2510599581400554, 0.28993883311748503]},
        26: {'angle_deg': 0.2192, 'extreme_dist': {'min': -0.0034, 'max': 0.0032}, 'outside_std_min': 12, 'outside_std_max': 13, 'std_dev': 0.0021, 'dot_prod': 1.0, 'normal_aver': [0.6670972822487973, -0.7448085575265413, 0.015537974498024716], 'normal_pca': [0.6693587179699027, -0.7427257664107596, 0.017814112024965637], 'point_on_plane': [2.9483811602447973, 2.9258708773237285, 0.04445848122916438]},
        9: {'angle_deg': 0.7175, 'extreme_dist': {'min': -0.0162, 'max': 0.0079}, 'outside_std_min': 19, 'outside_std_max': 25, 'std_dev': 0.0066, 'dot_prod': 0.9999, 'normal_aver': [-0.5594003014029323, -0.8211519408179986, -0.11305216884758437], 'normal_pca': [-0.5685724421099556, -0.814140078179662, -0.11790382171386532], 'point_on_plane': [-2.390991238933621, 2.4868554798039524, -0.3950634581234419]}
    }
    '''

    # Create vtkDoubleArray for normals
    own_normals = vtk.vtkDoubleArray()
    own_normals.SetNumberOfComponents(3)  # Each normal has 3 components
    own_normals.SetName("FacetNormals")

    # Create vtkPoints for points
    own_points = vtk.vtkPoints()

    # Populate normals and points from data
    for facet_id, values in data.items():
        # Add normal_pca to normals / 'normal_pca' 'normal_aver'
        normal_pca = values['normal_pca']
        # normal_pca = values['normal_aver']
        normals.InsertNextTuple(normal_pca)
        own_normals.InsertNextTuple(normal_pca)

        # Add point_on_plane to points
        point_on_plane = values['point_on_plane']
        points.InsertNextPoint(point_on_plane)
        own_points.InsertNextPoint(point_on_plane)

    return own_points, own_normals
    # Output results (optional verification)
    # print(f"Number of normals: {normals.GetNumberOfTuples()}")
    # print(f"Number of points: {points.GetNumberOfPoints()}")

