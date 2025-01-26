import numpy as np
import matplotlib.pyplot as plt
import vtk

def is_ccw(points):
    """
    Verifies if a list of 2D points is in counterclockwise (CCW) order.

    Parameters:
        points (list): List of points [(x1, y1), (x2, y2), ...]

    Returns:
        bool: True if the points are in CCW order, False otherwise.
    """
    total = 0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        total += (x2 - x1) * (y2 + y1)
    return total < 0

def remove_non_convex(points, ccw_verification=True):
    """
    Removes points that violate the convexity of the polygon.

    Parameters:
        points (list): List of points [(x1, y1), (x2, y2), ...]
        Open contour (last point differs from first point)

    Returns:
        list: List of points where convexity is maintained.
    """
    def cross_product(o, a, b):
        """Calculate the cross product of vectors OA and OB."""
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    n = len(points)
    if n < 3:
        return points

    # Check if the polygon is CCW and reverse if necessary
    if ccw_verification:
        ccw = is_ccw(points)
        if not ccw:
            points = points[::-1]
    else:
        ccw = True

    convex_points = points[:]
    i = 0
    while i < len(convex_points):
        o = convex_points[i - 2]
        a = convex_points[i - 1]
        b = convex_points[i % len(convex_points)]
        if cross_product(o, a, b) < 0:
            convex_points.pop((i - 1) % len(convex_points))
            i = max(i - 1, 0)
        else:
            i += 1

    return convex_points, ccw

def check_convexity_and_ccw(flat_points):
    """
    Removes points that violate convexity and checks if points are in CCW order.

    Parameters:
        flat_points (list): Flattened list of XYZ coordinates [x1, y1, z1, x2, y2, z2, ...]

    Returns:
        tuple: (cleaned_points_flat, is_ccw_order)
    """
    # Convert flat points to 2D (ignoring Z coordinate)
    points = [(flat_points[i], flat_points[i + 1]) for i in range(0, len(flat_points), 3)]

    # Remove non-convex points
    convex_points, ccw = remove_non_convex(points)

    # Convert back to flat format including Z
    cleaned_points_flat = [coord for point in convex_points for coord in (*point, 0)]

    return cleaned_points_flat, ccw

def ensure_ccw_or_cw(flat_points, desired_ccw=True):
    """
    Ensures the polygon points are either CCW or CW based on the desired order.

    Parameters:
        flat_points (list): Flattened list of XYZ coordinates [x1, y1, z1, x2, y2, z2, ...]
        desired_ccw (bool): True for CCW, False for CW.

    Returns:
        list: Flattened list of XYZ coordinates in the desired order.
    """
    # Convert flat points to 2D (ignoring Z coordinate)
    points = [(flat_points[i], flat_points[i + 1]) for i in range(0, len(flat_points), 3)]

    # Check current order
    ccw = is_ccw(points)

    # Reverse if the current order does not match the desired order
    if ccw != desired_ccw:
        points = points[::-1]

    # Convert back to flat format including Z
    corrected_points_flat = [coord for point in points for coord in (*point, 0)]

    return corrected_points_flat

def plot_polygons(original_flat, cleaned_flat):
    """
    Plot the original and cleaned polygons.

    Parameters:
        original_flat (list): Original flattened list of XYZ coordinates.
        cleaned_flat (list): Cleaned flattened list of XYZ coordinates.
    """
    original_points = [(original_flat[i], original_flat[i + 1]) for i in range(0, len(original_flat), 3)]
    cleaned_points = [(cleaned_flat[i], cleaned_flat[i + 1]) for i in range(0, len(cleaned_flat), 3)]

    original_x, original_y = zip(*original_points)
    cleaned_x, cleaned_y = zip(*cleaned_points)

    plt.figure(figsize=(8, 8))
    plt.plot(original_x + (original_x[0],), original_y + (original_y[0],), label="Original Polygon", linestyle='--', marker='o')
    plt.plot(cleaned_x + (cleaned_x[0],), cleaned_y + (cleaned_y[0],), label="Cleaned Polygon", linestyle='-', marker='x')

    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Original and Cleaned Polygons")
    plt.axis("equal")
    plt.grid(True)
    plt.show()



def save_polygons_HR(polygons:list, file_path, w, h, pixel_size, center_offset=0.34409, y_max=0):
    """
    Saves a list of polygons to a file in a specific format.

    Parameters:
        polygons (list): List of flat_points, each representing a polygon in [x1, y1, z1, ...] format.
        file_path (str): Name of the output file.
        w (int): Width of the space.
        h (int): Height of the space.
        pixel_size (float): Size of each pixel.
        center_offset (float): Offset to the center of the space.
        y_max (float): Maximum Y value of the space.
        
        1936            w
        1216            h
        0.01172         pixel_size
        0.34409         center_offset 
        522.425         y_max
        200             N/2
        15
        811.26192076900861 179.34776457071757 0
        1117.4422673309462 179.34776457071757 0
        1171.9545923982605 205.54923568283576 0
        1204.3386840820312 234.70270538330078 0
        1241.8966064453125 268.4536727718189 0
        1241.8966064453125 281.71456464452598 0
        1139.242431640625 375.18965148925781 0
        1038 467.37930297851562 0
        1023 480.79412841796875 0
        966.53140576547969 527.29766367029163 0
        907 478.21875 0
        866 441.48486328125 0
        688.44528198242188 280.62806131037212 0
        688.44528198242188 267.29132238046219 0
        756.90007808790176 206.65993356749675 0
        19        
    """
    n2 = int(len(polygons)/2)
    try:
        with open(file_path, 'w') as file:
            # Write header information
            file.write(f"{w}\n")
            file.write(f"{h}\n")
            file.write(f"{pixel_size}\n")
            file.write(f"{center_offset}\n")
            file.write(f"{y_max}\n")

            # Write the total number of polygons
            file.write(f"{n2}\n")

            for polygon in polygons[:n2]:
                # Convert flat points to 3D tuples
                points = [(polygon[i], polygon[i + 1], polygon[i + 2]) for i in range(0, len(polygon), 3)]

                # Write the number of points in the polygon
                file.write(f"{len(points)}\n")

                # Write the points with maximum precision
                for x, y, z in points:
                    file.write(f"{x:.16f} {y:.16f} {z:.16f}\n")
    except IOError as e:
        print(f"Error opening or writing to file {file_path}: {e}")



def example_check_convexity_and_ccw():
    # Example usage
    flat_points = [966.0, 183.34031677246094, 0.0, 1124.0, 183.39157104492188, 0.0, 1129.0, 184.0, 0.0, 1130.076904296875, 185.0, 0.0, 1131.0, 185.375, 0.0, 1133.6842041015625, 187.0, 0.0, 1137.5238037109375, 189.0, 0.0, 1170.0, 204.60975646972656, 0.0, 1172.0, 205.80555725097656, 0.0, 1174.0, 207.40541076660156, 0.0, 1200.0, 231.0, 0.0, 1234.6773681640625, 262.0, 0.0, 1238.8648681640625, 266.0, 0.0, 1241.3590087890625, 269.0, 0.0, 1241.7142333984375, 270.0, 0.0, 1241.9630126953125, 272.0, 0.0, 1241.8302001953125, 280.0, 0.0, 1241.6444091796875, 281.0, 0.0, 1240.48486328125, 283.0, 0.0, 1092.0, 418.20001220703125, 0.0, 1090.0, 420.23333740234375, 0.0, 1089.0, 420.8780517578125, 0.0, 1038.0, 467.3793029785156, 0.0, 1023.0, 480.79412841796875, 0.0, 976.0, 519.5, 0.0, 973.0, 521.6756591796875, 0.0, 971.0, 522.3076782226562, 0.0, 966.0, 522.2820434570312, 0.0, 961.0, 521.7916870117188, 0.0, 960.0, 521.5, 0.0, 959.0, 520.9302368164062, 0.0, 956.0, 518.6153564453125, 0.0, 907.0, 478.21875, 0.0, 866.0, 441.48486328125, 0.0, 692.0, 283.8484802246094, 0.0, 689.4418334960938, 281.0, 0.0, 688.5714111328125, 279.0, 0.0, 688.3191528320312, 272.0, 0.0, 688.4666748046875, 269.0, 0.0, 689.4053955078125, 267.0, 0.0, 694.0, 262.3714294433594, 0.0, 753.0, 210.11428833007812, 0.0, 757.0, 206.81080627441406, 0.0, 763.0, 203.59524536132812, 0.0, 794.0, 188.02040100097656, 0.0, 798.0, 185.62069702148438, 0.0, 799.368408203125, 185.0, 0.0, 800.4000244140625, 184.0, 0.0, 803.0, 183.5371856689453, 0.0, 807.0, 183.34759521484375, 0.0, 966.0, 183.34031677246094, 0.0]
    cleaned_points, ccw = check_convexity_and_ccw(flat_points)

    # print("Cleaned Points:", cleaned_points)
    print("Is CCW, len of input points:", ccw, len(flat_points)/3)

    print("len of output points:", ccw, len(cleaned_points)/3)

    # Plot polygons
    plot_polygons(flat_points, cleaned_points)


def get_min_max_xy(vertices):
    """
    Calculates the minimum and maximum values of X and Y coordinates
    from a flattened list of XYZ values where Z is always 0.

    Args:
        vertices (list): Flattened list of XYZ values. Example: [x1, y1, z1, x2, y2, z2, ...]

    Returns:
        dict: Dictionary with the minimum and maximum values of X and Y.

    # Example usage
    polygon_vertices = [1, 2, 0, 3, 4, 0, 5, 1, 0, -2, -3, 0]
    result = get_min_max_xy(polygon_vertices)
    print("Minimum X:", result["min_x"])
    print("Maximum X:", result["max_x"])
    print("Minimum Y:", result["min_y"])
    print("Maximum Y:", result["max_y"])
    """
    # Extract X and Y coordinates
    x_coords = vertices[0::3]  # Every third value starting from index 0
    y_coords = vertices[1::3]  # Every third value starting from index 1

    # Calculate minimum and maximum values
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)

    return {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y}


# def filter_cells_by_ids_with_existing_normals(polydata:vtk.vtkPolyData, cell_ids_to_keep):
#     """
#     Removes cells from vtkPolyData that are not in the specified list of cell IDs,
#     while preserving the existing cell normals.

#     Args:
#         polydata (vtk.vtkPolyData): The input vtkPolyData object with precomputed cell normals.
#         cell_ids_to_keep (list): List of cell IDs to retain in the polydata.

#     Returns:
#         vtk.vtkPolyData: A new vtkPolyData object with only the specified cells and their normals.
#     """
#     # Create a new vtkCellArray to hold the filtered cells
#     filtered_cells = vtk.vtkCellArray()

#     # Get the existing cell normals
#     cell_normals = polydata.GetCellData().GetNormals()
#     if not cell_normals:
#         raise ValueError("Input vtkPolyData does not contain precomputed cell normals.")

#     # New array to store normals of filtered cells
#     filtered_normals = vtk.vtkFloatArray()
#     filtered_normals.SetNumberOfComponents(3)
#     filtered_normals.SetName("Normals")

#     # Convert cell_ids_to_keep to a set for efficient lookups
#     cell_ids_to_keep_set = set(cell_ids_to_keep)

#     # Iterate directly over cell_ids_to_keep_set
#     for cell_id in sorted(cell_ids_to_keep_set):  # Ensure traversal order if necessary
#         cell = vtk.vtkIdList()
#         polydata.GetCell(cell_id, cell)  # Get the cell using its ID
#         filtered_cells.InsertNextCell(cell)
#         # Copy the normal for this cell
#         filtered_normals.InsertNextTuple(cell_normals.GetTuple(cell_id))

#     # Create a new vtkPolyData and assign the filtered cells and normals
#     new_polydata = vtk.vtkPolyData()
#     new_polydata.SetPoints(polydata.GetPoints())
#     new_polydata.SetPolys(filtered_cells)

#     # Add the filtered normals to the new polydata
#     new_polydata.GetCellData().SetNormals(filtered_normals)

#     # Copy other data arrays if needed
#     new_polydata.GetCellData().PassData(polydata.GetCellData())
#     new_polydata.GetPointData().PassData(polydata.GetPointData())

#     return new_polydata


import vtk

def filter_cells_by_ids_with_existing_normals(polydata, cell_ids_to_keep):
    """
    Removes cells from vtkPolyData that are not in the specified list of cell IDs,
    while preserving the existing cell normals.

    Args:
        polydata (vtk.vtkPolyData): The input vtkPolyData object with precomputed cell normals.
        cell_ids_to_keep (list): List of cell IDs to retain in the polydata.

    Returns:
        vtk.vtkPolyData: A new vtkPolyData object with only the specified cells and their normals.
    """
    # Create a new vtkCellArray to hold the filtered cells
    filtered_cells = vtk.vtkCellArray()

    # Get the existing cell normals
    cell_normals = polydata.GetCellData().GetNormals()
    if not cell_normals:
        raise ValueError("Input vtkPolyData does not contain precomputed cell normals.")

    # New array to store normals of filtered cells
    filtered_normals = vtk.vtkFloatArray()
    filtered_normals.SetNumberOfComponents(3)
    filtered_normals.SetName("Normals")

    # Convert cell_ids_to_keep to a set for efficient lookups
    cell_ids_to_keep_set = set(cell_ids_to_keep)

    # Iterate directly over cell_ids_to_keep_set
    for cell_id in sorted(cell_ids_to_keep_set):  # Ensure traversal order if necessary
        point_ids = vtk.vtkIdList()
        polydata.GetCellPoints(cell_id, point_ids)  # Get the point IDs for the cell
        filtered_cells.InsertNextCell(point_ids)
        # Copy the normal for this cell
        filtered_normals.InsertNextTuple(cell_normals.GetTuple(cell_id))

    # Create a new vtkPolyData and assign the filtered cells and normals
    new_polydata = vtk.vtkPolyData()
    new_polydata.SetPoints(polydata.GetPoints())
    new_polydata.SetPolys(filtered_cells)

    # Add the filtered normals to the new polydata
    new_polydata.GetCellData().SetNormals(filtered_normals)

    # Copy other data arrays if needed
    new_polydata.GetCellData().PassData(polydata.GetCellData())
    new_polydata.GetPointData().PassData(polydata.GetPointData())

    return new_polydata

def y_cutter(polydata:vtk.vtkPolyData, y_low, y_high):
    '''
    surface cutting berween two Y planes, without capped geometry
    '''

    # Define the first clipping plane
    plane1 = vtk.vtkPlane()
    plane1.SetOrigin(0, y_low, 0)  # Origin of the plane
    plane1.SetNormal(0, 1, 0)  # Normal vector pointing along +Y

    # Define the second clipping plane
    plane2 = vtk.vtkPlane()
    plane2.SetOrigin(0, y_high, 0)  # Origin of the plane
    plane2.SetNormal(0, -1, 0)  # Normal vector pointing along -Y

    # Apply the first clipping operation
    clipper1 = vtk.vtkClipPolyData()
    clipper1.SetInputData(polydata)  # Input is the sphere
    clipper1.SetClipFunction(plane1)  # Clipping with the first plane
    clipper1.GenerateClippedOutputOff()  # Do not generate capped geometry
    clipper1.Update()

    # Apply the second clipping operation
    clipper2 = vtk.vtkClipPolyData()
    clipper2.SetInputConnection(clipper1.GetOutputPort())  # Input is the result of the first clip
    clipper2.SetClipFunction(plane2)  # Clipping with the second plane
    clipper2.GenerateClippedOutputOff()  # Do not generate capped geometry
    clipper2.Update()
    
    output_polydata = vtk.vtkPolyData()
    output_polydata.DeepCopy(clipper2.GetOutput())
    return output_polydata


def get_true_y_range(polydata):
    """
    Get the true Y range of the cells in a vtkPolyData.
    
    Parameters:
        polydata (vtk.vtkPolyData): The input vtkPolyData to process.

    Returns:
        tuple: A tuple (y_min, y_max) representing the range in Y.
    """
    # Initialize variables to store the min and max Y values
    y_min = float("inf")
    y_max = float("-inf")

    # Access the points and cells of the vtkPolyData
    points = polydata.GetPoints()
    cells = polydata.GetPolys()

    # Traverse through all the cells
    cells.InitTraversal()
    id_list = vtk.vtkIdList()
    while cells.GetNextCell(id_list):
        for i in range(id_list.GetNumberOfIds()):
            point_id = id_list.GetId(i)  # Get the point index
            point = points.GetPoint(point_id)  # Get the coordinates of the point
            y_min = min(y_min, point[1])  # Update the minimum Y value
            y_max = max(y_max, point[1])  # Update the maximum Y value

    return y_min, y_max


import math

# Function to compute orientation table
def compute_orientation(data):
    """
    Computes the Y component and the angle in the XZ plane for a given dataset.
    Args:
        data (dict): Input dataset containing plane normals and points.
    Returns:
        dict: Output dictionary with Y normal magnitude and XZ angle.
        
    # Example dataset
    t_data = {
        25: {'normal_pca': [1.0004047533723864e-08, -0.9999999999999978, -6.757364368059626e-08], 'point_on_plane': [0.252583793166912, 2.1019559508622296, -0.2236007246931996]},
        28: {'normal_pca': [-0.6629634790570077, -0.7485959471192655, -0.00913966045531517], 'point_on_plane': [-3.052697952242865, 2.935328863669133, 0.19844222900228223]},
        0: {'normal_pca': [-0.37424936644193474, -0.9245446993402856, 0.07179492070886062], 'point_on_plane': [-2.095296850204468, 2.2510599581400554, 0.28993883311748503]},
        26: {'normal_pca': [0.6693587179699027, -0.7427257664107596, 0.017814112024965637], 'point_on_plane': [2.9483811602447973, 2.9258708773237285, 0.04445848122916438]},
        9: {'normal_pca': [-0.5685724421099556, -0.814140078179662, -0.11790382171386532], 'point_on_plane': [-2.390991238933621, 2.4868554798039524, -0.3950634581234419]}
    }

    Example code:
    
    # Compute orientations for t_data
    t_orientation = compute_orientation(t_data)

    # Print results
    for id_key, values in t_orientation.items():
        print(f"ID: {id_key}, Y Normal: {values['y normal']:.6f}, Angle XZ: {values['angle xz']:.6f}")
        
    """
    data_out = {}
    for key, value in data.items():
        normal_pca = value['normal_pca']
        
        # Compute the absolute value of the Y component of the normal vector
        y_normal = abs(normal_pca[1])

        # Project the normal vector onto the XZ plane and calculate the angle relative to the X-axis
        x, z = normal_pca[0], normal_pca[2]
        angle_xz = math.degrees(math.atan2(z, x))
        if angle_xz < 0:
            angle_xz += 360  # Ensure the angle is in the range 0-360°

        # Store the computed values in the output dictionary
        data_out[key] = {'y normal': y_normal, 'angle xz': angle_xz}

    # Sort the dictionary by the Y normal magnitude in ascending order
    sorted_data_out = dict(sorted(data_out.items(), key=lambda item: item[1]['y normal']))
    return sorted_data_out

def find_angular_matching(dictionary_1, dictionary_2, max_angular_difference=20):
    """
    Finds items in dictionary_2 for each item in dictionary_1 based on angular difference.
    
    Args:
        dictionary_1 (dict): First dictionary with 'angle xz' values.
        dictionary_2 (dict): Second dictionary with 'angle xz' values.
        max_angular_difference (float): Maximum allowed angular difference in degrees.

    Returns:
        dict: A dictionary where each key from dictionary_1 maps to a list of matching keys from dictionary_2.

    # Example usage
    dictionary_1 = {
    16: {'y normal': 0.9335582308755539, 'angle xz': 4.372493590518368},
    14: {'y normal': 0.937369264551424, 'angle xz': 49.092728685261804},
    27: {'y normal': 0.9429110063656305, 'angle xz': 93.43780052593965},
    }
    dictionary_2 = {
    3: {'y normal': 0.7543357115716794, 'angle xz': 2.988958540732672},
    7: {'y normal': 0.7532744164250463, 'angle xz': 23.168092041390928},
    19: {'y normal': 0.7500159067258075, 'angle xz': 37.26987230710107},
    6: {'y normal': 0.751519142548347, 'angle xz': 60.7353155646664},
    9: {'y normal': 0.7599202318840171, 'angle xz': 72.77038512329092},
    5: {'y normal': 0.7440275596216013, 'angle xz': 84.70583765406504},
    }
    result = find_angular_matching(dictionary_1, dictionary_2, max_angular_difference=20)
    print(result)
    """
    def angular_difference(angle1, angle2):
        """Compute the magnitude of the angular difference."""
        diff = abs(angle1 - angle2)
        return min(diff, 360 - diff)

    result = {}
    for key1, item1 in dictionary_1.items():
        result[key1] = [
            key2 for key2, item2 in dictionary_2.items()
            if angular_difference(item1['angle xz'], item2['angle xz']) < max_angular_difference
        ]
    return result


'''
dictionary_1 = {
16: {'y normal': 0.9335582308755539, 'angle xz': 4.372493590518368},
14: {'y normal': 0.937369264551424, 'angle xz': 49.092728685261804},
27: {'y normal': 0.9429110063656305, 'angle xz': 93.43780052593965},
10: {'y normal': 0.9434331718996396, 'angle xz': 137.66380218191827},
11: {'y normal': 0.9397818571794959, 'angle xz': 184.67626639710608},
12: {'y normal': 0.9369275159525225, 'angle xz': 229.9974738739},
31: {'y normal': 0.9401202546730377, 'angle xz': 274.34290454176596},
0: {'y normal': 0.9394131599679459, 'angle xz': 318.80081704906206}
}

dictionary_2 = {
3: {'y normal': 0.7543357115716794, 'angle xz': 2.988958540732672},
7: {'y normal': 0.7532744164250463, 'angle xz': 23.168092041390928},
19: {'y normal': 0.7500159067258075, 'angle xz': 37.26987230710107},
6: {'y normal': 0.751519142548347, 'angle xz': 60.7353155646664},
9: {'y normal': 0.7599202318840171, 'angle xz': 72.77038512329092},
5: {'y normal': 0.7440275596216013, 'angle xz': 84.70583765406504},
15: {'y normal': 0.744629901144194, 'angle xz': 107.28621941598819},
24: {'y normal': 0.7528854702489031, 'angle xz': 119.61781197213507},
10: {'y normal': 0.7468523817068006, 'angle xz': 125.87211796749098},
1: {'y normal': 0.7476828464163888, 'angle xz': 152.70321051402988},
14: {'y normal': 0.7459431381731003, 'angle xz': 171.28372883703867},
8: {'y normal': 0.7481482609809516, 'angle xz': 195.47562531517391},
20: {'y normal': 0.7574371477428472, 'angle xz': 203.4633171795014},
0: {'y normal': 0.744974591093767, 'angle xz': 217.4566976679637},
16: {'y normal': 0.746338765401153, 'angle xz': 240.94805726323517},
17: {'y normal': 0.75636902549275, 'angle xz': 252.67581655194607},
23: {'y normal': 0.7451390259328402, 'angle xz': 263.02720913668225},
13: {'y normal': 0.7515886246414339, 'angle xz': 272.59170738948126},
22: {'y normal': 0.7492572382059302, 'angle xz': 282.90047262462247},
4: {'y normal': 0.7464390236257363, 'angle xz': 287.8622794817344},
2: {'y normal': 0.7489187148033378, 'angle xz': 306.741268789673},
18: {'y normal': 0.7552181867345877, 'angle xz': 319.0622565689856},
12: {'y normal': 0.7504784019349836, 'angle xz': 334.73476052426236},
11: {'y normal': 0.7469775245864639, 'angle xz': 353.171244633299}
}

result = find_matching_items(dictionary_1, dictionary_2, max_angular_difference=20)
print(result)
'''



if __name__ == "__main__":
    example_check_convexity_and_ccw()
    pass

