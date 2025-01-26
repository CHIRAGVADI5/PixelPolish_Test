import os
import sys
import time
import math
import numpy as np
import vtk

from Pixel3D.pxl_tools import get_pp3d_data_path
from Pixel3D.pxl_vtk_tools import get_objects, visualizator, visualizator_two, save_as_ply, save_as_vtp, get_local_appdata_folder

def load_polygons(filename):
    '''
    carga un archivo de texto con información de múltiples polígonales definidas 
    como se muestra en este ejemplo, coordenadas en formato double, y regresa
    listas anidadas:
    w
    h
    pixel_size
    center_offset
    y_culette
    2
    3
    1 1 1
    2 2 2
    3 3 3
    2
    4 4 4
    5 5 5    

    eso es todo, explicación:
    este archivo contiene dos listas anidadas: 
    2: dos poligonales
    3: número de puntos de la primera poligonal
    1 1 1   coordenada
    2 2 2   coordenada
    3 3 3   coordenada
    2: número de puntos de la segunda poligonal
    4 4 4   coordenada
    5 5 5   coordenada   
    '''
    polygons = []

    with open(filename, 'r') as file:
        # Leer tamaño de la imagen
        w = float(file.readline().strip())
        h = float(file.readline().strip())

        # Leer tamaño de pixel en micras
        pixel_size = float(file.readline().strip())

        center_offset = float(file.readline().strip())
        y_culette = pixel_size * float(file.readline().strip())

        axes_size = h * pixel_size
        dx = w / 2.0 + center_offset

        x_min = sys.float_info.max
        y_min = sys.float_info.max
        x_max = sys.float_info.min
        y_max = sys.float_info.min
        
        # Leer el número total de poligonales
        num_polygons = int(file.readline().strip())
        
        for _ in range(num_polygons):
            # Leer el número de puntos en la poligonal
            num_points = int(file.readline().strip())
            
            polygon = []
            for _ in range(num_points):
                # Leer las coordenadas del punto
                point = list(map(float, file.readline().strip().split()))
                # point processing
                point[0] = point[0] - dx
                point = [x * pixel_size for x in point]
                # extreme values
                x_min = min(x_min, point[0])
                y_min = min(y_min, point[1])
                x_max = max(x_max, point[0])
                y_max = max(y_max, point[1])
                # point agregation
                polygon.append(point)
            
            polygons.append(polygon)
    
    return polygons, x_min, x_max, y_min, y_max, y_culette, pixel_size, center_offset

# def load_polygons_example(case_id:str=None):
#     data_path = get_pp3d_data_path()
#     if data_path is None:
#         return
#     if case_id==None:
#         case_id = PxlModeling.case_id
#     polygons_path = os.path.join(data_path, case_id + '.txt')
#     polygons, x_min, x_max, y_min, y_max, y_culette, pixel_size, center_offset = load_polygons(polygons_path)
#     pass

def prime_factorization(n):
    '''
    This method performs "n" factorization and returns 
    the factors in a list in ascending order.
    '''
    factors = []
    divisor = 2
    while n > 1:
        while n % divisor == 0:
            factors.append(divisor)
            n //= divisor
        divisor += 1
    factors.sort(reverse=False)  # Sort the list of factors in ascending order
    return factors

# def construct_solid_from_planes(points, normals, w=20):
def construct_solid_from_planes(points:vtk.vtkPoints, normals:vtk.vtkDoubleArray, w=40):
    
    # Plane initialization
    planes = vtk.vtkPlanes()
    planes.SetPoints(points)
    planes.SetNormals(normals)

    # Create a polyhedron defined by the planes
    polyhedron = vtk.vtkConvexPointSet()
    for i in range(points.GetNumberOfPoints()):
        polyhedron.GetPoints().InsertNextPoint(points.GetPoint(i))
    
    # Now create the polyhedron from the planes
    hull_filter = vtk.vtkHull()
    hull_filter.SetPlanes(planes)
    
    # Output the solid
    polydata = vtk.vtkPolyData()
    hull_filter.GenerateHull(polydata, -w, w, -w, w, 0, w)

    # Limpiar el polydata para unificar puntos coincidentes
    clean_filter = vtk.vtkCleanPolyData()
    clean_filter.SetInputData(polydata)
    clean_filter.Update()
    
    # Aplicar el filtro de conectividad para obtener las celdas conectadas
    connectivity_filter = vtk.vtkPolyDataConnectivityFilter()
    connectivity_filter.SetInputData(clean_filter.GetOutput())
    connectivity_filter.Update()
    # return polydata
    # Retornar el resultado con conectividad mejorada
    return connectivity_filter.GetOutput()

def calculate_normals(points):
    
    # Convert the list of points to a numpy array for easier manipulation
    points = np.array(points)
    normals = []

  # Convertir la lista de puntos a un array de NumPy para facilidad de operaciones
    points_ = np.array(points)
    
    if points_.shape == 1:
        pass 
    
    # Calcular el valor medio de las coordenadas x e y
    mx = np.mean(points_[:, 0])
    my = np.mean(points_[:, 1])
    
    num_points = len(points)
    inverted = 0

    for i in range(num_points):
        # Get the current point and the next point (with wrap-around)
        p1 = points[i]
        p2 = points[(i + 1) % num_points]
        
        vx = p1[0] - mx 
        vy = p1[1] - my 

        # Calculate the vector of the segment
        segment = p2 - p1

        # Calculate the normal vector in the XY plane
        normal = np.array([segment[1], -segment[0], 0])

        if np.linalg.norm(normal) == 0:
            pass
        
        # Normalize the vector to get a unit vector
        unit_normal = normal / np.linalg.norm(normal)
        
        if vx*unit_normal[0] + vy*unit_normal[1] < 0:
            inverted += 1
            unit_normal = -unit_normal

        # Add the normal vector to the list of normals
        normals.append(unit_normal)

    return list(zip(points, normals)), mx, my, inverted


def cutting_cube_by_polygonals(polygon_data:list, pixel_size, mode=2, max_index=-1, index=-1):
    '''
    filepath: path to polygons file.
    mode options:
        1: traverse the polygons in a hierarchical manner by coarser angle increments to finer angle increments.
        2. traverse the polygons in ascending order from angle zero to last angle. 
    max_index: maximum angle index (until 199 for 400 silhouettes or 399 for 800 silhouettes)
    '''
    
    polygons, x_min, x_max, y_min, y_max, y_culette, pixel_size, center_offset = polygon_data

    D = 2.5 * max(x_max, y_max)
    
    # se ontienes los dos objetos para iniciar el proceso de tallado usando planos de corte.
    surface, _ = get_objects(D)
    _cube = vtk.vtkPolyData()
    _cube.DeepCopy(surface)

    # len(polygons) is the total number of polygons in a range of pi radians
    N = len(polygons)
    if max_index==-1:
        max_index = N-1
    factors = prime_factorization(N)

    factor = math.pi/180.0
    Q = N # polygons sampling, each Q polygons
    ilist = []

    __normals = vtk.vtkDoubleArray()
    __normals.SetNumberOfComponents(3)
    __points = vtk.vtkPoints()

    exist_culette = False

    i = -1
    for F in factors:
        Q = Q / F  #  

        if mode==2 and i == N: 
            break
        
        for k in range(0, N, int(Q)):
            
            if mode==1:            
                if k in ilist:
                    continue
                ilist.append(k)
                i =  k
            else:
                i += 1
                if i == N: 
                    break
            
            if i > max_index:
                continue
            
            if i == 196:
                pass

            # return zip file with points and normals to each point
            zipped_info, mx, my, _ = calculate_normals(polygons[i])        
            _np = 0
            angle = i*math.pi/N

            c = math.cos(-angle)
            s = math.sin(-angle)
            pass

            # Recorrer los puntos cada M puntos
            for origin, normal in zipped_info:
                _np += 1
                # Calcular las nuevas coordenadas
                x_new = c * origin[0] +  s * origin[2]
                y_new = origin[1]  # same Y
                z_new = - s * origin[0] + c * origin[2]
                
                _origin = (x_new, y_new, z_new)
            
                nx_new = c * normal[0] +  s * normal[2]
                ny_new = normal[1]  # El valor de y no cambia
                nz_new = - s * normal[0] + c * normal[2]

                if ny_new > 0.9:
                    # for large value of Y component of normal
                    exist_culette = True
                
                _normal = (nx_new, ny_new, nz_new)
                
                # if _normal[1] < 0:
                #     _angle = np.degrees(np.atan(np.abs(_normal[0]/_normal[1])))
                #     if _angle < 10:
                #         _origin[1] = _origin[1] - 10*pixel_size

                __points.InsertNextPoint(tuple(_origin))
                __normals.InsertNextTuple(_normal)

                # surface = plane_clipper(surface, _origin, _normal, index_value = 0)


    bounds = [0] * 6
    __points.GetBounds(bounds)

    if True:
        _y_max = y_max
        if exist_culette:
            pass
            # y_culette -= 0.080
            # y_culette -= 2
            # y_culette -= 0.010
        else:
            pass
            # y_culette -= 0.080
            # y_culette -= 0.4  <<<<<<
            # y_culette -= 0.3
            # y_culette -= 0.8   # no se ve dominio, es redondeada
            # y_culette -= 1.5   # bueno para ver puntos de interes de cometas del pavellón

        if False:
            __points.InsertNextPoint((0, y_culette, 0))
            __normals.InsertNextTuple((0.0,1.0,0.0))

        # __points.InsertNextPoint((0.95* bounds[1], 0, 0))
        # __normals.InsertNextTuple((1.0,0.0,0.0))

    surface = construct_solid_from_planes(__points, __normals, w=40)
    
    surface.GetBounds(bounds)
    return surface, _cube

    # Imprimir los polígonos cargados para verificar
    # for points in polygons:
    #     # return zip file with points and normals to each point
    #     zipped_info = calculate_normals(points)
        
    #     # Recorrer los puntos cada M puntos
    #     for i in range(0, len(zipped_info), M):
    #         point, normal = zipped_info[i]
    #         pass
    #     pass

def execute_1(case_id:str=None, mode=2, max_index=-1, prefix="", poly_data:list=None, index=-1):

    data_path = get_pp3d_data_path()
    # path = get_local_appdata_folder()

    if data_path is None:
        return
    if case_id==None:
        return

    # if case_id==None:
    #     case_id = PxlModeling.case_id

    if poly_data is None:
        polygons_path = os.path.join(data_path, prefix + case_id + '.txt')
        polygons_data = load_polygons(polygons_path)
        polygons, x_min, x_max, y_min, y_max, y_culette, pixel_size, center_offset = polygons_data
    else:
        polygons_data = poly_data
        polygons, x_min, x_max, y_min, y_max, y_culette, pixel_size, center_offset = polygons_data
        

    surface, cube = cutting_cube_by_polygonals(polygons_data, pixel_size, mode, max_index, index=index)

    bounds = [0] * 6
    surface.GetBounds(bounds)
    
    # surface = Grouping(surface)
    surface.GetBounds(bounds)

    # visualizator_two(surface=surface)
    # render_window_2, interactor_2 = visualizator(surface, None, add_axes=False)
    # render_window_2.Render()
    # interactor_2.Start()
    return surface, len(polygons), pixel_size, center_offset


if __name__ == "__main__":
    # load_polygons_example()
    execute_1(case_id=None, mode=2, max_index=-1)
    
    



