


import os
import math
import numpy as np
import vtk
from numpy.linalg import norm
from PIL import Image, ImageFilter
from collections import defaultdict

global total_number_of_border_isolated_points  # inserted
global Y_CAMERA  # inserted
total_number_of_border_isolated_points = 0


def get_objects(size=1):
    """\n    retorna un cubo y un cilindro para probar el tallado (carving)\n    por recorte usando planos para usar en diamantes pulidos.\n    """  # inserted
    cube = vtk.vtkCubeSource()
    cube.SetXLength(2 * size)
    cube.SetYLength(2 * size)
    cube.SetZLength(2 * size)
    cube.Update()
    _cube = vtk.vtkPolyData()
    _cube.DeepCopy(cube.GetOutput())
    cylinder = vtk.vtkCylinderSource()
    cylinder.SetRadius(0.5 * size)
    cylinder.SetHeight(2 * size)
    cylinder.SetResolution(10)
    cylinder.SetCapping(False)
    cylinder.Update()
    transform = vtk.vtkTransform()
    transform.Translate(0, 0, 5)
    transform.RotateX(90)
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputConnection(cylinder.GetOutputPort())
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    transform2 = vtk.vtkTransform()
    transform2.RotateWXYZ(90, 0, 0, 1)
    transformFilter2 = vtk.vtkTransformPolyDataFilter()
    transformFilter2.SetTransform(transform2)
    transformFilter2.SetInputConnection(transform_filter.GetOutputPort())
    transformFilter2.Update()
    _cylinder = vtk.vtkPolyData()
    _cylinder.DeepCopy(transform_filter.GetOutput())
    return (_cube, _cylinder)

def save_as_vtp(polydata, path):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(polydata)
    writer.Write()

def read_as_vtp(path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    polydata = reader.GetOutput()
    return polydata

def read_as_ply(path):
    reader = vtk.vtkPLYReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()

def read_as_stl(path):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()

def save_as_stl(polydata, path):
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(path)
    writer.SetInputData(polydata)
    writer.Write()

def save_as_ply(polydata, path):
    writer = vtk.vtkPLYWriter()
    writer.SetFileName(path)
    writer.SetInputData(polydata)
    writer.Write()

def save_as_obj(polydata, path):
    writer = vtk.vtkOBJWriter()
    writer.SetFileName(path)
    writer.SetInputData(polydata)
    writer.Write()

def get_local_appdata_folder():
    appdata_local = os.getenv('LOCALAPPDATA')
    if appdata_local:
        return appdata_local
    print('No se pudo obtener la ruta de AppData\\Local.')


Y_CAMERA = (-20)
pass
pass
def visualizator(surface: vtk.vtkPolyData, cylinder: vtk.vtkPolyData=None, add_axes=False, external_actor=None, actor_list=None):
    """\n    \n    """  # inserted
    lut_azimuth = vtk.vtkLookupTable()
    lut_azimuth.SetNumberOfTableValues(201)
    lut_azimuth.Build()
    N2 = 180
    lut_elevation = vtk.vtkLookupTable()
    lut_elevation.SetNumberOfTableValues(N2 + 1)
    lut_elevation.Build()
    lut_group = vtk.vtkLookupTable()
    lut_group.SetNumberOfTableValues(133)
    lut_group.Build()
    resistance_colors = [(0.2, 0.2, 0.2), (0.6, 0.3, 0), (1, 0, 0), (1, 0.5, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (0.9176470588235294, 0.2, 0.7372549019607844), (0.6, 0.6, 0.6), (1, 1, 1)]
    # color_palette_33 = [(0.2, 0.2, 0.2), (0.4, 0.4, 0.4), (0.8, 0.8, 0.8), (1.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.5), (1.0, 1.0, 0.0), (0.5, 0.5, 0.0), (1.0, 0.0, 1.0), (0.5, 0.0, 0.5), (0.0, 1.0, 1.0), (0.0, 0.5, 0.5), (0.5, 1.0, 0.5), (1.0, 0.75, 0.8), (0.75, 0.75, 0.75), (1.0, 0.85, 0.7), (0.6, 0.3, 0), (0.5, 0.5, 1.0), (1.0, 0.5, 0.75), (0.5, 0.0, 0.25), (0.75, 1.0
    color_palette_51 = [(1, 0, 0), (0.8, 0, 0), (0.6, 0, 0), (0, 1, 0), (0, 0.8, 0), (0, 0.6, 0), (0, 0, 1), (0, 0, 0.8), (0, 0, 0.6), (1, 1, 0), (0.8, 0.8, 0), (0.6, 0.6, 0), (1, 0.65, 0), (0.8, 0.33, 0.56), (0.6, 0.25, 0.42), (0.5, 0, 0.5), (0.4, 0, 0.4), (0.3, 0, 0.3), (1, 0, 1), (0.8, 0, 0.8), (0.6, 0, 0.6), (0.2, 0.8, 0.2), (0.16, 0.64, 0.16)]

    color_palette_66 = [
    (0.06, 0.93, 0.68), (0.47, 0.99, 0.18), (0.99, 0.08, 0.93),
    (0.78, 0.94, 0.74), (0.62, 0.38, 0.10), (0.23, 0.77, 0.36),
    (0.89, 0.46, 0.30), (0.15, 0.92, 0.60), (0.80, 0.07, 0.74),
    (0.66, 0.88, 0.52), (0.25, 0.45, 0.80), (0.56, 0.72, 0.43),
    (0.97, 0.14, 0.28), (0.31, 0.15, 0.86), (0.39, 0.57, 0.92),
    (0.87, 0.64, 0.12), (0.68, 0.34, 0.87), (0.41, 0.80, 0.05),
    (0.93, 0.88, 0.28), (0.20, 0.66, 0.53), (0.73, 0.04, 0.57),
    (0.50, 0.88, 0.34), (0.96, 0.36, 0.75), (0.26, 0.53, 0.31),
    (0.45, 0.23, 0.77), (0.59, 0.95, 0.43), (0.13, 0.84, 0.29),
    (0.82, 0.10, 0.35), (0.91, 0.31, 0.60), (0.75, 0.56, 0.19),
    (0.37, 0.90, 0.46), (0.69, 0.15, 0.92), (0.53, 0.32, 0.20),
    (0.77, 0.52, 0.92), (0.04, 0.61, 0.79), (0.60, 0.29, 0.58),
    (0.24, 0.96, 0.82), (0.48, 0.40, 0.13), (0.11, 0.72, 0.50),
    (0.83, 0.49, 0.04), (0.65, 0.93, 0.09), (0.91, 0.19, 0.43),
    (0.20, 0.81, 0.69), (0.54, 0.53, 0.98), (0.31, 0.64, 0.10),
    (0.98, 0.11, 0.67), (0.72, 0.77, 0.22), (0.56, 0.24, 0.66),
    (0.88, 0.58, 0.47), (0.14, 0.42, 0.95), (0.63, 0.21, 0.35),
    (0.42, 0.86, 0.74), (0.70, 0.05, 0.16), (0.78, 0.38, 0.89),
    (0.30, 0.25, 0.63), (0.16, 0.57, 0.41), (0.89, 0.08, 0.99),
    (0.48, 0.92, 0.26), (0.10, 0.76, 0.22), (0.51, 0.49, 0.80),
    (0.95, 0.07, 0.53), (0.35, 0.71, 0.11), (0.17, 0.61, 0.90),
    (0.62, 0.41, 0.21), (0.75, 0.28, 0.69), (0.68, 0.47, 0.44)
]

    
    for i in range(200):
        color = resistance_colors[i % 10]
        lut_azimuth.SetTableValue(i, color[0], color[1], color[2], 1.0)
    lut_azimuth.SetTableValue(200, 0.7, 0.7, 1.0, 1.0)
    for i in range(N2):
        color = resistance_colors[i % 10]
        lut_elevation.SetTableValue(i, color[0], color[1], color[2], 1.0)
    lut_elevation.SetTableValue(N2, 0, 0.6235294117647059, 0.3137254901960784, 1.0)
    for i in range(132):
        color = color_palette_66[i % 66]
        lut_group.SetTableValue(i, color[0], color[1], color[2], 1.0)
    lut_group.SetTableValue(132, 1.0, 1.0, 1.0, 1.0)
    lut_group.SetTableValue(131, 0.7, 0.7, 0.7, 1.0)
    ncells = surface.GetNumberOfCells()
    bounds = [0] * 6
    surface.GetBounds(bounds)
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    cube_mapper = vtk.vtkPolyDataMapper()
    cube_mapper.SetLookupTable(lut_elevation)
    cube_mapper.SetScalarRange(0, 180)
    cube_mapper.SetScalarModeToUseCellData()
    cube_mapper.SetInputData(surface)
    surface.GetCellData().SetActiveScalars('Elevation')
    cube_mapper.ScalarVisibilityOn()
    s1 = surface.GetCellData().GetScalars('Elevation')
    s2 = surface.GetCellData().GetScalars('Azimuth')
    cube_actor = vtk.vtkActor()
    cube_actor.SetMapper(cube_mapper)
    property = cube_actor.GetProperty()
    property.BackfaceCullingOn()
    property.SetInterpolationToFlat()
    property.SetLineWidth(1)
    cylinder_mapper = vtk.vtkPolyDataMapper()
    cylinder_mapper.SetInputData(cylinder)
    cylinder_actor = vtk.vtkActor()
    cylinder_actor.SetMapper(cylinder_mapper)
    property = cylinder_actor.GetProperty()
    property.SetColor(0, 0, 0)
    property.SetLineWidth(3)
    cylinder_actor.GetProperty().SetRepresentationToWireframe()
    pass
    axes_actor = vtk.vtkAxesActor()
    axes_size = 2 * abs(xmax)
    axes_actor.SetTotalLength(axes_size, axes_size, axes_size)
    axes_actor.AxisLabelsOff()
    axes_actor.SetConeRadius(0.1)
    axes_actor.SetNormalizedTipLength(0, 0, 0)
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.1, 0.2, 0.4)
    renderer.SetBackground(0.8, 0.8, 0.8)
    renderer.SetBackground(0.0, 0.0, 0.0)
    renderer.AddActor(cube_actor)
    if add_axes:
        renderer.AddActor(axes_actor)
    if cylinder is not None:
        renderer.AddActor(cylinder_actor)
    camera = renderer.GetActiveCamera()
    camera.ParallelProjectionOn()
    camera.SetPosition(0, Y_CAMERA, 0)
    camera.SetFocalPoint(0, 0, 0)
    camera.SetViewUp(0, 0, 1)
    renderer.SetActiveCamera(camera)
    renderer.ResetCamera()
    renderer.ResetCameraClippingRange()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)
    opt1 = True
    text_actors_1 = []
    text_actors_2 = []
    text_actors_3 = []
    text_actors_4 = []
    text_actors_5 = []
    text_actors_6 = []
    text_actors_7 = []
    pass
    def create_billboard_text(text, position):
        text_actor = vtk.vtkBillboardTextActor3D()
        text_actor.SetPosition(position)
        text_actor.SetInput(text)
        text_actor.GetTextProperty().SetFontSize(12)
        text_actor.GetTextProperty().BoldOn()
        text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
        text_actor.VisibilityOff()
        return text_actor
    cell_centers = vtk.vtkCellCenters()
    cell_centers.SetInputData(surface)
    cell_centers.Update()
    cell_centers = cell_centers.GetOutput()
    cell_normals = surface.GetCellData().GetNormals()
    group_array = surface.GetCellData().GetArray('Group')
    group_array_2 = surface.GetCellData().GetArray('Normal Grouped')
    offset = 0.07
    group_exist = group_array is not None
    group_exist_2 = group_array_2 is not None
    n_cells = surface.GetNumberOfCells()
    for cell_id in range(n_cells):
        center = cell_centers.GetPoint(cell_id)
        normal = cell_normals.GetTuple(cell_id)
        if group_exist:
            id = group_array.GetValue(cell_id)
            allowing_label = id < 100
        else:  # inserted
            id = cell_id
            allowing_label = abs(normal[2]) >= 0.01
        id2 = (-1)
        if group_exist_2:
            id2 = group_array_2.GetValue(cell_id)
        if allowing_label:
            adjusted_center = [center[j] + offset * normal[j] for j in range(3)]
            text_actor = create_billboard_text(str(id), adjusted_center)
            text_actors_1.append(text_actor)
            renderer.AddActor(text_actor)
            if id2 >= 0:
                text_actor = create_billboard_text(str(id2), adjusted_center)
                text_actors_2.append(text_actor)
                renderer.AddActor(text_actor)
    pass
    def create_billboard_text(text, position):
        text_actor = vtk.vtkBillboardTextActor3D()
        text_actor.SetPosition(position)
        text_actor.SetInput(text)
        text_actor.GetTextProperty().SetFontSize(12)
        text_actor.GetTextProperty().BoldOn()
        text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
        text_actor.VisibilityOff()
        return text_actor
    centers = vtk.vtkCellCenters()
    centers.SetInputData(surface)
    centers.Update()
    centers = centers.GetOutput()
    normals = surface.GetCellData().GetNormals()
    azimuth_array = surface.GetCellData().GetArray('Azimuth')
    offset = 0.007
    n = surface.GetNumberOfCells()
    for id in range(n):
        center = cell_mass_center(surface, id)
        normal = normals.GetTuple(id)
        valor = azimuth_array.GetValue(id)
        adjusted_center = [center[j] + offset * normal[j] for j in range(3)]
        text_actor = create_billboard_text(str(valor), adjusted_center)
        text_actors_3.append(text_actor)
        renderer.AddActor(text_actor)
    pass
    def create_billboard_text(text, position):
        text_actor = vtk.vtkBillboardTextActor3D()
        text_actor.SetPosition(position)
        text_actor.SetInput(text)
        text_actor.GetTextProperty().SetFontSize(12)
        text_actor.GetTextProperty().BoldOn()
        text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
        text_actor.VisibilityOff()
        return text_actor
    centers = vtk.vtkCellCenters()
    centers.SetInputData(surface)
    centers.Update()
    centers = centers.GetOutput()
    normals = surface.GetCellData().GetNormals()
    elevation_array = surface.GetCellData().GetArray('Elevation')
    offset = 0.02
    n = surface.GetNumberOfCells()
    for id in range(n):
        center = cell_mass_center(surface, id)
        normal = normals.GetTuple(id)
        valor = round(elevation_array.GetValue(id), 1)
        adjusted_center = [center[j] + offset * normal[j] for j in range(3)]
        text_actor = create_billboard_text(str(valor), adjusted_center)
        text_actors_5.append(text_actor)
        renderer.AddActor(text_actor)
    pass
    def create_billboard_text(text, position):
        text_actor = vtk.vtkBillboardTextActor3D()
        text_actor.SetPosition(position)
        text_actor.SetInput(text)
        text_actor.GetTextProperty().SetFontSize(16)
        text_actor.GetTextProperty().BoldOn()
        text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
        text_actor.VisibilityOff()
        return text_actor
    point_centers = surface.GetPoints()
    point_normals = surface.GetPointData().GetNormals()
    offset = 0.003
    n_points = surface.GetNumberOfPoints()
    node_diedral_array = surface.GetPointData().GetArray('Node Diedral')
    node_solid_array = surface.GetPointData().GetArray('Node Solid')
    y_table = bounds[2]
    for point_id in range(n_points):
        center = point_centers.GetPoint(point_id)
        normal = point_normals.GetTuple(point_id)
        weight = node_diedral_array.GetValue(point_id)
        adjusted_center = [center[j] + offset * normal[j] for j in range(3)]
        label = str(weight) if weight > 1 else ''
        text_actor = create_billboard_text(label, adjusted_center)
        text_actors_4.append(text_actor)
        renderer.AddActor(text_actor)
        point = surface.GetPoint(point_id)
        label = str(round(point[1] - y_table, 3))
        text_actor = create_billboard_text(label, adjusted_center)
        text_actors_7.append(text_actor)
        renderer.AddActor(text_actor)
        diedral_angle = node_solid_array.GetValue(point_id)
        label = str(round(diedral_angle, 1))
        text_actor = create_billboard_text(label, adjusted_center)
        text_actors_6.append(text_actor)
        renderer.AddActor(text_actor)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    style_camera = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style_camera)
    textActor = vtk.vtkTextActor()
    label_color_option = 0
    g_text_on = 0
    h_text_on = 0
    v_text_on = 0
    o_text_on = 0
    b_text_on = 0
    k_text_on = 0
    j_text_on = 0
    last_key = '-'

    def keypress_callback(obj, event):
        """\n        
        x: scalar off\n        v: azimuth\n             b: elevation\n        
        g: azimuth grouping\n  h: notmal grouping\n     o: point spatial angle (average azimuth of the facets)\n        
        k: diedral angle (average diedral angles betwen planes)\n        j: y coordinate\n        y: text color\n        
        u: camera position (upper, lower)\n        """  # inserted
        
        nonlocal g_text_on  # inserted
        nonlocal o_text_on  # inserted
        nonlocal k_text_on  # inserted
        nonlocal v_text_on  # inserted
        nonlocal j_text_on  # inserted
        nonlocal b_text_on  # inserted
        nonlocal h_text_on  # inserted
        nonlocal last_key  # inserted
        nonlocal label_color_option  # inserted
        global Y_CAMERA  # inserted
        has_azimuth = False
        has_elevation = False
        has_group = False
        has_normal_grouped = False
        has_weights = False
        has_diedral = False
        print('I am here!!')
        n_arrays = surface.GetCellData().GetNumberOfArrays()
        if n_arrays > 0:
            has_azimuth = surface.GetCellData().GetScalars('Azimuth')!= None
            has_elevation = surface.GetCellData().GetScalars('Elevation')!= None
            has_group = surface.GetCellData().GetScalars('Group')!= None
            has_normal_grouped = surface.GetCellData().GetScalars('Normal Grouped')!= None
            has_weights = surface.GetPointData().GetArray('Node Diedral')!= None
            has_diedral = surface.GetPointData().GetArray('Node Solid')!= None
        key = obj.GetKeySym()
        if key.lower() == 'x':                          # scalar off
            cube_mapper.ScalarVisibilityOff()
            g_text_on = 0
            h_text_on = 0
            v_text_on = 0
            o_text_on = 0
            b_text_on = 0
            k_text_on = 0
            j_text_on = 0
        else:  # inserted
            if key.lower() == 'v':                      # azimuth
                if has_azimuth:
                    surface.GetCellData().SetActiveScalars('Azimuth')
                    cube_mapper.ScalarVisibilityOn()
                    cube_mapper.SetLookupTable(lut_azimuth)
                    cube_mapper.SetScalarRange(0, 200)
                    v_text_on = 1 - v_text_on
                    g_text_on = 0
                    h_text_on = 0
                    o_text_on = 0
                    b_text_on = 0
                    k_text_on = 0
                    j_text_on = 0
            else:  # inserted
                if key.lower() == 'b':                  # elevation
                    if has_elevation:
                        surface.GetCellData().SetActiveScalars('Elevation')
                        cube_mapper.ScalarVisibilityOn()
                        cube_mapper.SetLookupTable(lut_elevation)
                        cube_mapper.SetScalarRange(0, 180)
                        b_text_on = 1 - b_text_on
                        g_text_on = 0
                        h_text_on = 0
                        v_text_on = 0
                        o_text_on = 0
                        k_text_on = 0
                        j_text_on = 0
                else:  # inserted
                    if key.lower() == 'g':              # azimuth grouping
                        if has_group:
                            surface.GetCellData().SetActiveScalars('Group')
                            cube_mapper.ScalarVisibilityOn()
                            cube_mapper.SetLookupTable(lut_group)
                            cube_mapper.SetScalarRange(0, 132)
                            if last_key == 'g':
                                g_text_on = 1 - g_text_on
                            else:  # inserted
                                g_text_on = h_text_on
                            h_text_on = 0
                            o_text_on = 0
                            v_text_on = 0
                            b_text_on = 0
                            k_text_on = 0
                            j_text_on = 0
                            last_key = key
                    else:  # inserted
                        if key.lower() == 'h':          # normal grouping
                            if has_normal_grouped:
                                surface.GetCellData().SetActiveScalars('Normal Grouped')
                                cube_mapper.ScalarVisibilityOn()
                                cube_mapper.SetLookupTable(lut_group)
                                cube_mapper.SetScalarRange(0, 132)
                                if last_key == 'h':
                                    h_text_on = 1 - h_text_on
                                else:  # inserted
                                    h_text_on = g_text_on
                                g_text_on = 0
                                v_text_on = 0
                                o_text_on = 0
                                b_text_on = 0
                                k_text_on = 0
                                j_text_on = 0
                                last_key = key
                        else:  # inserted
                            if key.lower() == 'o':      # point spatial angle (average azimuth of the facets)
                                if has_weights:
                                    g_text_on = 0
                                    h_text_on = 0
                                    o_text_on = 1 - o_text_on
                                    v_text_on = 0
                                    b_text_on = 0
                                    k_text_on = 0
                                    j_text_on = 0
                            else:  # inserted
                                if key.lower() == 'j':  # y coordinate
                                    g_text_on = 0
                                    h_text_on = 0
                                    o_text_on = 0
                                    v_text_on = 0
                                    b_text_on = 0
                                    k_text_on = 0
                                    j_text_on = 1 - j_text_on
                                else:  # inserted
                                    if key.lower() == 'k':          # diedral angle (average diedral angles betwen planes)
                                        if has_diedral:
                                            g_text_on = 0
                                            h_text_on = 0
                                            o_text_on = 0
                                            v_text_on = 0
                                            k_text_on = 1 - k_text_on
                                            b_text_on = 0
                                            j_text_on = 0
                                    else:  # inserted
                                        if key.lower() == 'y':      # text color
                                            if len(text_actors_1) > 0:
                                                label_color_option = (label_color_option + 1) % 2
                                                if label_color_option == 0:
                                                    for text_actor in text_actors_1:
                                                        text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
                                                    for text_actor in text_actors_2:
                                                        text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
                                                    for text_actor in text_actors_3:
                                                        text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
                                                    for text_actor in text_actors_4:
                                                        text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
                                                    for text_actor in text_actors_5:
                                                        text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
                                                    for text_actor in text_actors_6:
                                                        text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
                                                    for text_actor in text_actors_7:
                                                        text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
                                                    textActor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
                                                else:  # inserted
                                                    if label_color_option == 1:
                                                        for text_actor in text_actors_1:
                                                            text_actor.GetTextProperty().SetColor(0.0, 0.0, 0.0)
                                                        for text_actor in text_actors_2:
                                                            text_actor.GetTextProperty().SetColor(0.0, 0.0, 0.0)
                                                        for text_actor in text_actors_3:
                                                            text_actor.GetTextProperty().SetColor(0.0, 0.0, 0.0)
                                                        for text_actor in text_actors_4:
                                                            text_actor.GetTextProperty().SetColor(0.0, 0.0, 0.0)
                                                        for text_actor in text_actors_5:
                                                            text_actor.GetTextProperty().SetColor(0.0, 0.0, 0.0)
                                                        for text_actor in text_actors_6:
                                                            text_actor.GetTextProperty().SetColor(0.0, 0.0, 0.0)
                                                        for text_actor in text_actors_7:
                                                            text_actor.GetTextProperty().SetColor(0.0, 0.0, 0.0)
                                                        textActor.GetTextProperty().SetColor(0.0, 0.0, 0.0)
                                        else:  # inserted
                                            if key.lower() == 'u':          # camera position (upper, lower)
                                                Y_CAMERA = -Y_CAMERA
                                                camera.SetPosition(0, Y_CAMERA, 0)
                                                camera.SetFocalPoint(0, 0, 0)
                                                camera.SetViewUp(0, 0, 1)
                                                renderer.ResetCameraClippingRange()
        if len(text_actors_1) > 0 and g_text_on!= text_actors_1[0].GetVisibility():
            for text_actor in text_actors_1:
                text_actor.SetVisibility(g_text_on)
        if len(text_actors_2) > 0 and h_text_on!= text_actors_2[0].GetVisibility():
            for text_actor in text_actors_2:
                text_actor.SetVisibility(h_text_on)
        if len(text_actors_3) > 0 and v_text_on!= text_actors_3[0].GetVisibility():
            for text_actor in text_actors_3:
                text_actor.SetVisibility(v_text_on)
        if len(text_actors_4) > 0 and o_text_on!= text_actors_4[0].GetVisibility():
            for text_actor in text_actors_4:
                text_actor.SetVisibility(o_text_on)
        if len(text_actors_5) > 0 and b_text_on!= text_actors_5[0].GetVisibility():
            for text_actor in text_actors_5:
                text_actor.SetVisibility(b_text_on)
        if len(text_actors_6) > 0 and k_text_on!= text_actors_6[0].GetVisibility():
            for text_actor in text_actors_6:
                text_actor.SetVisibility(k_text_on)
        if len(text_actors_7) > 0 and j_text_on!= text_actors_7[0].GetVisibility():
            for text_actor in text_actors_7:
                text_actor.SetVisibility(j_text_on)
        surface.Modified()
        cube_mapper.Modified()
        render_window.Render()
    interactor.AddObserver('KeyPressEvent', keypress_callback)
    cylinder_actor.GetProperty().SetColor(0, 0, 0)
    cylinder_mapper.ScalarVisibilityOff()
    pass
    scalarBar = vtk.vtkScalarBarActor()
    scalarBar.SetNumberOfLabels(2)
    lookupTable = vtk.vtkLookupTable()
    lookupTable.SetNumberOfTableValues(1)
    lookupTable.SetTableValue(0, 0.5, 0.5, 0.5, 1.0)
    lookupTable.Build()
    scalarBar.SetLookupTable(vtk.vtkLookupTable())
    scalarBar.GetLookupTable().SetRange(0, 2)
    scalarBar.GetLookupTable().SetAlpha(1)
    scalarBar.GetLabelTextProperty().SetColor(1, 1, 0)
    scalarBar.SetLookupTable(lookupTable)
    scalarBar.SetPosition(0.9, 0.05)
    scalarBar.SetWidth(0.05)
    scalarBar.SetHeight(0.5)
    scalarBar.GetLabelTextProperty().SetFontSize(100)
    textActor.SetInput('Here')
    textActor.GetTextProperty().SetFontSize(20)
    textActor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
    textActor.GetTextProperty().BoldOn()
    textActor.SetPosition(0.2, 0.2)
    renderer.AddActor2D(scalarBar)
    renderer.AddActor2D(textActor)

    def UpdateScalarBar(renderer, scalarBar, textActor):
        camera = renderer.GetActiveCamera()
        parallelScale = camera.GetParallelScale()
        windowHeight = parallelScale * 2
        scalarBar.SetMaximumNumberOfColors(1)
        scalarBar.GetLookupTable().SetRange(0, windowHeight * 0.49)
        scalarBar.Modified()
        range = scalarBar.GetLookupTable().GetRange()
        textActor.SetInput(str(round(range[1], 3)))

    def RenderCallback(caller, event):
        UpdateScalarBar(renderer, scalarBar, textActor)
    render_window.AddObserver('RenderEvent', RenderCallback)
    pass
    if external_actor!= None:
        renderer.AddActor(external_actor)
    if actor_list!= None:
        for act in actor_list:
            renderer.AddActor(act)
    return (render_window, interactor)


def get_planes(polyData: vtk.vtkPolyData):
    """\n    retorna los planos de una superficie como una lista de pares [origin, normal]\n    """  # inserted
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polyData)
    normals.ComputeCellNormalsOn()
    normals.Update()
    polyData = normals.GetOutput()
    points = polyData.GetPoints()
    cellNormals = normals.GetOutput().GetCellData().GetNormals()
    planes_info = []
    for i in range(polyData.GetNumberOfCells()):
        cell = polyData.GetCell(i)
        cellNormal = cellNormals.GetTuple(i)
        cellPointIds = cell.GetPointIds()
        pointId = cellPointIds.GetId(0)
        point = points.GetPoint(pointId)
        planes_info.append([point, cellNormal])
    return planes_info

def construct_solid_from_planes(points: vtk.vtkPoints, normals: vtk.vtkDoubleArray, w=20):
    planes = vtk.vtkPlanes()
    planes.SetPoints(points)
    planes.SetNormals(normals)
    polyhedron = vtk.vtkConvexPointSet()
    for i in range(points.GetNumberOfPoints()):
        polyhedron.GetPoints().InsertNextPoint(points.GetPoint(i))
    hull_filter = vtk.vtkHull()
    hull_filter.SetPlanes(planes)
    polydata = vtk.vtkPolyData()
    hull_filter.GenerateHull(polydata, -w, w, -w, w, 0, w)
    clean_filter = vtk.vtkCleanPolyData()
    clean_filter.SetInputData(polydata)
    clean_filter.Update()
    connectivity_filter = vtk.vtkPolyDataConnectivityFilter()
    connectivity_filter.SetInputData(clean_filter.GetOutput())
    connectivity_filter.Update()
    return connectivity_filter.GetOutput()

def visualizator_two(surface: vtk.vtkPolyData, cylinder: vtk.vtkPolyData=None):
    bounds = [0] * 6
    surface.GetBounds(bounds)
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    cube_mapper = vtk.vtkPolyDataMapper()
    cube_mapper.SetInputData(surface)
    cube_actor = vtk.vtkActor()
    cube_actor.SetMapper(cube_mapper)
    cube_actor.GetProperty().BackfaceCullingOn()
    cube_actor.GetProperty().SetInterpolationToFlat()
    cube_actor.GetProperty().SetColor(1, 0, 0)
    cylinder_mapper = vtk.vtkPolyDataMapper()
    cylinder_mapper.SetInputData(cylinder)
    cylinder_actor = vtk.vtkActor()
    cylinder_actor.SetMapper(cylinder_mapper)
    property = cylinder_actor.GetProperty()
    property.SetColor(0, 0, 1)
    pass
    axes_actor = vtk.vtkAxesActor()
    axes_actor.SetTotalLength(2 * xmax, 2 * xmax, 2 * xmax)
    axes_actor.AxisLabelsOff()
    axes_actor.SetConeRadius(0.1)
    axes_actor.SetNormalizedTipLength(0, 0, 0)
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.1, 0.2, 0.4)
    renderer.AddActor(cube_actor)
    renderer.AddActor(axes_actor)
    if cylinder!= None:
        renderer.AddActor(cylinder_actor)
    camera = renderer.GetActiveCamera()
    camera.ParallelProjectionOn()
    renderer.ResetCamera()
    renderer.ResetCameraClippingRange()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1936, 1216)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    style_camera = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style_camera)
    interactor.Initialize()
    interactor.Start()

def move(_polydata: vtk.vtkPolyData, x=0.0, y=0.0, z=0.0):
    transform = vtk.vtkTransform()
    transform.Translate(x, y, z)
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(_polydata)
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    __polydata = vtk.vtkPolyData()
    __polydata.DeepCopy(transform_filter.GetOutput())
    return __polydata

def rotate(_polydata: vtk.vtkPolyData, x_angle=0.0, y_angle=0.0, z_angle=0.0):
    transform = vtk.vtkTransform()
    if x_angle!= 0:
        transform.RotateX(x_angle)
    if y_angle!= 0:
        transform.RotateY(y_angle)
    if z_angle!= 0:
        transform.RotateZ(z_angle)
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(_polydata)
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    __polydata = vtk.vtkPolyData()
    __polydata.DeepCopy(transform_filter.GetOutput())
    return __polydata

def scale(_polydata: vtk.vtkPolyData, factor=1.0):
    transform = vtk.vtkTransform()
    transform.Scale(factor, factor, factor)
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(_polydata)
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    __polydata = vtk.vtkPolyData()
    __polydata.DeepCopy(transform_filter.GetOutput())
    return __polydata


def cell_mass_center(polyData, cell_index):
    cell = polyData.GetCell(cell_index)
    num_points = cell.GetNumberOfPoints()
    x_sum = 0.0
    y_sum = 0.0
    z_sum = 0.0
    for j in range(num_points):
        point_id = cell.GetPointId(j)
        point = polyData.GetPoint(point_id)
        x_sum += point[0]
        y_sum += point[1]
        z_sum += point[2]
    centro_x = x_sum / num_points
    centro_y = y_sum / num_points
    centro_z = z_sum / num_points
    return (centro_x, centro_y, centro_z)



