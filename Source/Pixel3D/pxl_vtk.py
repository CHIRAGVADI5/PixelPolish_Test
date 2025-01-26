
import sys
import os
import time
import gc # garbage collector
import math
import copy
from enum import Enum
import numpy as np
import cv2

from PySide6.QtWidgets import QApplication, QMainWindow, QFrame, QVBoxLayout
# from PySide6.QtOpenGLWidgets import QOpenGLWidget
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

import vtk
from vtk.util.numpy_support import vtk_to_numpy

from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor
)

import Pixel3D.pxl_tools
from Pixel3D.pxl_dll import get_cleaned_points_from_decimated, \
    save_polygons, transfer_fast_3d_model_arrays, transfer_precise_3d_model

from Pixel3D.pxl_3d_fast_model import execute_1, construct_solid_from_planes
from Pixel3D.pxl_girdle import GirdleHandler
from Pixel3D.pxl_facet_clustering import *
from Pixel3D.pxl_points import check_convexity_and_ccw, remove_non_convex, save_polygons_HR, get_min_max_xy, \
    filter_cells_by_ids_with_existing_normals, y_cutter, get_true_y_range, compute_orientation,\
        find_angular_matching
from Pixel3D.pxl_tools import get_pp3d_data_path, read_vector_from_file, \
    adjust_points_to_same_plane, tri_points_angle, cv2_processing, using_cv2_coordinates

enable_print = False

RED = 0
GREEN = 1
BLUE = 2
YELLOW = 3
MAGENTA = 4
CYAM = 5
GRAY = 6
WHITE = 7

EDGE = 0
DECIMATED = 1
SEGMENTS = 2
CLEANED = 3
SURFACE = 4
TABLE_SURFACE = 5
GIRDLE_SURFACE = 6
PAVILION_SURFACE = 7
_TABLE_SURFACE = 8
_PAVILION_SURFACE = 9

SURFACE_POSITION = 0
EDGE_POSITION = 1
DECIMATED_POSITION = 2
SEGMENTS_POSITION = 3
CLEANED_POSITION = 4

class State(Enum):
    INITIAL = 0
    LOADED = 1 
    IMAGING = 2
    FAST_MODEL = 3
    PRECISE_MODEL = 4
    
class ActorPipeline():
    colors = [
        (1.0, 0.0, 0.0),  # Red
        (0.0, 1.0, 0.0),  # Green
        (0.5, 0.5, 1.0),  # Blue
        (1.0, 1.0, 0.0),  # Yellow
        (1.0, 0.0, 1.0),  # Magenta
        (0.0, 1.0, 1.0),  # Cyan
        (0.5, 0.5, 0.5),  # Gray
        (1.0, 1.0, 1.0)]  # White

    def __init__(self, renderer:vtk.vtkRenderer):

        self.mapper:vtk.vtkPolyDataMapper = None
        self.actor:vtk.vtkActor = None
        self.label_actors = [] 
        self.polydata:vtk.vtkPolyData = None
        self.renderer = renderer
        
    def build(self, line_width = 3, selected_color=WHITE, z_position=0):

        self.mapper = vtk.vtkPolyDataMapper()
        self.actor = vtk.vtkActor()

        self.actor.GetProperty().SetColor(__class__.colors[selected_color])
        self.actor.GetProperty().SetLineWidth(line_width)

        self.actor.SetMapper(self.mapper)
        self.actor.AddPosition(0.0, 0.0, z_position)
        self.actor.VisibilityOff()

        # Add the actors to the first_renderer
        self.renderer.AddActor(self.actor)

    def visibility(self, vis:bool):
        self.actor.SetVisibility(vis and 1 or 0)
        for actor in self.label_actors:
            actor.SetVisibility(vis and 1 or 0)
        
    def labels_visibility(self, vis:bool):
        for actor in self.label_actors:
            actor.SetVisibility(vis and 1 or 0)

    def set_input(self, _polydata:vtk.vtkPolyData):
        self.mapper.SetInputData(_polydata)
        self.polydata = _polydata

    def single_color_mapping(self, color=(0.7, 0.7, 1.0)):
        
        # Surface visibility settings
        self.mapper.ScalarVisibilityOff()
        # Flat visualization settings
        self.actor.GetProperty().BackfaceCullingOff()  # Enable backface culling
        self.actor.GetProperty().SetInterpolationToFlat()  # Set flat interpolation
        # Disable specular shading
        self.actor.GetProperty().SetSpecular(0.0)  # No specular shading
        self.actor.GetProperty().SetColor(color)  # Set to selected color
        self.actor.GetProperty().SetDiffuse(1.0)  # Full diffuse shading
        # Edge visualization settings
        self.actor.GetProperty().EdgeVisibilityOn()  # Enable edge visibility
        self.actor.GetProperty().SetEdgeColor(1.0, 1.0, 1.0)  # Set edge color to white
        # self.actor.GetProperty().SetEdgeColor(0.0, 0.0, 0.0)  # Set edge color to white
        self.actor.GetProperty().SetLineWidth(1.0)  # Set line width to 1.0

    def edge_visibility(self, value:bool=True):
        self.actor.GetProperty().SetEdgeVisibility(value and 1 or 0)  # Enable edge visibility
        
    def multi_color_mapping(self, _polydata, table_faces):

        self.mapper.SetInputData(_polydata)
        self.polydata = _polydata
        # self.actor.GetProperty().EdgeVisibilityOff()  # Disable edge visibility
        self.actor.GetProperty().EdgeVisibilityOn()  # Disable edge visibility
        self.actor.GetProperty().BackfaceCullingOn()  # Enable backface culling

        # Lookup Table for coloring
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(table_faces)  # Assuming 33 groups
        lut.SetRange(0,table_faces-1)
        lut.Build()
        
        # Generar una lista de enteros del 0 al 32
        integer_list = list(range(table_faces))  # Crea [0, 1, 2, ..., 32]

        # Generate random colors for each group
        np.random.seed(int(time.time()))
        
        # random.seed()
        for i in range(table_faces):
            lut.SetTableValue(i, np.random.rand(), np.random.rand(), np.random.rand(), 1.0)
        
        if True:
            self.polydata.GetCellData().SetActiveScalars("FacetGroups")
            self.mapper.SetScalarModeToUseCellData()
            self.mapper.SelectColorArray("FacetGroups")  ##
            self.mapper.SetColorModeToMapScalars()
            self.mapper.ScalarVisibilityOn()
            self.mapper.SetLookupTable(lut)
            self.mapper.UseLookupTableScalarRangeOn()
            self.mapper.SetUseLookupTableScalarRange(True)  ##   
            self.mapper.SetScalarRange(0,table_faces-1)
        else:
            self.mapper.ScalarVisibilityOff()
            self.actor.GetProperty().SetColor(1.0,0.0,0.0)  
            back_property = vtk.vtkProperty()
            back_property.SetColor(0.0,0.0,1.0)  # Back face color
            self.actor.SetBackfaceProperty(back_property)

    def add_label_actors(self, offset=0.18):

        if len(self.label_actors) > 0:
            for actor in self.label_actors:
                self.renderer.RemoveActor(actor)
            self.label_actors.clear()
                
        # Manage texts assigned to clustered cells
        label_actors = add_group_labels_to_surface(self.polydata, offset)
        # actor:vtk.vtkActor = None
        
        for actor in label_actors:
            actor.VisibilityOff()
            self.renderer.AddActor(actor)
            self.label_actors.append(actor)
                
# *************************************************************************

class PxlModeling():
    
    iso = vtk.vtkMarchingSquares()
    connectivity_filter = vtk.vtkPolyDataConnectivityFilter()
    edge_decimator = vtk.vtkDecimatePolylineFilter()
    case_id = "LHPO_Round_Polished_400_0002_11p7_2024-08-03-dll"
    pixel_size = 0.01172
    _image_number = 0
    using_dll = True   # PENDIENTE

    def __init__(self, widget, layout):

        # Create a QVTKRenderWindowInteractor to render the VTK scene
        self.vtk_widget = QVTKRenderWindowInteractor(widget)
        layout.addWidget(self.vtk_widget)

        self.prefix = "HR_"
        self.x_mins = []
        self.x_maxs = []
        self.y_mins = []
        self.y_maxs = []

        self.aps = []
        self.code_id = None
        
        self.gray_images = []       # opencv images (type np.ndarray)        
        self.gray_images_cut = []   # opencv images (type np.ndarray)        
        self.images = []            # vtk images (type vtkImageData)        
        self.image_qtty = 400
        # self.pixel_size = 0.01172
        self.center_offset = 0
        
        self.edges = []             # vtk edges from images (list of vtkPolyData)
        self.decimated_edges = []   # decimated versión of edges (list of vtkPolyData)
        self.decimated_arrays = []  # list of flatted list of points (3D)
        # self.segments = []        # enabled segments of decimated_edges
        self.cleaned = []           # fimal points for polydataCleaned (list of vtkPolyData)

        self.fast_surface:vtk.vtkPolyData = None        
        self.precise_surface:vtk.vtkPolyData = None    

        self.bounds2D = []
        # Create a global vtkOBBTree object
        self.obb_tree:vtk.vtkOBBTree = None

        # Dictionary to store the facets (cells) associated with each vertex
        self.vertice_facetas = {}
        self.faceta_vertices = {}
        
        # Fast model
        self.points_array = []      # verts
        self.normals_array = []     # cellnormals
        self.cell_type_array = []   # cell type

        # Positioning of fast model, required to get secure pavilion locations. 
        self.table_center = None
        self.minAngle = 0
        
        # Precise model, it require secure pavillion locations
        self.cells = []                 # polygonal cells
        self.pavilion_locations = None  # tuplas (node_id, y_coordinate, amgle)

        # self.polyDataEdge:vtk.vtkPolyData = None
        # self.polyDataDecimated:vtk.vtkPolyData = None
        # self.polyDataSegments:vtk.vtkPolyData = None
        # self.polydataCleaned:vtk.vtkPolyData = None
        
        # all mappers
        self.surfaceMapper:vtk.vtkPolyDataMapper = None
        self.edgeMapper:vtk.vtkPolyDataMapper = None
        self.decimatedMapper:vtk.vtkPolyDataMapper = None
        self.segmentsMapper:vtk.vtkPolyDataMapper = None
        self.cleanedMapper:vtk.vtkPolyDataMapper = None

        # all actors 
        self.surfaceActor:vtk.vtkActor = None
        self.edgeActor:vtk.vtkActor = None
        self.decimatedActor:vtk.vtkActor = None
        self.segmentsActor:vtk.vtkActor = None
        self.cleanedActor:vtk.vtkActor = None
        self.imageActor:vtk.vtkImageActor = None
        self.coordinateAxesActor:vtk.vtkActor = None 

        # image texture
        self.texture:vtk.vtkTexture = None
        # vtk render window
        self.render_window = None
        self.first_renderer:vtk.vtkRenderer = None
        self.second_renderer:vtk.vtkRenderer = None
        self.labels_renderer:vtk.vtkRenderer = None
        self.interactor:vtk.vtkRenderWindowInteractor = None
        
        self.vector_text:vtk.vtkVectorText = None
        self.text_actor:vtk.vtkActor = None

        self.index = 0
        self.first_time = True
        # self.bounds2D

        # Configure connectivity filter to extract the largest region
        __class__.connectivity_filter.SetInputConnection(__class__.iso.GetOutputPort())
        __class__.connectivity_filter.SetExtractionModeToLargestRegion()

        # Set up the VTK scene
        self.setup_vtk_scene()
        self.girdle_handler = GirdleHandler()

        self.xmin = 0
        self.xmax = 0
        self.ymin = 0
        self.ymax = 0
        self.center_offset = 0
        self.y_table = 0
        self.y_culette = 0
        self.cleaned_polygons = []
        self.w = 0
        self.h = 0
        self.all_contour_points = []
        self.girdle_spheres_actor:vtk.vtkActor = None
        self.state = State.INITIAL

        # Create and assign the key press callback
        # self.key_press_callback = KeyPressCallback.New()
        # self.key_press_callback.pxl_visualization = self  # Link the PxlModeling instance

    def setup_vtk_scene(self):
        
        # Get the render window from QVTKRenderWindowInteractor
        self.render_window = self.vtk_widget.GetRenderWindow()
        
        # add two renderers        
        self.add_renderers()

        # Initialize the mappers
        self.surfaceMapper = vtk.vtkPolyDataMapper()
        self.edgeMapper = vtk.vtkPolyDataMapper()
        self.decimatedMapper = vtk.vtkPolyDataMapper()
        self.segmentsMapper = vtk.vtkPolyDataMapper()
        self.cleanedMapper = vtk.vtkPolyDataMapper()

        # Initialize the actors
        self.surfaceActor = vtk.vtkActor()
        self.edgeActor = vtk.vtkActor()
        self.decimatedActor = vtk.vtkActor()
        self.segmentsActor = vtk.vtkActor()
        self.cleanedActor = vtk.vtkActor()
        self.imageActor = vtk.vtkImageActor()

        # self.edgeActor.GetProperty().SetColor(0.0, 1.0, 0.0)
        # self.decimatedActor.GetProperty().SetColor(1.0, 0.0, 0.0)
        # self.segmentsActor.GetProperty().SetColor(0.0, 0.0, 1.0)
        # self.cleanedActor.GetProperty().SetColor(1.0, 0.0, 0.0)
        
        self.edgeActor.GetProperty().SetColor(1.0, 1.0, 0.0)        # Yellow
        self.decimatedActor.GetProperty().SetColor(0.5, 0.1, 1.0)   # Blue
        self.segmentsActor.GetProperty().SetColor(1.0, 1.0, 1.0)    # White
        self.cleanedActor.GetProperty().SetColor(1.0, 0.0, 0.0)     # Red

        # Surface visibility settings
        self.surfaceMapper.ScalarVisibilityOff()
        # Flat visualization settings
        self.surfaceActor.GetProperty().BackfaceCullingOff()  # Enable backface culling
        self.surfaceActor.GetProperty().SetInterpolationToFlat()  # Set flat interpolation
        # Disable specular shading
        self.surfaceActor.GetProperty().SetSpecular(0.0)  # No specular shading
        self.surfaceActor.GetProperty().SetColor(0.7, 0.7, 1.0)  # Set color to light blue
        self.surfaceActor.GetProperty().SetDiffuse(1.0)  # Full diffuse shading
        # Edge visualization settings
        self.surfaceActor.GetProperty().EdgeVisibilityOn()  # Enable edge visibility
        self.surfaceActor.GetProperty().SetEdgeColor(1.0, 1.0, 1.0)  # Set edge color to white
        # self.surfaceActor.GetProperty().SetEdgeColor(0.0, 0.0, 0.0)  # Set edge color to white
        self.surfaceActor.GetProperty().SetLineWidth(1.0)  # Set line width to 1.0

        self.coordinateAxesActor = Pixel3D.pxl_tools.create_coordinate_axes(1)
        self.coordinateAxesActor.SetScale(5)
        self.coordinateAxesActor.GetProperty().SetLineWidth(3)
        self.coordinateAxesActor.VisibilityOn()
        
        # Disable interpolation to achieve a pixelated display
        self.imageActor.InterpolateOff()
        # self.imageActor.GetProperty().SetInterpolationTypeToNearest()
        
        # self.edgeActor.GetProperty().SetLineWidth(6)
        # self.decimatedActor.GetProperty().SetLineWidth(4)
        # self.segmentsActor.GetProperty().SetLineWidth(2)
        # self.cleanedActor.GetProperty().SetLineWidth(1)

        lw = 3
        self.edgeActor.GetProperty().SetLineWidth(lw)
        self.decimatedActor.GetProperty().SetLineWidth(lw)
        self.segmentsActor.GetProperty().SetLineWidth(lw)
        self.cleanedActor.GetProperty().SetLineWidth(lw)

        a = 1
        self.edgeActor.AddPosition(0.0, 0.0, a)
        self.decimatedActor.AddPosition(0.0, 0.0, 2*a)
        self.segmentsActor.AddPosition(0.0, 0.0, 4*a)
        self.cleanedActor.AddPosition(0.0, 0.0, 3*a)
        
        self.surfaceActor.VisibilityOff()
        self.edgeActor.VisibilityOff()
        self.segmentsActor.VisibilityOff()
        self.cleanedActor.VisibilityOff()
        self.imageActor.VisibilityOff()

        # Set the actor's mapper 
        self.surfaceActor.SetMapper(self.surfaceMapper)
        self.edgeActor.SetMapper(self.edgeMapper)
        self.decimatedActor.SetMapper(self.decimatedMapper)
        self.segmentsActor.SetMapper(self.segmentsMapper)
        self.cleanedActor.SetMapper(self.cleanedMapper)

        # Add the actors to the first_renderer
        self.first_renderer.AddActor(self.surfaceActor)
        self.first_renderer.AddActor(self.edgeActor)
        self.first_renderer.AddActor(self.decimatedActor)
        self.first_renderer.AddActor(self.segmentsActor)
        self.first_renderer.AddActor(self.cleanedActor)
        self.first_renderer.AddActor(self.imageActor)
        self.first_renderer.AddActor(self.coordinateAxesActor)
 
        # self.imageActor.GetProperty().SetInterpolationTypeToNearest()
        # self.imageActor.GetMapper().SetInterpolationModeToNearest()

        # self.imageActor.GetMapper().SetColorWindow(255)
        # self.imageActor.GetMapper().SetColorLevel(127.5)

        self.setup_more_surfaces()
        
        # Render the scene
        self.render_window.Render()

        # Set up the interactor style and initialize it
        self.interactor = self.render_window.GetInteractor()
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        # Add the KeyPressEvent observer and link it to the class method
        self.interactor.AddObserver(vtk.vtkCommand.KeyPressEvent, self.on_key_press)

        self.interactor.Initialize()

    def setup_more_surfaces(self):
        
        # # Get the render window from QVTKRenderWindowInteractor
        # self.render_window = self.vtk_widget.GetRenderWindow()
        
        # # add two renderers        
        # self.add_renderers()

        '''
        EDGE = 0
        DECIMATED = 1
        SEGMENTS = 2
        CLEANED = 3
        SURFACE = 4
        TABLE_SURFACE = 5
        GIRDLE_SURFACE = 6
        PAVILION_SURFACE = 7
        '''

        self.aps = [ActorPipeline(self.first_renderer) for _ in range(10)]
        
        lw = 5
        self.aps[EDGE].build(line_width=lw, selected_color=YELLOW, z_position=1)
        self.aps[DECIMATED].build(line_width=lw, selected_color=BLUE, z_position=2)
        self.aps[SEGMENTS].build(line_width=lw, selected_color=WHITE, z_position=3)
        self.aps[CLEANED].build(line_width=lw, selected_color=RED, z_position=4)
        self.aps[SURFACE].build(line_width=1, selected_color=WHITE, z_position=0)
        self.aps[TABLE_SURFACE].build(line_width=1, selected_color=WHITE, z_position=0)
        self.aps[GIRDLE_SURFACE].build(line_width=1, selected_color=WHITE, z_position=0)
        self.aps[PAVILION_SURFACE].build(line_width=1, selected_color=WHITE, z_position=0)
        self.aps[_TABLE_SURFACE].build(line_width=1, selected_color=WHITE, z_position=0)
        self.aps[_PAVILION_SURFACE].build(line_width=1, selected_color=WHITE, z_position=0)
                
        self.aps[SURFACE].single_color_mapping()
        self.aps[TABLE_SURFACE].single_color_mapping(color=ActorPipeline.colors[YELLOW])
        self.aps[GIRDLE_SURFACE].single_color_mapping(color=ActorPipeline.colors[RED])
        self.aps[PAVILION_SURFACE].single_color_mapping(color=ActorPipeline.colors[GREEN])
        self.aps[_TABLE_SURFACE].single_color_mapping(color=ActorPipeline.colors[YELLOW])
        self.aps[_PAVILION_SURFACE].single_color_mapping(color=ActorPipeline.colors[GREEN])

        if False:
            self.imageActor = vtk.vtkImageActor()
            # Disable interpolation to achieve a pixelated display
            self.imageActor.InterpolateOff()
            # self.imageActor.GetProperty().SetInterpolationTypeToNearest()
            self.imageActor.VisibilityOff()
            self.first_renderer.AddActor(self.imageActor)

            self.coordinateAxesActor = Pixel3D.pxl_tools.create_coordinate_axes(1)
            self.coordinateAxesActor.SetScale(4)
            self.coordinateAxesActor.GetProperty().SetLineWidth(3)
            self.coordinateAxesActor.VisibilityOn()
            self.first_renderer.AddActor(self.coordinateAxesActor)

        if False:
            # Render the scene
            self.render_window.Render()

            # Set up the interactor style and initialize it
            self.interactor = self.render_window.GetInteractor()
            self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

            # Add the KeyPressEvent observer and link it to the class method
            self.interactor.AddObserver(vtk.vtkCommand.KeyPressEvent, self.on_key_press)

            self.interactor.Initialize()

    def actors_off(self):
        for _aps in self.aps:
            _aps.visibility(False)
        self.imageActor.VisibilityOff()
        if self.text_actor is not None:
            self.text_actor.VisibilityOff()
        self.surfaceActor.VisibilityOff()
        self.coordinateAxesActor.VisibilityOff()
        if self.girdle_spheres_actor is not None:
            self.girdle_spheres_actor.VisibilityOff()

    def add_renderers(self):
        """
        Set up a second rendering pipeline in the render window.
        
        :param ren_win: vtkRenderWindow instance to which the second renderer will be added.
        """
        
        if self.render_window is None:
            return
 
        if self.first_renderer is not None or self.second_renderer is not None:
            return
        
        # Create a first_renderer
        self.first_renderer = vtk.vtkRenderer()
        # self.first_renderer.SetBackground(0.1, 0.2, 0.4)  # Set background color (RGB)
        a = 0.0
        self.first_renderer.SetBackground(a, a, a)
        self.render_window.AddRenderer(self.first_renderer)

        camera = self.first_renderer.GetActiveCamera()
        camera.ParallelProjectionOn()
        xFocal, yFocal, cameraDistance = 0, 0, 100
        camera.SetFocalPoint(xFocal, yFocal, 0.0)
        camera.SetPosition(xFocal, yFocal, cameraDistance)
        camera.SetViewUp(0, 1, 0)

        # Create the second renderer
        self.second_renderer = vtk.vtkRenderer()
        self.render_window.AddRenderer(self.second_renderer)

        # Configure the second renderer
        self.second_renderer.SetInteractive(0)
        self.second_renderer.SetLayer(1)

        # Ensure the render window supports multiple layers
        if self.render_window.GetNumberOfLayers() < 2:
            self.render_window.SetNumberOfLayers(2)

        # Configure the second camera with parallel projection
        camera2 = self.second_renderer.GetActiveCamera()
        camera2.ParallelProjectionOn()
        camera2.SetClippingRange(0.1, 200)  # Adjust clipping range
        camera2.SetParallelScale(500)  # Parallel scale for 1000 visible units (500 is half)
        camera2.SetPosition(500, 500, 100)  # Position the camera to view the scene from above
        camera2.SetFocalPoint(500, 500, 0)  # Focal point at the center of the scene
        camera2.SetViewUp(0.0, 1.0, 0.0)  # Keep the "up" direction aligned with Y-axis

        # Adjust the clipping range for the scene scale
        self.second_renderer.ResetCameraClippingRange()


    def set_surface(self, surface:vtk.vtkPolyData):
        
        if surface is None:
            # Create a cube source
            cube = vtk.vtkCubeSource()
            cube.SetXLength(2.0)
            cube.SetYLength(3.0)
            cube.SetZLength(4.0)
            cube.Update()
            poly = vtk.vtkPolyData()
            self.bounds2D = cube.GetOutput().GetBounds()
            self.bounds2D = [value * 3 for value in self.bounds2D]
            self.surfaceMapper.SetInputConnection(cube.GetOutputPort())
            
        else:
            self.surfaceMapper.SetInputData(surface)
            if self.state == State.FAST_MODEL:
                self.surfaceActor.GetProperty().EdgeVisibilityOn() 
            elif self.state == State.PRECISE_MODEL:
                self.surfaceActor.GetProperty().EdgeVisibilityOn() 
                

        self.actors_off()
        self.surfaceActor.VisibilityOn()
        self.coordinateAxesActor.VisibilityOff()
        # self.edgeActor.VisibilityOff()
        # self.decimatedActor.VisibilityOff()
        # self.segmentsActor.VisibilityOff()
        # self.imageActor.VisibilityOff()
        # self.cleanedActor.VisibilityOff()
        
        self.first_renderer.ResetCamera()
        self.render_window.Render()
            
        # texture.SetInputData(vtk_image)

    # def set_image(self, image:vtk.vtkImageData):
    #     dims = image.GetDimensions()   
    #     self.imageActor.GetMapper().SetInputData(image)
    #     # self.texture.SetInputData(image)
    #     self.surfaceActor.VisibilityOff()
    #     self.imageActor.VisibilityOn()

    def on_key_press(self, caller, event):
        """
        Callback for KeyPressEvent. Handles key presses in the VTK interactor.

        :param caller: The vtkRenderWindowInteractor that triggered the event.
        :param event: The type of event (should be 'KeyPressEvent').
        """
        # Get the interactor and the key pressed
        interactor = caller
        key = interactor.GetKeySym()
        # print(f"Key pressed: {key}")

        if key == "s":  # Reset the camera
            self.ResetCamera()

        if key == "r":  # Reset the camera
            self.ResetCameraObject()

        elif key == "m" or key == 'n':  
            n = len(self.images) / 2
            if key == 'm':
                self.index += 1
                if self.index >= n:
                    self.index = 0
            elif key == 'n':
                self.index -= 1
                if self.index < 0:
                    self.index = n-1
            self.select_camera(int(self.index))
            pass

        # Re-render the scene
        # self.render_window.Render()
        
    # select input image for processing
    def select_camera(self, number):
        
        if self.state != State.IMAGING:
            return

        self.imageActor.GetMapper().SetInputData(self.images[number])
        self.aps[EDGE].set_input(self.edges[number])
        self.aps[DECIMATED].set_input(self.decimated_edges[number])
        if __class__.using_dll:
            self.aps[CLEANED].set_input(self.cleaned[number])

        # self.segmentsMapper.SetInputData(self.segments[number])
        self.add_text(label=str(number), scale_text=40)

        # set visibility
        self.actors_off()
        self.aps[EDGE].visibility(True)
        self.aps[DECIMATED].visibility(True)
        if __class__.using_dll:
            self.aps[CLEANED].visibility(True)
        
        self.imageActor.VisibilityOn()
        self.text_actor.VisibilityOn()

        # set some values
        self.index = number
        self.bounds2D = self.decimated_edges[self.index].GetBounds()
        self.bounds2D = self.aps[EDGE].polydata.GetBounds()
        
        if self.first_time == True:
            self.first_renderer.ResetCamera()
            self.first_time = False
        self.render_window.Render()

        
    # select input image for processing
    def select_camera_OLD(self, number):

        self.imageActor.GetMapper().SetInputData(self.images[number])
        self.edgeMapper.SetInputData(self.edges[number])
        self.decimatedMapper.SetInputData(self.decimated_edges[number])
        self.cleanedMapper.SetInputData(self.cleaned[number])
        # self.segmentsMapper.SetInputData(self.segments[number])
        self.add_text(label=str(number), scale_text=40)

        # set visibility
        self.edgeActor.VisibilityOn()
        self.decimatedActor.VisibilityOn()
        self.segmentsActor.VisibilityOff()
        self.imageActor.VisibilityOn()
        self.cleanedActor.VisibilityOn()
        self.surfaceActor.VisibilityOff()
        # set some values
        self.index = number
        self.bounds2D = self.decimated_edges[self.index].GetBounds()
        
        # self.imageActor.GetMapper().SetInterpolationModeToNearest()
        # self.imageActor.GetProperty().SetInterpolationTypeToNearest()

        # Create a vtkTexture and set the image data
        # texture = vtk.vtkTexture()
        # texture.SetInputData(self.images[number])
        # Disable interpolation to make the texture appear pixelated
        # texture.InterpolateOff()
        # Set the texture to the actor
        # self.imageActor.SetTexture(texture)

        # Access the texture associated with the actor
        # Disable interpolation to ensure a pixelated appearance
        # self.imageActor.GetTexture().SetInterpolate(False)

        if self.first_time == True:
            self.first_renderer.ResetCamera()
            self.first_time = False
        self.render_window.Render()

        # vtkSmartPointer<vtkPolyData> polyDataSegments;
        # vtkSmartPointer<vtkPolyData> polydataCleaned;
        # int n = (int)this->cameras.size();
        # LinesProcessing(this->cameras[number].vtk_decimated_edges, polyDataSegments,
        #     polydataCleaned, this->renderWindow, number % (n/2), false);

    def ResetCamera(self):
        return
        if self.first_renderer is None:
            return        
        self.first_renderer.ResetCamera()
        camera = self.first_renderer.GetActiveCamera()     
        camera.ParallelProjectionOn() 
        xFocal, yFocal, cameraDistance = 0, 0, 100
        camera.SetFocalPoint(xFocal, yFocal, 0.0)
        camera.SetPosition(xFocal, yFocal, cameraDistance)
        camera.SetViewUp(0, 1, 0)
    
    def ResetCameraObject(self):
        if self.first_renderer is None:
            return        
        # self.first_renderer.ResetCamera()
        camera = self.first_renderer.GetActiveCamera()     
        camera.SetParallelScale(0.5*(self.bounds2D[3] - self.bounds2D[2]))  # Parallel scale: half fo height 
        xFocal, yFocal, cameraDistance = (self.bounds2D[0] + self.bounds2D[1])/2.0, (self.bounds2D[2] + self.bounds2D[3])/2.0, 100
        # print("Foco:", xFocal, yFocal)
        # print("Scale:", 0.5*(self.bounds2D[3] - self.bounds2D[2]))
        # print("Bounds", self.bounds2D)
        camera.SetFocalPoint(xFocal, yFocal, 0.0)
        camera.SetPosition(xFocal, yFocal, cameraDistance)
        camera.SetViewUp(0, 1, 0)
        self.first_renderer.ResetCameraClippingRange()
        camera.Modified()
        # self.render_window.Render()

    def free_memory(self):
        # Release memory for images stored in self.images
        for vtk_image in self.images:
            # num = sys.getrefcount(vtk_image)
            vtk_image = None  # Remove the reference to each vtkImageData
        self.images.clear()  # Clear the list of accumulated images
        self.index = 0

        for edge_polydata in self.edges:             # vtk edges from images (list of vtkPolyData)
            edge_polydata = None
        self.edges.clear()
        
        for edge_polydata in self.decimated_edges:   # decimated versión of edges (list of vtkPolyData)
            edge_polydata = None
        self.decimated_edges.clear()
                
        for _array in self.decimated_arrays:  # list of flatted list of points (3D)
            _array.clear()
        self.decimated_arrays.clear()
        
        for edge_polydata in self.cleaned:   # final points for polydataCleaned (list of vtkPolyData)
            edge_polydata = None
        self.cleaned.clear()

        for poly in self.cleaned_polygons:
            poly.clear()
        self.cleaned_polygons.clear()

        # all actors go non vivible
        self.actors_off()
        
        if self.girdle_spheres_actor is not None:
            self.first_renderer.RemoveActor(self.girdle_spheres_actor)
            self.girdle_spheres_actor = None
        
        for actor_pipeline in self.aps:
            actor_pipeline.set_input(None)
            
        # self.set_surface(None)
        # self.ResetCamera()    
        # self.ResetCameraObject()    
        gc.collect()         # Force garbage collection to free memory

    @staticmethod
    def get_parent_of_images_folder(file_path: str) -> str:
        """
        Returns the name of the parent folder containing the 'Images' directory.
        Example:
        input:
        "....\LHP_Round_Polished_400_0002_11p7_2024-08-03\Images\Photo000.bmp"
        output:
        "LHP_Round_Polished_400_0002_11p7_2024-08-03"
        """
        # Normalize the path for cross-platform compatibility
        normalized_path = os.path.normpath(file_path)
        # Move one level up to the 'Images' folder
        images_folder = os.path.dirname(normalized_path)  # Directory of 'Images'
        # Move one more level up to the parent of 'Images'
        parent_folder = os.path.dirname(images_folder)   # Parent directory of 'Images'
        # Extract the name of the parent folder
        parent_folder_name = os.path.basename(parent_folder)
        return parent_folder_name

    @staticmethod
    def ImagesReader(image_paths:list, _pxl_instance):
        _pxl_instance.gray_images.clear()
        for path in image_paths:
            gray_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if gray_image is not None:
                _pxl_instance.gray_images.append(gray_image)

        __class__.case_id = __class__.get_parent_of_images_folder(image_paths[0])
        _pxl_instance.code_id = __class__.case_id

    def SetInputs(self, images:list, code_id:str):
        """
        Sets the input data for processing, including grayscale images and a code identifier.

        This method initializes the instance with a list of grayscale images and an associated
        unique code identifier. The images are expected to be provided as NumPy arrays.

        Args:
            images (list): A list of grayscale images represented as NumPy arrays. Each image
                should already be preprocessed into the desired format.
            code_id (str): A unique identifier associated with the images, typically indicating
                the source or context of the dataset.

        Returns:
            None

        NOTE:
            - The `images` list is directly assigned to the instance variable `gray_images`.
            - The `code_id` string is directly assigned to the instance variable `code_id`.

        Example:
            images = [numpy_array1, numpy_array2, numpy_array3]
            code_id = "folder_123"
            instance.SetInputs(images, code_id)
        """
        self.gray_images = images
        self.code_id = code_id
   
    def Go(self):
        """
        This method executes the complete modeling pipeline, including 2D and 3D processing, and adjusts the display.

        This method orchestrates a series of steps to process images, clean polygons, and build both fast
        and precise 3D models. It updates the internal state of the object, prepares the output for visualization, 
        and adjusts the display for optimal viewing.

        Workflow:
            1. Executes the 2D modeling phase using `self.modeling_2d()`.
            2. Performs polygon cleaning and calibration with `self.cleaned_polygons_calibration()`.
            3. Builds a fast preliminary 3D model using `self.fast_3d_modeling()`.
            4. Constructs an accurate 3D model with `self.precise_3d_modeling()`.
            5. Updates the internal state to indicate precise modeling is complete (`State.PRECISE_MODEL`).
            6. Prepares the precise surface for visualization by calling `self.set_surface()`.
            7. Adjusts the camera for optimal visualization of the 3D model with `self.ResetCameraObject()`.

        Returns:
            bool: The result of `self.precise_3d_modeling()` is a status code called "is_above"."

        Example:
            is_above = instance.Go()
        """
        # image processing phase
        self.modeling_2d(self)
        self.cleaned_polygons_calibration()
        # Fast 3D model building
        self.fast_3d_modeling() 
        # Accurate 3D modeling
        is_above = self.precise_3d_modeling()
        self.state = State.PRECISE_MODEL
        # set surface output for visualization
        self.set_surface(surface=self.precise_surface)       
        # Adjust the display for a good size of the 3D model.
        self.ResetCameraObject()
        return is_above


    @staticmethod
    def Go(_pxl_instance):        
        _pxl_instance.modeling_2d(_pxl_instance)
        _pxl_instance.cleaned_polygons_calibration()
        _pxl_instance.fast_3d_modeling() 
        is_above = _pxl_instance.precise_3d_modeling()
        
        _pxl_instance.state = State.PRECISE_MODEL

        _pxl_instance.bounds2D = list(_pxl_instance.bounds2D)
        _pxl_instance.precise_surface.GetBounds(_pxl_instance.bounds2D)
        _pxl_instance.bounds2D = [2 * x for x in _pxl_instance.bounds2D]
        _d = _pxl_instance.bounds2D[3]/4     
        _pxl_instance.bounds2D[3] -= _d    
        _pxl_instance.bounds2D[2] -= _d    

        _pxl_instance.set_surface(surface=_pxl_instance.precise_surface)        
        # print(_pxl_instance.bounds2D)
        _pxl_instance.ResetCameraObject()
        _pxl_instance.render_window.Render()
        return is_above



    @staticmethod
    def modeling_2d(_pxl_instance):
        # self.free_memory()

        # Initialize output variables
        # y_table = [-1, 0, 0] # Simulate an output parameter
        # max_val = [-1] # Simulate an output parameter
        image_number = 0
        image_qtty = len(_pxl_instance.gray_images)
        # w, h = 0, 0

        w = _pxl_instance.gray_images[0].shape[1]
        h = _pxl_instance.gray_images[0].shape[0]
        
        _pxl_instance.x_mins.clear()
        _pxl_instance.x_maxs.clear()
        _pxl_instance.y_mins.clear()
        _pxl_instance.y_maxs.clear()
        
        # proceessing images until isolines
        _pxl_instance.all_contour_points.clear()
        _pxl_instance.processing_images_to_isolines(_pxl_instance)
        
        for poly in _pxl_instance.cleaned_polygons:
            if isinstance(poly, list):
                poly.clear()
        _pxl_instance.cleaned_polygons.clear()
                
        # processing all contours
        for image_number in range(image_qtty):
            __class__.contour_processing_(_pxl_instance, image_number, image_qtty)
            pass
        
        # DLL saving
        if __class__.using_dll:
            status = save_polygons(__class__.case_id, w, h, __class__.pixel_size)

        # ******* Required values ​​related to image calibration  *******
        # xmin, xmax as the average of all values in all the images (400, 800)
        _pxl_instance.xmin = sum(_pxl_instance.x_mins) / len(_pxl_instance.x_mins)
        _pxl_instance.xmax = sum(_pxl_instance.x_maxs) / len(_pxl_instance.x_maxs)
        # The center of the camera
        xc=(_pxl_instance.xmin + _pxl_instance.xmax)/2
        # The offset of the center of the camera
        _pxl_instance.center_offset = xc - w/2
        # The table position (be careful)
        _pxl_instance.y_table = max(_pxl_instance.y_mins)
        # the culette position
        _pxl_instance.y_culette = min(_pxl_instance.y_maxs)

        # Local saving
        # save_polygons_HR(polygons:list, file_path, w, h, pixel_size, center_offset=0.34409, y_max=0)
        data_path = get_pp3d_data_path()

        if data_path is not None:
            # path to save the polygons in txt format
            polygons_path = os.path.join(data_path, "HR_" + _pxl_instance.code_id + '.txt')
            # save the polygons, but only the polygons from half of the images
            save_polygons_HR(_pxl_instance.decimated_arrays, file_path=polygons_path, \
                w=w, h=h, pixel_size=__class__.pixel_size, center_offset=_pxl_instance.center_offset, y_max=_pxl_instance.y_culette)        
        
        pass
            # image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            # if image is not None:
            #     self.images.append(image)
        # self.index = 0
        # self.select_camera(self.index)
        
        
        gc.collect()       # Force garbage collection to free memory

    @staticmethod
    def processing_images_to_isolines(_pxl_instance):

        y_table = [-1, 0, 0] # Simulate an output parameter
        max_val = [-1] # Simulate an output parameter
        
        _pxl_instance.w = _pxl_instance.gray_images[0].shape[1]
        _pxl_instance.h = _pxl_instance.gray_images[0].shape[0]
        
        image_qtty = len(_pxl_instance.gray_images)

        # processing gray_images for base cutting
        _pxl_instance.gray_images_cut.clear()
        for gray_image in _pxl_instance.gray_images:
            gray_image_cut = cv2_processing(gray_image.copy(), y_table, max_val)
            _pxl_instance.gray_images_cut.append(gray_image_cut)
        
            # Converting image to VTK format for visualization
            vtk_image = vtk.vtkImageData()            
            # Vertical flip is necessary from OpenCV to VTK
            img2 = gray_image_cut.copy()
            img2 = cv2.flip(img2, 0)
            # Convert the processed image to VTK format
            Pixel3D.pxl_tools.cv_to_vtk_image(img2, vtk_image)
            img2 = None            
            _pxl_instance.images.append(vtk_image)
            # Pixel3D.pxl_tools.processing_and_convert_to_vtk_image(gray_image_cut, vtk_image, y_table, max_val)    
        
        cx = y_table[1]
        cy = y_table[2]

        if not Pixel3D.pxl_tools.using_cv2_coordinates:
            cy = _pxl_instance.gray_images_cut[0].shape[0] - y_table[2]
                
        start_point=(cx, cy)
        
        iso_value = 128
        image_number = 0
        loop_condition = False
        _pxl_instance.all_contour_points.clear()

        loop_condition = True
        while(loop_condition):

            image_number = 0
            loop_condition = False
            _pxl_instance.all_contour_points.clear()

            # contours from cv2 images 
            for gray_image_cut in _pxl_instance.gray_images_cut:
                contour_points = Pixel3D.pxl_tools.detect_contour_subpixel(gray_image_cut, start_point, iso_value, image_number=image_number)
                if contour_points == None:
                    loop_condition = True
                    break
                _pxl_instance.all_contour_points.append(contour_points)
                image_number += 1

            if loop_condition:
                if enable_print:
                    print("Loop condition detected:", image_number)

                if iso_value < 60: 
                    iso_value -= 10
                else:
                    iso_value -= 20

                   
        print("Iso value:", iso_value)
        
        if loop_condition:
            if enable_print:
                print("Loop condition again.")
            pass
        pass
        
     
    @staticmethod        
    def contour_processing_(_pxl_instance, image_number, image_qtty):
        '''
        ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        '''
        
        __class__._image_number = image_number
        
        contour_points = _pxl_instance.all_contour_points[image_number] 
        
        # return two vtkPolyData
        iso_polyline, decimated_edges = \
            __class__.contour_decimation(contour_points, decimator_target=51)   # PENDIENTE  51 good 121

        _pxl_instance.edges.append(iso_polyline)
        _pxl_instance.decimated_edges.append(decimated_edges)

        # decimated points to ndarray, 51 points
        points = decimated_edges.GetPoints()
        decimated_point_array = vtk_to_numpy(points.GetData())
        decimated_point_array = decimated_point_array[:-1]
        decimated_point_array:np.ndarray = decimated_point_array.flatten()

        # processing decimate array for HR points 
        decimated_array = decimated_point_array.tolist()
        '''
        [968.0, 162.459228515625, 0.0, 1142.0, 162.46929931640625, 0.0, 
        1144.0, 162.57838439941406, 0.0, 1145.79541015625, 163.0, 0.0, 
        1150.0, 165.0625, 0.0, 1198.0, 186.84210205078125, 0.0, 1200.0, 
        ...]
        '''
        
        # Checking for convexity, non-convex points are eliminated
        if True or not __class__.using_dll:
            decimated_array, ccw = check_convexity_and_ccw(decimated_array)  # PENDIENTE
        
        # saving convex arrays of points for fast modeling (HR prefix)
        _pxl_instance.decimated_arrays.append(decimated_array)
        # get min, max values for XY coordinates
        # return {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y}
        result = get_min_max_xy(decimated_array)
        _pxl_instance.x_mins.append(result["min_x"])
        _pxl_instance.x_maxs.append(result["max_x"])
        _pxl_instance.y_mins.append(result["min_y"])
        _pxl_instance.y_maxs.append(result["max_y"])
        
        # Changing array format to sent to DLL
        decimated_array_flat = np.array(decimated_array).flatten()

        '''
        ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        '''
        select_check_convexity_and_ccw_option = False   # PENDIENTE
        
        if __class__.using_dll:    
                
            if select_check_convexity_and_ccw_option:
                # new option with convexity checking  
                cleaned_array = get_cleaned_points_from_decimated(decimated_array_flat, image_number, image_qtty)
            else:
                # original option with 51 points   
                # print("get_cleaned:", image_number)     
                cleaned_array = get_cleaned_points_from_decimated(decimated_point_array, image_number, image_qtty)
    
            # cleaned_array, decimated_array = get_cleaned_points_from_decimated(decimated_edges, image_number, image_qtty)
            if cleaned_array is None or len(cleaned_array) == 0:
                pass
            else:        
                cleaned_array.append(cleaned_array[0])
                cleaned_array.append(cleaned_array[1])
                cleaned_array.append(cleaned_array[2])

            cleaned_polydata = __class__.create_vtk_polydata_from_flat_list(cleaned_array)
            _pxl_instance.cleaned.append(cleaned_polydata) 
            _pxl_instance.cleaned_polygons.append(cleaned_array)
        
        decimated_point_array = None
        decimated_array_flat = None
        # if not ccw:
        #     pass 
        
        # get the XY coordinates
        # decimated_point_array = decimated_point_array[:, :2]

        # get cleaned polydata form decimeted polydata

        # h, w = gray_image_cut.shape
        # # gray_image = None  # Remove the reference to gray_image
        # # gc.collect()       # Force garbage collection to free memory
        # return w, h
        
    
    def add_text(self, label, scale_text):
        """
        Add text to a VTK renderer with specified scale and properties.

        :param renderer: vtkRenderer instance where the text will be added.
        :param label: The text to display.
        :param scale_text: Scale of the text for better visibility.
        :param vector_text: vtkVectorText instance to define the text (optional, creates a new one if None).
        :param text_actor: vtkFollower instance to represent the text actor (optional, creates a new one if None).
        :return: Updated vector_text and text_actor.
        """
        if self.second_renderer == None:
            return
        
        if self.text_actor is None:
            self.text_actor = vtk.vtkActor()
        
            # Initialize vtkVectorText if not provided
            if self.vector_text is None:
                self.vector_text = vtk.vtkVectorText()

            # Create a mapper for the text
            text_mapper = vtk.vtkPolyDataMapper()
            text_mapper.SetInputConnection(self.vector_text.GetOutputPort())

            # Set mapper and color of text_actor
            self.text_actor.SetMapper(text_mapper)
            self.text_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Yellow color for the text

            # Set position (example position; adjust as needed)
            self.text_actor.SetPosition(-100, 20, 0)
            self.text_actor.VisibilityOn()

            # Add the text actor to the renderer
            self.second_renderer.AddActor(self.text_actor)

        self.vector_text.SetText(label)
        self.text_actor.SetScale(scale_text)  # Scale the text for better visibility

    @staticmethod
    def connectivity3(input_poly_data, result_poly_data):
        """
        Reorders the segments of a polyline in input_poly_data to form a closed loop 
        and stores the result in result_poly_data.

        :param input_poly_data: vtkPolyData, input containing the polyline segments.
        :param result_poly_data: vtkPolyData, output containing the closed polyline.
        """
        # Create a new vtkPoints to store the resulting points
        result_points = vtk.vtkPoints()
        result_lines = vtk.vtkCellArray()

        # List to store the ordered point IDs
        line_id_list = vtk.vtkIdList()

        # Map to quickly find the next tuple
        next_point_map = {}
        id_list = vtk.vtkIdList()

        # Build the nextPointMap from input_poly_data
        num_cells = input_poly_data.GetNumberOfCells()
        for i in range(num_cells):
            input_poly_data.GetCellPoints(i, id_list)
            p0 = id_list.GetId(0)
            p1 = id_list.GetId(1)
            next_point_map[p0] = p1

        # Find the starting point for the polyline
        # The iter() function returns an iterator for the dictionary keys.
        # The next() function retrieves the next item from the iterator.
        # When next() is used for the first time, it retrieves the first key in the dictionary
        current_point = next(iter(next_point_map))  # Get the first key from the map
        start_point = current_point

        # Mapping of original point IDs to new point IDs in resultPoints
        point_id_map = {}
        new_point_id = 0

        # Traverse the polyline to order points
        while True:
            next_point = next_point_map[current_point]

            # If the point has not been mapped, add it to result_points
            if current_point not in point_id_map:
                point = [0.0, 0.0, 0.0]
                input_poly_data.GetPoint(current_point, point)
                result_points.InsertNextPoint(point)
                point_id_map[current_point] = new_point_id
                line_id_list.InsertNextId(new_point_id)
                new_point_id += 1

            # Move to the next point
            current_point = next_point

            # Stop when the loop closes
            if current_point == start_point:
                break

        # Create a polyline and add the ordered points
        poly_line = vtk.vtkPolyLine()
        poly_line.GetPointIds().SetNumberOfIds(line_id_list.GetNumberOfIds())

        for i in range(line_id_list.GetNumberOfIds()):
            poly_line.GetPointIds().SetId(i, line_id_list.GetId(i))

        # Add the polyline to the resulting cells
        result_lines.InsertNextCell(poly_line)

        # Set the points and lines in the result_poly_data
        result_poly_data.SetPoints(result_points)
        result_poly_data.SetLines(result_lines)

        # result_poly_data now contains a single closed polyline

    @staticmethod
    def calculate_signed_area(poly_data, sample_rate=10):
        """
        Calculate the signed area of a convex polyline using a reduced number of points.

        :param poly_data: vtkPolyData containing the polyline.
        :param sample_rate: Step size for sampling points (e.g., every 10th point).
        :return: The signed area (positive for CCW, negative for CW).
        """
        cell = poly_data.GetCell(0)  # Get the polyline cell
        points = poly_data.GetPoints()
        num_points = cell.GetNumberOfPoints()

        if num_points < 3:
            return 0

        # Sample points at intervals of 'sample_rate'
        sampled_indices = range(0, num_points, sample_rate)
        sampled_indices = list(sampled_indices)  # Ensure it's a list
        if sampled_indices[-1] != num_points - 1:  # Ensure the last point is included
            sampled_indices.append(num_points - 1)

        signed_area = 0.0

        for i in range(len(sampled_indices)):
            p1_id = cell.GetPointId(sampled_indices[i])
            p2_id = cell.GetPointId(sampled_indices[(i + 1) % len(sampled_indices)])  # Wrap around
            p1 = points.GetPoint(p1_id)
            p2 = points.GetPoint(p2_id)

            # Add the cross product of the edge (p1, p2) to the signed area
            signed_area += (p2[0] - p1[0]) * (p2[1] + p1[1])

        return signed_area

    @staticmethod
    def is_ccw_and_fix(poly_data):
        """
        Checks if the polyline in the given vtkPolyData is ordered counter-clockwise (CCW).
        If not, reverses the point order to make it CCW.

        :param poly_data: vtkPolyData containing a single polyline in the XY plane.
        """
        if poly_data.GetNumberOfCells() != 1:
            return

        # Get the polyline points
        cell = poly_data.GetCell(0)  # Get the first (and only) cell
        points = poly_data.GetPoints()
        num_points = cell.GetNumberOfPoints()

        if num_points < 3:
            return

        # Calculate the signed area of the polygon
        signed_area = 0
        sample_rate = 1
        
        if True:
            signed_area = __class__.calculate_signed_area(poly_data, sample_rate=sample_rate)
        else:
            # Sample points at intervals of 'sample_rate'
            sampled_indices = range(0, num_points, sample_rate)
            sampled_indices = list(sampled_indices)  # Ensure it's a list
            if sampled_indices[-1] != num_points - 1:  # Ensure the last point is included
                sampled_indices.append(num_points - 1)

            for i in range(len(sampled_indices)):
                p1_id = cell.GetPointId(sampled_indices[i])
                p2_id = cell.GetPointId(sampled_indices[(i + 1) % len(sampled_indices)])  # Wrap around
                p1 = points.GetPoint(p1_id)
                p2 = points.GetPoint(p2_id)

                # Add the cross product of the edge (p1, p2) to the signed area
                signed_area += (p2[0] - p1[0]) * (p2[1] + p1[1])
            
        # CCW if signed_area > 0, CW if signed_area < 0
        is_ccw = signed_area > 0

        if not is_ccw:
            # print("Polyline is clockwise. Reversing the point order.")

            # Reverse the point order
            reversed_ids = [cell.GetPointId(i) for i in range(num_points)][::-1]
            reversed_cell = vtk.vtkCellArray()
            reversed_cell.InsertNextCell(num_points)
            for point_id in reversed_ids:
                reversed_cell.InsertCellPoint(point_id)

            # Update the polyline in the poly_data
            poly_data.SetLines(reversed_cell)
        else:
            # print("Polyline is counter-clockwise.")
            pass

    @staticmethod
    def replace_closed_line_with_segments(poly_data):
        """
        Replace a closed polyline in the given vtkPolyData with individual line segments.

        :param poly_data: vtkPolyData containing the closed polyline.
        """
        # Get points from the existing polyData
        points = poly_data.GetPoints()
        num_points = points.GetNumberOfPoints()

        # Create a new cell array to hold the line segments
        lines = vtk.vtkCellArray()

        # Loop through points and create lines between consecutive points
        for i in range(num_points):
            # Create a line between point i and point (i + 1) % numPoints
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, i)
            line.GetPointIds().SetId(1, (i + 1) % num_points)

            # Insert the line into the cell array
            lines.InsertNextCell(line)

        # Set the new lines to the polyData
        poly_data.SetLines(lines)


    @staticmethod
    def process_image_to_segments(img:vtk.vtkImageData, iso_value=128, decimator_target=51):
        """
        Processes a vtkImageData to extract, decimate, and convert a contour into line segments.

        :param cameras: List of camera objects containing vtk_image.
        :param nx: Index of the camera to process.
        :param iso_value: Isovalue for vtkMarchingSquares.
        :param decimator_target: Target number of points for decimation.
        :return: Processed vtkPolyData containing line segments.
        """
        # Initialize the pipeline components
        # iso = vtk.vtkMarchingSquares()
        # connectivity_filter = vtk.vtkPolyDataConnectivityFilter()
        # edge_decimator = vtk.vtkDecimatePolylineFilter()

        # Get the input vtkImageData
        # img = cameras[nx].vtk_image

        dims = img.GetDimensions()
        
        
        start_time = time.time()
        # Configure vtkMarchingSquares
        __class__.iso.SetInputData(img)
        __class__.iso.SetValue(0, iso_value)
        __class__.connectivity_filter.Update()
        lapse = round(time.time() - start_time, 3)
        # print("iso lapse:", lapse)

        # Get the output of the connectivity filter
        iso_output = __class__.connectivity_filter.GetOutput()
        num_points = iso_output.GetNumberOfPoints()

        if num_points == 0:
            return None, None

        # Prepare a new vtkPolyData for the sorted polyline
        polyline = vtk.vtkPolyData()

        # start_time = time.time()
        # Perform connectivity sorting
        __class__.connectivity3(iso_output, polyline)
        # lapse = round(time.time() - start_time, 3)
        # print("conectivity lapse:", lapse)

        # start_time = time.time()
        # Edge decimation using vtkDecimatePolylineFilter
        number_of_points = polyline.GetNumberOfPoints()
        reduction = 1.0 - (1.0 * decimator_target) / number_of_points
        __class__.edge_decimator.SetInputData(polyline)
        __class__.edge_decimator.SetTargetReduction(reduction)
        __class__.edge_decimator.Update()
        # lapse = round(time.time() - start_time, 3)
        # print("decimator lapse:", lapse)

        # Get the decimated segments
        decimated_edges = vtk.vtkPolyData()
        decimated_edges.DeepCopy(__class__.edge_decimator.GetOutput())

        # start_time = time.time()
        # Ensure counter-clockwise contour and single-cell conversion
        __class__.is_ccw_and_fix(decimated_edges)
        # lapse = round(time.time() - start_time, 3)
        # print("ccw verification lapse:", lapse)

        # start_time = time.time()
        # Convert the contour into individual line segments
        __class__.replace_closed_line_with_segments(decimated_edges)
        # lapse = round(time.time() - start_time, 3)
        # print("poliline to segments lapse:", lapse)

        del iso_output
        return polyline, decimated_edges


    @staticmethod
    def contour_decimation(contour_points, decimator_target=51):
        """
        Processes a vtkImageData to extract, decimate, and convert a contour into line segments.

        :param cameras: List of camera objects containing vtk_image.
        :param nx: Index of the camera to process.
        :param iso_value: Isovalue for vtkMarchingSquares.
        :param decimator_target: Target number of points for decimation.
        :return: Processed vtkPolyData containing line segments.
        """

        if len(contour_points) == 0:
            return

        polyline = Pixel3D.pxl_tools.create_polydata_from_contour(contour_points)
        
        # Edge decimation using vtkDecimatePolylineFilter
        number_of_points = polyline.GetNumberOfPoints()
        reduction = 1.0 - (1.0 * decimator_target) / number_of_points
        __class__.edge_decimator.SetInputData(polyline)
        __class__.edge_decimator.SetTargetReduction(reduction)
        __class__.edge_decimator.Update()


        # Get the decimated segments
        decimated_edges = vtk.vtkPolyData()
        decimated_edges.DeepCopy(__class__.edge_decimator.GetOutput())

        # Ensure counter-clockwise contour and single-cell conversion
        __class__.is_ccw_and_fix(decimated_edges)


        # Convert the contour into individual line segments
        __class__.replace_closed_line_with_segments(decimated_edges)

        # del iso_output
        return polyline, decimated_edges


    # dont use more, replaced by contour_decimation()
    @staticmethod
    def process_image_to_segments_2(img, iso_value=128, start_point = (1000, 1000), decimator_target=51, image_number=-1):
        """
        Processes a vtkImageData to extract, decimate, and convert a contour into line segments.

        :param cameras: List of camera objects containing vtk_image.
        :param nx: Index of the camera to process.
        :param iso_value: Isovalue for vtkMarchingSquares.
        :param decimator_target: Target number of points for decimation.
        :return: Processed vtkPolyData containing line segments.
        """
        # Initialize the pipeline components
        # iso = vtk.vtkMarchingSquares()
        # connectivity_filter = vtk.vtkPolyDataConnectivityFilter()
        # edge_decimator = vtk.vtkDecimatePolylineFilter()

        # Get the input vtkImageData
        # img = cameras[nx].vtk_image

        # dims = img.GetDimensions()
        
        
        start_time = time.time()
        

        # def detect_contour_subpixel(image, start_point, threshold):
        # def create_polydata_from_contour(contour_points):
        
        # start_point = (1000, 1000)
        contour_points = Pixel3D.pxl_tools.detect_contour_subpixel(img, start_point, iso_value, image_number=image_number)
        
        if len(contour_points) == 0:
            return

        polyline = Pixel3D.pxl_tools.create_polydata_from_contour(contour_points)


        # # Configure vtkMarchingSquares
        # __class__.iso.SetInputData(img)
        # __class__.iso.SetValue(0, iso_value)
        # __class__.connectivity_filter.Update()

        
        lapse = round(time.time() - start_time, 3)
        # print("iso lapse:", lapse)

        # Get the output of the connectivity filter
        # iso_output = __class__.connectivity_filter.GetOutput()
        
        
        # num_points = iso_output.GetNumberOfPoints()

        # if num_points == 0:
        #     return None, None

        # Prepare a new vtkPolyData for the sorted polyline
        # polyline = vtk.vtkPolyData()

        # start_time = time.time()
        # Perform connectivity sorting
        # __class__.connectivity3(iso_output, polyline)
        # lapse = round(time.time() - start_time, 3)
        # print("conectivity lapse:", lapse)



        # start_time = time.time()
        # Edge decimation using vtkDecimatePolylineFilter
        number_of_points = polyline.GetNumberOfPoints()
        reduction = 1.0 - (1.0 * decimator_target) / number_of_points
        __class__.edge_decimator.SetInputData(polyline)
        __class__.edge_decimator.SetTargetReduction(reduction)
        __class__.edge_decimator.Update()
        # lapse = round(time.time() - start_time, 3)
        # print("decimator lapse:", lapse)

        # Get the decimated segments
        decimated_edges = vtk.vtkPolyData()
        decimated_edges.DeepCopy(__class__.edge_decimator.GetOutput())

        # start_time = time.time()
        # Ensure counter-clockwise contour and single-cell conversion
        __class__.is_ccw_and_fix(decimated_edges)
        # lapse = round(time.time() - start_time, 3)
        # print("ccw verification lapse:", lapse)

        # start_time = time.time()
        # Convert the contour into individual line segments
        __class__.replace_closed_line_with_segments(decimated_edges)
        # lapse = round(time.time() - start_time, 3)
        # print("poliline to segments lapse:", lapse)

        # del iso_output
        return polyline, decimated_edges

    
    def fast_3d_modeling(self):
        # if __class__.using_dll:
        #     self.prefix = "HR_"
        # else:
        #     self.prefix = ""
        self.prefix = "" if __class__.using_dll else "HR_"
        
        poly_data = [self.cleaned_polygons, self.xmin, self.xmax, self.ymin, self.ymax, self.ymax, __class__.pixel_size, self.center_offset] 
        # polygons, x_min, x_max, y_min, y_max, y_culette, pixel_size, center_offset
        
        self.fast_surface, N, self.pixel_size, self.center_offset =\
            execute_1(case_id=__class__.case_id, prefix=self.prefix, poly_data=poly_data, index=self.index)        
        # self.set_surface(surface=self.fast_surface)
        self.image_qtty = 2*N
    
    def split_surface(self):
        '''
        ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        '''
        
        if self.fast_surface is None:
            return        
        
        # Extract points, normals, and cell types from vtkPolyData.
        self.extract_polydata_information(self.fast_surface)

        start_time_1 = time.time()

        # Step #1: surface splitting in table, girdle and pavilion
        _s1, s2, _s3 = split_polydata_by_elevation(self.fast_surface)
        # self.aps[_TABLE_SURFACE].set_input(_s1)
        # self.aps[_PAVILION_SURFACE].set_input(_s3)    
        # self.aps[_TABLE_SURFACE].edge_visibility(False)
        # self.aps[_PAVILION_SURFACE].edge_visibility(False)

        # girdle thickness
        y1, y2 = get_true_y_range(s2)
        thickness = y2-y1
        # pixel size
        pixel_size = __class__.pixel_size

        # table clipping 
        y1, y2 = get_true_y_range(_s1)
        y1 += pixel_size  # 2*pixel_size 
        y2 -= thickness/2 # thickness/2  
        _s1 = y_cutter(_s1, y1, y2)
        
        # pavillion clipping 
        y1, y2 = get_true_y_range(_s3)
        y1 += thickness 
        y2 -= 2*thickness   
        _s3 = y_cutter(_s3, y1, y2)
        
        def get_area_dict(polydata:vtk.vtkPolyData):
            num_cells = polydata.GetNumberOfCells()
            area_dict = {}
            for i in range(num_cells):
                area_dict[i] = round(float(get_cell_area(polydata, i)), 5)
            return area_dict
                
        '''
        ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        '''
        # Step #2: mesh regularization
        select_vtk_option = True
        # Step #2: mesh regularization
        if select_vtk_option:
            s1 = regularize_mesh_with_target_points_vtk(_s1)
            s3 = regularize_mesh_with_target_points_vtk(_s3)
        else:  # gmsh option
            s1 = regularize_mesh_with_target_points_gmsh(_s1, target_size=10)
            s3 = regularize_mesh_with_target_points_gmsh(_s3, target_size=10)

        # s1 = regularize_mesh_with_target_points(_s1, target_size=10)
        # s3 = regularize_mesh_with_target_points(_s3, target_size=10)
        # s1 = _s1
        # s3 = _s3
        
        s1_area_dict = get_area_dict(s1)
        s2_area_dict = get_area_dict(s2)
        s3_area_dict = get_area_dict(s3)
        
        s1_area_dict = dict(sorted(s1_area_dict.items(), key=lambda item: item[1]))
        s2_area_dict = dict(sorted(s2_area_dict.items(), key=lambda item: item[1]))
        s3_area_dict = dict(sorted(s3_area_dict.items(), key=lambda item: item[1]))
        
        
        n1 = 32             # table facets 
        offset1 = -0.18
        n3 = 24            # pavilion facets
        offset3 = 0.18

        _l1 = int(len(s1_area_dict)/2)
        _l3 = int(len(s3_area_dict)/2)
        s1_area_sub_dict = dict(list(s1_area_dict.items())[-_l1:])
        s3_area_sub_dict = dict(list(s3_area_dict.items())[-_l3:])
        # s1_area_sub_dict = dict(list(s1_area_dict.items())[-n1*2:])
        # s3_area_sub_dict = dict(list(s3_area_dict.items())[-n3*2:])

        # s1 = filter_cells_by_ids_with_existing_normals(s1, list(s1_area_sub_dict.keys()))
        # s3 = filter_cells_by_ids_with_existing_normals(s3, list(s3_area_sub_dict.keys()))
        
        self.aps[_TABLE_SURFACE].set_input(s1)
        self.aps[_PAVILION_SURFACE].set_input(s3)    
        self.aps[_TABLE_SURFACE].edge_visibility(False)
        self.aps[_PAVILION_SURFACE].edge_visibility(False)

        # ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        
        # Step #3
        '''
        classifier options.
        1. classify_facets_by_normals
        2. classify_facets_by_normals_area_weighting
        3. classify_facets_by_normals_agglomerative
        4. classify_facets_by_normals_dbscan
        '''
        classified_table_surface, classified_pavilion_surface, i1, i3, \
            table_labels, pavilion_labels, \
                table_group_areas, pavilion_group_areas = \
            facets_clustering_and_classification(s1, s3, table_faces=n1, pavilion_faces=n3, selected_classifier=3)
        '''
        table_labels =     
        # one label by cell
        array([25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25,
            25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 28,
            28,  0,  0,  0, 26, 26, 26, 26, 26, 26, 26, 26, 28, 28, 28, 28, 26,
            26, 26, 26, 26, 26, 26, 26, 26, 28, 28, 28,  0,  0,  0, 26, 26, 26,
            etc.. ])

        pavilion_labels =     
        # one label by cell
        array([ 5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  7,  7,  7,  7,
                7,  7, 20, 20, 22, 22, 22, 22, 22, 20, 20, 20, 20, 20, 20, 20, 20,
            20, 20, 20, 22, 22, 22, 22, 22, 22, 22, 22, 22, 20, 20, 20, 20, 22,
                6,  6,  6,  6,  6, 22, 22,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,
                etc...])

        table_group_areas =    
        one value by cluster, for example (4:0.064), area of cluster 4 is 0.064 
        {4: 0.604, 15: 0.616, 6: 0.62, 0: 0.621, 20: 0.621, 23: 0.633, 8: 0.642, 30: 0.644, etc...}

        pavilion_group_areas =    
        one value by cluster, for example (17:0.015), area of cluster 17 is 0.015 
        {17: 0.015, 20: 0.766, 18: 0.957, 3: 1.038, 19: 1.106, 21: 1.12, 22: 1.177, 13: 1.185, etc...}
        in this example cluster 17 is subsequently removed.
        '''
            
        # table_groups = list(set(table_labels.tolist()))
        # pavilion_groups = list(set(pavilion_labels.tolist()))

        if enable_print:
            print("i1, i3: ", i1, i3)
        
        # ********** Filtering by area of small clusters ***********
        # filter of small group areas
        if len(table_group_areas) > n1:
            # Filter from index k1 to the end
            k1 = len(table_group_areas) - n1              
            # Convert the dictionary to a list of tuples and filter out small area positions
            items = list(table_group_areas.items())
            table_group_areas = dict(items[k1:])

        # filter of small group areas
        if len(pavilion_group_areas) > n3:
            # Filter from index k1 to the end
            k3 = len(pavilion_group_areas) - n3              
            # Convert the dictionary to a list of tuples and filter out small area positions
            items = list(pavilion_group_areas.items())
            pavilion_group_areas = dict(items[k3:])
        
        
        '''
        Precision 3D modeling
        '''
        
        
        def cells_by_group(cell_labels, group_areas):
            '''
            inputs:
            surface_labels: List of labels assigned to cells, 
                            each label is the number of a group, 
                            each group represents a facet (first 
                            version of facet as a group)
            group_areas: Area of ​​each group corresponds to the area 
                         of ​​the first version of a facet conceived as a group.
                         Small groups have already been eliminated, leaving 
                         only the groups that represent the facets
            output:
            result:
                    The output is a dictionary where the key represents 
                    a group index or "diamond facet" index and the value 
                    assigned to each key is the list of cells that make up the facet

            Samples of a result:
            Label representing the group: [list of cells belonging to the group]                    
            np.int64(25): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            np.int64(28): [33, 34, 46, 47, 48, 49, 59, 60, 61, 68, 69, 70, 71, 72, 73, 989, 990, 991, 992, 993, 994, 995, 996]
            np.int64(0): [35, 36, 37, 62, 63, 64, 74, 75, 76, 82, 83, 946, 947, 948, 949, 950, 951, 957, 958, 959, 960, 964, 965, 971, 972]
            '''
            # Create the dictionary
            result = {}
            # Allowed values to filter the dictionary
            allowed_values = list(group_areas.keys())

            for index, value in enumerate(cell_labels):
                # Filter the dictionary based on allowed values
                if value not in allowed_values:
                    continue 
                # Filtered value is processed
                if value not in result:
                    result[value] = []
                result[value].append(index)                
            return result
            # Filter the dictionary based on allowed values
            # allowed_values = list(group_areas.keys())
            # result = {key: value for key, value in result.items() if key in allowed_values}
        
        # Cells for each preliminary diamond facet classified as a group
        table_cells_id_by_group = cells_by_group(table_labels, table_group_areas)
        pavilion_cells_id_by_group = cells_by_group(pavilion_labels, pavilion_group_areas)
        
        facet_data_for_table_statistics = {}
        facet_data_for_pavilion_statistics = {}
        
        # ************* Statistics **************
        # Build statistics to obtain cutting planes for final model.
        # The cutting plane includes the normal and a point.
        for index, values in table_cells_id_by_group.items():
            statistics_table = get_facet_properties(s1, values)
            facet_data_for_table_statistics[int(index)] = statistics_table

        for index, values in pavilion_cells_id_by_group.items():
            statistics_table = get_facet_properties(s3, values)
            facet_data_for_pavilion_statistics[int(index)] = statistics_table


        # save statistics in Excel        
        facet_data_to_excel(facet_data_for_table_statistics, prefix="facet_data_table_", code_id=self.code_id)
        facet_data_to_excel(facet_data_for_pavilion_statistics, prefix="pavilion_data_table_", code_id=self.code_id)

        # ********** multicolor mapping **********
        # mappers are set to multi-color, set_input() is called within this method
        self.aps[TABLE_SURFACE].multi_color_mapping(classified_table_surface, table_faces=n1)
        self.aps[PAVILION_SURFACE].multi_color_mapping(classified_pavilion_surface, table_faces=n3)

        # add label actors to table abd pavilion
        self.aps[TABLE_SURFACE].add_label_actors(offset=-0.18)
        self.aps[PAVILION_SURFACE].add_label_actors(offset=0.18)

        normals:vtk.vtkDoubleArray = None
        points:vtk.vtkPoints = None

        # ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

        # ************ Cutting by planes *************
        # points, normals = get_planes_from_polydata(s2, points, normals)

        # all the points and normals of the final model        
        points = vtk.vtkPoints()
        normals = vtk.vtkDoubleArray()
        normals.SetNumberOfComponents(3)
        normals.SetName("FacetNormals")

        pav_points, pav_normals = \
            get_planes_from_facet_data(facet_data_for_pavilion_statistics, points, normals)
        tab_points, tab_normals = \
            get_planes_from_facet_data(facet_data_for_table_statistics, points, normals)

        # Computes the Y component and the angle in the XZ plane for a given dataset.
        # ascendent sorted by Y values.
        table_orientations = compute_orientation(facet_data_for_table_statistics)        
        pavilion_orientations = compute_orientation(facet_data_for_pavilion_statistics)

        # get 8 items with best Y value 
        table_orientations_best_y = dict(list(table_orientations.items())[-8:])
        # sorting last result by XY angle
        table_orientations_best_y_angle_xy_sorted = dict(sorted(table_orientations_best_y.items(), \
            key=lambda item: math.atan2(item[1]['angle xz'], item[1]['y normal'])))

        # pavilion orietation dictionary sorted by XY angle
        pavilion_orientations_angle_xy_sorted = dict(sorted(pavilion_orientations.items(), \
            key=lambda item: math.atan2(item[1]['angle xz'], item[1]['y normal'])))

        # angle matching between table upper cells and pavillion upper cells
        angular_matching = find_angular_matching(table_orientations_best_y_angle_xy_sorted, \
            pavilion_orientations_angle_xy_sorted, max_angular_difference=20)
        
        s4 = construct_solid_from_planes(points, normals, w=40)
        self.surfaceMapper.SetInputData(s4)
                
        lapse = round(time.time() - start_time_1, 3)
        if enable_print:
            print("facet classification lapse:", lapse)

        self.ViewSurface()
        
        self.first_renderer.ResetCamera()
        self.render_window.Render()


        option = 2
        if option == 1:
            # # mappers are set to multi-color, set_input() is called within this method
            # self.aps[TABLE_SURFACE].multi_color_mapping(classified_table_surface, table_faces=n1)
            # self.aps[PAVILION_SURFACE].multi_color_mapping(classified_pavilion_surface, table_faces=n3)
        
            # set_input() is called for girdle, no multicolor mappers 
            self.aps[GIRDLE_SURFACE].set_input(s2)

            # add label actors to table abd pavilion
            # self.aps[TABLE_SURFACE].add_label_actors(offset=-0.18)
            # self.aps[PAVILION_SURFACE].add_label_actors(offset=0.18)

            # set visibility
            self.actors_off()
            self.surfaceActor.VisibilityOff()        
            self.aps[TABLE_SURFACE].visibility(True)
            self.aps[GIRDLE_SURFACE].visibility(False)
            self.aps[PAVILION_SURFACE].visibility(True)    
            
        elif option == 2:
            pass
        
        # Render the scene
        self.render_window.Render()


    def ViewGroups(self):
        self.actors_off()
        self.surfaceActor.VisibilityOff()        
        self.aps[TABLE_SURFACE].visibility(True)
        self.aps[GIRDLE_SURFACE].visibility(False)
        self.aps[PAVILION_SURFACE].visibility(True)    
        self.render_window.Render()
        
    def ViewSplits(self):
        self.actors_off()
        self.surfaceActor.VisibilityOff()        
        self.aps[_TABLE_SURFACE].visibility(True)
        self.aps[GIRDLE_SURFACE].visibility(False)
        self.aps[_PAVILION_SURFACE].visibility(True)    
        self.render_window.Render()

    def ViewSurface(self):
        self.actors_off()
        self.surfaceActor.VisibilityOn()
        self.aps[TABLE_SURFACE].labels_visibility(True)
        self.aps[PAVILION_SURFACE].labels_visibility(True)    
        self.render_window.Render()
            
    def SwitchEdgeVisibility(self):
        edge_visibility = self.surfaceActor.GetProperty().GetEdgeVisibility()
        if edge_visibility == 1:
            self.surfaceActor.GetProperty().EdgeVisibilityOff()
        else: 
            self.surfaceActor.GetProperty().EdgeVisibilityOn()
        self.render_window.Render()
                    
    def relocate_pavilion_points(self, pavilion_locations:list):

        def PointInterception(surface, y1, angle, intersection):
            """
            Calculate the interception point of a line with a vtkPolyData surface.

            Parameters:
            surface (vtk.vtkPolyData): The surface to intersect.
            y1 (float): The y-coordinate for the start and end points of the line.
            angle (float): The angle (in degrees) to compute the line direction.
            intersection (list of float): The intersection point will be stored in this list.

            Returns:
            bool: True if an intersection occurs, False otherwise.
            """

            # Get the bounds of the surface
            bounds = [0.0] * 6
            surface.GetBounds(bounds)
            R = max(bounds[1] - bounds[0], bounds[5] - bounds[4])

            # Define the start and end points of the line
            p1 = [0, y1, 0]
            p2 = [R * np.cos(angle * (np.pi / 180.0)), y1, R * np.sin(angle * (np.pi / 180.0))]

            # Create and configure the vtkOBBTree if it hasn't been initialized
            # if self.obb_tree is None:
            #     self.obb_tree = vtk.vtkOBBTree()
            #     self.obb_tree.SetDataSet(surface)
            #     self.obb_tree.BuildLocator()

            # Create a vtkPoints object to store the intersection points
            intersect_points = vtk.vtkPoints()

            # Intersect the line (p1, p2) with the surface
            if self.obb_tree.IntersectWithLine(p1, p2, intersect_points, None) != 0:
                # Get the first intersection point
                intersect_points.GetPoint(0, intersection)
                return True
            else:
                return False

        self.obb_tree = vtk.vtkOBBTree()
        self.obb_tree.SetDataSet(self.fast_surface)
        self.obb_tree.BuildLocator()

        # ******* Relocating pavilion points ******* 
        # Auxiliary point
        p3 = [0.0, 0.0, 0.0]            
        # Move the detected location nodes
        for node_id, y, angle in pavilion_locations:                        
            # Get pavilion location
            result = PointInterception(self.fast_surface, y, angle, p3)
            
            if result:
                # Relocate pavilion points
                self.points_array[node_id][:] = p3
                if enable_print:
                    print("p3[1]:",p3[1])
                # self.points_array[node_id][1] = p3[1]

                # points = polydata.GetPoints()
                # points.SetPoint(point_id, new_position)
                # polydata.Modified()

    def relocate_pavilion_points_good(self, pavilion_locations:list):

        def PointInterception(surface, y1, angle, intersection):
            """
            Calculate the interception point of a line with a vtkPolyData surface.

            Parameters:
            surface (vtk.vtkPolyData): The surface to intersect.
            y1 (float): The y-coordinate for the start and end points of the line.
            angle (float): The angle (in degrees) to compute the line direction.
            intersection (list of float): The intersection point will be stored in this list.

            Returns:
            bool: True if an intersection occurs, False otherwise.
            """

            # Get the bounds of the surface
            bounds = [0.0] * 6
            surface.GetBounds(bounds)
            R = max(bounds[1] - bounds[0], bounds[5] - bounds[4])

            # Define the start and end points of the line
            p1 = [0, y1, 0]
            p2 = [R * np.cos(angle * (np.pi / 180.0)), y1, R * np.sin(angle * (np.pi / 180.0))]

            # Create and configure the vtkOBBTree if it hasn't been initialized
            # if self.obb_tree is None:
            #     self.obb_tree = vtk.vtkOBBTree()
            #     self.obb_tree.SetDataSet(surface)
            #     self.obb_tree.BuildLocator()

            # Create a vtkPoints object to store the intersection points
            intersect_points = vtk.vtkPoints()

            # Intersect the line (p1, p2) with the surface
            if self.obb_tree.IntersectWithLine(p1, p2, intersect_points, None) != 0:
                # Get the first intersection point
                intersect_points.GetPoint(0, intersection)
                return True
            else:
                return False

        self.obb_tree = vtk.vtkOBBTree()
        self.obb_tree.SetDataSet(self.fast_surface)
        self.obb_tree.BuildLocator()

        # ******* Relocating pavilion points ******* 
        # Auxiliary point
        p3 = [0.0, 0.0, 0.0]            
        # Move the detected location nodes
        for node_id, y, angle in pavilion_locations:                        
            # Get pavilion location
            result = PointInterception(self.fast_surface, y, angle, p3)
            
            if result:
                # Relocate pavilion points
                self.points_array[node_id][:] = p3
                if enable_print:
                    print("p3[1]:",p3[1])
                # self.points_array[node_id][1] = p3[1]

                # points = polydata.GetPoints()
                # points.SetPoint(point_id, new_position)
                # polydata.Modified()

    def relocate_pavilion_points_new(self, pavilion_locations:list, cells:list, table_center):
        '''
        relocate rhomboid pavillion points
        '''

        # Encontrar los primeros 8 enteros faltantes
        def find_missing_numbers_from_set(number_set, count):
            # Ordenar el conjunto en una lista
            sorted_numbers = sorted(number_set)
            # Rango de búsqueda
            start = min(sorted_numbers)
            end = max(sorted_numbers)
            
            # Lista para almacenar los números faltantes
            missing_numbers = []
            
            # Iterar por el rango y detener cuando se alcance el límite
            # for num in range(start, end + 1):
            for num in range(end, start - 1, -1):  # Paso negativo para ir en orden descendente
                if num not in number_set:
                    missing_numbers.append(num)
                    if len(missing_numbers) == count:
                        break  # Detener cuando se encuentren los primeros 'count' números faltantes

            return missing_numbers
        
        #  >>>>>>>>>> get free indices <<<<<<<<<<
        # make a deep copy of cells
        cells_copy = copy.deepcopy(cells)
        # Crear un conjunto vacío para almacenar los índices
        buzy = set()
        # Agregar todos los índices enteros de cells_copy al conjunto buzy
        for cell in cells_copy:
            buzy.update(cell)
        free_indices = find_missing_numbers_from_set(buzy, 8)
                
        pxl_path = pxl_path_to_appdata()
        filepath = os.path.join(pxl_path, "A_codes.txt")
        A_codes = read_vector_from_file(filename=filepath)
        n1 = len(A_codes)
        A_codes = A_codes + free_indices

        A_dict = {num: idx for idx, num in enumerate(A_codes)}
        
        # cells refactoring
        index_code = 64
        for index in range(17, 25):
            # mapping 17 -> 64,  24 -> 71
            cells[index][-1] = A_codes[index_code]
            # cells[index][-1] = A_codes[index + 47]
            index_code += 1
            pass
        
        # cells refactoring
        index_code = 64
        for index in range(42, 58, 2):
            # mapping 42 -> 64,  56 -> 71
            cells[index][0] = A_codes[index_code]
            index_code += 1
            pass

        # id_negatives = []
        # id = 0
        # for node_id, y, angle in pavilion_locations:                        
        #     if y < 0:
        #         id_negatives.append(id)
        #     id += 1
        #     pass
        
        # if len(id_negatives) > 0:
        #     y_prom = 0
        #     num = 0
        #     for node_id, y, angle in pavilion_locations:
        #         if y > 0:                        
        #             y_prom += y
        #             num += 1
        #         pass
        #     y_prom /= num
        #     id = 0
        #     for node_id, y, angle in pavilion_locations:                        
        #         if y < 0:
        #             pass
        #             # pavilion_locations[id][1] = y_prom
        #         id += 1
        #         pass

        # # Step 1: Detect keys with a negative component in the second position of the tuple
        # negative_keys = [key for key, value in pavilion_locations.items() if value[1] < 0]
        # # print("Keys with negative second component:", negative_keys)

        # # Step 2: Extract positive values from the second position of tuples
        # positive_values = [value[1] for key, value in pavilion_locations.items() if value[1] > 0]

        # # Step 3: Calculate the average of the positive values
        # average_value = sum(positive_values) / len(positive_values)
        # # print("Calculated average of positive values:", average_value)

        # # Step 4: Update negative values in the dictionary with the calculated average
        # for key in negative_keys:
        #     # Replace the second element of the tuple with the calculated average
        #     pavilion_locations[key] = (pavilion_locations[key][0], average_value, pavilion_locations[key][2])
    

        # Step 1: Extract the second element from tuples with positive values
        positive_values = [location[1] for location in pavilion_locations if location[1] > 0]

        # Step 2: Calculate the average of the positive values
        average_value = sum(positive_values) / len(positive_values)
        if enable_print:
            print("Calculated average of positive values:", average_value)

        # Step 3: Replace negative values with the calculated average
        pavilion_locations = [
            (location[0], average_value if location[1] < 0 else location[1], location[2])
            for location in pavilion_locations]

        y_coord = []
        y_node = []

        for node_id, y, angle in pavilion_locations:                        
            # Get pavilion location
            # result = PointInterception(self.fast_surface, y, angle, p3)
            y_coord.append(y)
            y_node.append(node_id)
            pass

        # tuplas de puntos alineados en cada radial de pabellón iniciando la primera radial a 22.5º    
        # por ejemplo nodo con índice 64 de primera tupla tiene la misma posición que nodo con índice 16
        # esta posición se copia en código que sigue a esta definición. 
        pavilion_tuples_on_line = \
        [(49, 24, 16, 64),
        (51, 25, 17, 65),
        (53, 26, 18, 66),
        (55, 27, 19, 67),
        (57, 28, 20, 68),
        (59, 29, 21, 69),
        (61, 30, 22, 70),
        (63, 31, 23, 71)]
        # ecualización de puntos indexados por dos últimos elementos de las tuplas
        id = 0
        for tupla in pavilion_tuples_on_line:
            i1 = tupla[0]
            i2 = tupla[1]
            i3 = tupla[2]
            i4 = tupla[3]
            p0 = self.points_array[A_codes[i1]] 
            p1 = self.points_array[A_codes[i2]] 
            # p0 = self.points_array[A_codes[tupla[0]]] 
            # p1 = self.points_array[A_codes[tupla[1]]] 
            y = y_coord[id]
            node_y = y_node[id] 
            t = (y - p0[1])/ (p1[1] - p0[1])
            x = p0[0] + t * (p1[0] - p0[0])
            z = p0[2] + t * (p1[2] - p0[2])
            if True:
                self.points_array[A_codes[i4]] = np.array([x, y, z])
                self.points_array[A_codes[i3]] = np.array([x, y, z])
            else:
                vector = np.array([x,0,z])
                norm = np.linalg.norm(vector)
                vector  = vector / norm
                t2 = 0.00
                # self.points_array[A_codes[i4]] = np.copy(self.points_array[A_codes[i3]]) 
                self.points_array[A_codes[i4]] = np.array([x, y, z]) + t2*vector
                self.points_array[A_codes[i3]] = np.array([x, y, z]) + t2*vector
            id += 1

        pavilion_tuples_rhomb = \
        [
            (48, 31, 23, 64, 16), # v1 v2 v3 v4 v3'(neighboor v4) 
            (50, 24, 16, 65, 17),
            (52, 25, 17, 66, 18),
            (54, 26, 18, 67, 19),
            (56, 27, 19, 68, 20),
            (58, 28, 20, 69, 21),
            (60, 29, 21, 70, 22),
            (62, 30, 22, 71, 23)]

        previous = []        
        if True:
            # t1 = [0.04]
            # results = {}
            for k in range(0, 8):
                tupla = pavilion_tuples_rhomb[k]
                i1 = tupla[0]
                i2 = tupla[1]
                i3 = tupla[2]
                i4 = tupla[3]
                v1 = self.points_array[A_codes[tupla[0]]] 
                v2 = self.points_array[A_codes[tupla[1]]] 
                v3 = self.points_array[A_codes[tupla[2]]] 
                v4 = self.points_array[A_codes[tupla[3]]] 
                # _v1, _v2, v3, v4 = adjust_points_to_same_plane_([v1,v2,v3,v4], tolerance=0.1, tm=t1)
                _v1, _v2, v3, v4 = adjust_points_to_same_plane([v1,v2,v3,v4])
                previous.append([v1, v2, v3, v4]) 
                self.points_array[A_codes[tupla[2]]] = np.array(v3)
                self.points_array[A_codes[tupla[3]]] = np.array(v4)
                # results[k] = [A_codes[tupla[2]], A_codes[tupla[3]], np.array(v3), np.array(v4)]
                # if t1[0] > 0:
                #     self.points_array[A_codes[tupla[2]]] = np.array(v3)
                #     self.points_array[A_codes[tupla[3]]] = np.array(v4)
                # results[k] = [A_codes[tupla[2]], A_codes[tupla[3]], np.array(v3), np.array(v4), t1[0]]
                # self.points_array[A_codes[i4]] = np.copy(self.points_array[A_codes[i3]]) 
                # self.points_array[A_codes[i4]] = np.array([x, y, z])
                # self.points_array[A_codes[i3]] = np.array([x, y, z])

        
        pavilion_tuples_on_line_analysis = []
        is_above = True
        # inv_cells = [[A_dict.get(num, num) for num in sublist] for sublist in cells]
        for tupla in pavilion_tuples_on_line:
            '''
            pavilion_tuples_on_line = \
            [(49, 24, 16, 64),
            (51, 25, 17, 65),
            (53, 26, 18, 66),
            (55, 27, 19, 67),
            (57, 28, 20, 68),
            (59, 29, 21, 69),
            (61, 30, 22, 70),
            (63, 31, 23, 71)]
            '''
            v1 = self.points_array[A_codes[tupla[0]]] 
            v2 = self.points_array[A_codes[tupla[2]]] 
            v3 = self.points_array[A_codes[tupla[1]]] 
            
            result_dict = tri_points_angle(v1, v2, v3)
            if not result_dict['is_above']:
                is_above = False
            pavilion_tuples_on_line_analysis.append(result_dict)
        
        if not is_above:
            pass
        return is_above

    def precise_3d_modeling(self):
        # 
        self.extract_vertices_to_facets(self.fast_surface)
        self.extract_facetas_to_vertices(self.fast_surface)
        self.extract_polydata_information(self.fast_surface)
        if enable_print:
            print('---------------------------------------------------------------------------------')
            print('---------------------------------------------------------------------------------')
        
        # DLL calling
        transfer_fast_3d_model_arrays(self.points_array, self.normals_array, self.cell_type_array, \
                                        self.vertice_facetas, self.faceta_vertices, self.image_qtty, \
                                        __class__.case_id)
        
        # save points_lower_peaks and points_upper_peaks in CSV format
        # this CSV file is read by DLL in the execcution of the next instrucción.
        # POINTS USED IN GIRDLE HANDLING ARE PREVIOUSLY CENTERED BY THE TABLE HANDLER
        self.girdle_handler.both_true_girdle_peek_detection(__class__.case_id, save=True, plot=False, path=None)
        
        # DLL calling
        cells, pavilion_locations, table_center, minAngle = transfer_precise_3d_model(__class__.case_id)
        
        
        
        # we have now
        # self.pixel_size, table_center, minAngle
        
        # converting "y" coordinate of pavilion_locations to mm
        # pavilion_locations = [
        #     (node_id, y * self.pixel_size - table_center[1], angle) 
        #     for node_id, y, angle in pavilion_locations
        # ]
        
        # move and rotate fast_surface
        # fast_surface HAS NOT BEEN YET CENTERED USING TABLE HANDLER VENTER
        self.fast_surface = Pixel3D.pxl_tools.transform_polydata(self.fast_surface, -table_center[0], -table_center[1], -table_center[2], minAngle)

        # move and rotate point_array
        Pixel3D.pxl_tools.move_xyz(self.points_array, -table_center[0], -table_center[1], -table_center[2])
        Pixel3D.pxl_tools.rotate_around_y(self.points_array, minAngle)
                
        # relocate rhomboid pavilion points to precise points
        is_above = self.relocate_pavilion_points_new(pavilion_locations, cells, table_center)

        self.add_girdle_spheres(table_center, minAngle)

        # final processing of precise surface
        self.precise_surface = Pixel3D.pxl_tools.create_vtk_polydata(self.points_array, cells)
        # self.set_surface(surface=self.precise_surface)

        # surface = execute_1(__class__.case_id)
        # self.set_surface(surface=surface)
        return is_above


    def add_girdle_spheres(self, table_center, minAngle):

        code_id = __class__.case_id

        if False:
            gh.load_data(code_id)
            girdle_lower_points_3d, girdle_upper_points_3d = gh.get_girdle_3d_positioning()

            # peaks_lower_girdle_top, peaks_lower_girdle_botton, peaks_upper_girdle_top, peaks_upper_girdle_botton
            lower_girdle_top_peaks, lower_girdle_botton_peaks, _, _ = gh.all_peak_detection()
            
            lower_girdle_top_3d_peaks = girdle_lower_points_3d[lower_girdle_top_peaks]
            lower_girdle_botton_3d_peaks = girdle_lower_points_3d[lower_girdle_botton_peaks]
            
            positions = lower_girdle_top_3d_peaks

            positions = \
                Pixel3D.pxl_tools.translate_points(positions, (-table_center[0], -table_center[1], -table_center[2]))
        
            positions = \
                Pixel3D.pxl_tools.rotate_points_around_y(positions, minAngle)
        elif False:        
            girdle_lower_points_3d, girdle_upper_points_3d = \
                GirdleHandler.both_true_girdle_peek_detection(code_id, save=False, plot=False, path=None)

        radio = 11.72 / 2000.0
        radio = 0.05

        self.girdle_handler.reset_components()
        sphere_actor = self.girdle_handler.create_colored_spheres(self.girdle_handler.points_lower_peaks, 1, radio)
        sphere_actor = self.girdle_handler.create_colored_spheres(self.girdle_handler.points_upper_peaks, 4, radio)
        
        if self.girdle_spheres_actor is None:
            self.girdle_spheres_actor = sphere_actor
            # self.first_renderer.RemoveActor(sphere_actor)
            self.first_renderer.AddActor(self.girdle_spheres_actor)    
    
    def extract_vertices_to_facets(self, polydata):
        """
        Function to get the facets associated with each vertex in a vtkPolyData object.
        
        Parameters:
        polydata (vtk.vtkPolyData): The input polydata from which to extract the vertex-facet mapping.
        """
        # Get the number of cells and points in the polydata
        num_cells = polydata.GetNumberOfCells()
        num_points = polydata.GetNumberOfPoints()
        
        # Clear the global vertex-facet map
        self.vertice_facetas.clear()
        
        # Initialize the map with empty lists for each vertex
        for i in range(num_points):
            self.vertice_facetas[i] = []
        
        # Fill the map with the facets (cell ids) that share each vertex
        for i in range(num_cells):
            cell = polydata.GetCell(i)
            cell_points = cell.GetPointIds()
            
            for j in range(cell_points.GetNumberOfIds()):
                point_id = cell_points.GetId(j)
                self.vertice_facetas[int(point_id)].append(int(i))


    def extract_facetas_to_vertices(self, polydata):
        """
        Obtiene los vértices asociados a cada faceta (célula) de un vtkPolyData.

        Parameters:
        polydata (vtk.vtkPolyData): El objeto vtkPolyData que contiene las facetas.

        Returns:
        dict: Un diccionario donde la clave es la ID de la faceta (célula) 
            y el valor es una lista de IDs de los vértices asociados a esa faceta.
        """
        # Inicializar el diccionario de faceta-vertices
        self.faceta_vertices.clear()

        # Obtener la cantidad de facetas (células) y puntos (vértices)
        num_cells = polydata.GetNumberOfCells()
        num_points = polydata.GetNumberOfPoints()

        # Inicializar el diccionario con una lista vacía para cada faceta
        for i in range(num_cells):
            self.faceta_vertices[i] = []

        # Rellenar el diccionario con los vértices (puntos) asociados a cada faceta (célula)
        for i in range(num_cells):
            cell = polydata.GetCell(i)
            cell_points = cell.GetPointIds()
            for j in range(cell_points.GetNumberOfIds()):
                point_id = cell_points.GetId(j)
                self.faceta_vertices[i].append(point_id)


    def extract_polydata_information(self, polydata:vtk.vtkPolyData):
        """
        Extract points, normals, and cell types from vtkPolyData.
        
        Parameters:
        polydata (vtk.vtkPolyData): The input polydata from which to extract points, normals, and cell types.

        output arrays:
        points_array, normals_array, cell_type_array
        """

        if polydata.GetCellData().GetNormals():
            return
                
        # Compute normals for the polyData
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(polydata)
        normals.SplittingOff()
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOn()
        normals.Update()
        polydata.DeepCopy(normals.GetOutput())
        # polydata = normals.GetOutput()
        
        # Verify if normals exist
        if not polydata.GetCellData().GetNormals():
            normal_generator = vtk.vtkPolyDataNormals()
            normal_generator.SetInputData(polydata)
            normal_generator.ComputeCellNormalsOn()
            normal_generator.SplittingOff()
            normal_generator.Update()
            polydata = normal_generator.GetOutput()
        
        # Get number of faces in polyData
        num_faces = polydata.GetNumberOfCells()
        cell_normals = polydata.GetCellData().GetNormals()
        
        n_cells = int(num_faces)
        cell_size = cell_normals.GetNumberOfComponents()
        
        # Resize and initialize the arrays
        self.normals_array.clear()
        self.normals_array = [[0.0] * cell_size for _ in range(n_cells)]
        self.cell_type_array.clear()
        self.cell_type_array = [0] * n_cells
        
        # Fill normals_array with cellNormals
        for i in range(n_cells):
            normal = [0.0] * cell_size
            cell_normals.GetTuple(i, normal)
            for j in range(cell_size):
                self.normals_array[i][j] = normal[j]
            
            # Get the cell type using GetCellType
            self.cell_type_array[i] = polydata.GetCellType(i)
        
        # Extract points from vtkPolyData and store in the vector
        number_of_points = polydata.GetNumberOfPoints()
        self.points_array.clear()
        for i in range(number_of_points):
            point = [0.0, 0.0, 0.0]
            polydata.GetPoint(i, point)
            self.points_array.append([point[0], point[1], point[2]])
            
        self.points_array = [np.array(point) for point in self.points_array]
        self.normals_array = [np.array(point) for point in self.normals_array]
        # valor = __class__.get_elevation(self.normals_array[100])
        pass

    @staticmethod
    def create_vtk_polydata_from_flat_list(flat_coords):
        # Crear un vtkPoints para almacenar los puntos
        points = vtk.vtkPoints()

        # Agregar los puntos a vtkPoints
        for i in range(0, len(flat_coords), 3):
            x, y, z = flat_coords[i], flat_coords[i+1], flat_coords[i+2]
            points.InsertNextPoint(x, y, z)

        # Crear una vtkCellArray para almacenar la polilínea cerrada
        lines = vtk.vtkCellArray()
        num_points = len(flat_coords) // 3

        # Definir la polilínea cerrada
        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(num_points + 1)
        for i in range(num_points):
            polyline.GetPointIds().SetId(i, i)
        polyline.GetPointIds().SetId(num_points, 0)  # Cerrar la polilínea

        # Agregar la polilínea a vtkCellArray
        lines.InsertNextCell(polyline)

        # Crear vtkPolyData y asignar puntos y celdas
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)
        return polydata

    def cleaned_polygons_calibration(self):
        '''
        cleaned_polygons is a list of polygons created from the get_cleaned_points_from_decimated() 
        method, which is in pixel coordinates.
        
        This method flattens that list, uses only half of the polygons (the other half 
        is currently empty), calibrates to mm, polygons are centered in the X direction.
        '''

        dx = self.w / 2.0 + self.center_offset

        self.xmin = sys.float_info.max
        self.ymin = sys.float_info.max
        self.xmax = sys.float_info.min
        self.ymax = sys.float_info.min
        
        polys = []
        n_polys = int(len(self.cleaned_polygons)/2)
        
        
        is_list = isinstance(self.cleaned_polygons[0][0], list)

        if is_list:
            print("***************** IT IS A LIST *****************")
        
        for k in range(n_polys):
            polygon = self.cleaned_polygons[k]
            # Leer el número de puntos en la poligonal
            num_points = int(len(polygon)/3)
            poly = []
            for i in range(num_points - 1):
                # Leer las coordenadas del punto
                point = [polygon[3*i], polygon[3*i+1], polygon[3*i+2]]
                # point processing
                point[0] = point[0] - dx
                point = [x * self.pixel_size for x in point]
                # extreme values
                self.xmin = min(self.xmin, point[0])
                self.ymin = min(self.ymin, point[1])
                self.xmax = max(self.xmax, point[0])
                self.ymax = max(self.ymax, point[1])
                poly.append(point)
                pass
            poly = [(x, y) for x, y, z in poly]
            poly, _ = remove_non_convex(poly, ccw_verification=False)
            poly = [[x, y, 0.0] for x, y in poly]
            polys.append(poly)
        self.cleaned_polygons = polys
        pass


    # Example usage
    def test_polydata_info(self):
        # Create an example vtkPolyData (for testing purposes only)
        sphere_source = vtk.vtkSphereSource()
        sphere_source.Update()
        polydata = sphere_source.GetOutput()
        
        # Call the function to generate the vertex-facet mapping
        self.vertices_to_facetas(polydata)

        # Extract information from polydata
        self.extract_polydata_information(polydata)
        
        if False:
            # Print the results (for testing purposes only)
            print("Vertex-Facet Mapping:")
            for vertex_id, facet_ids in self.vertice_facetas.items():
                print(f"Vertex {vertex_id}: Facets {facet_ids}")
            
            print("\nPoints Array:")
            for i, point in enumerate(self.points_array[:5]):  # Print only the first 5 points for brevity
                print(f"Point {i}: {point}")
            
            print("\nNormals Array:")
            for i, normal in enumerate(self.normals_array[:5]):  # Print only the first 5 normals for brevity
                print(f"Normal {i}: {normal}")
            
            print("\nCell Types Array:")
            for i, cell_type in enumerate(self.cell_type_array[:5]):  # Print only the first 5 cell types for brevity
                print(f"Cell {i}: Type {cell_type}")
    
    
def example_split_polydata_by_elevation():
    from pxl_vtk_tools import read_as_vtp
    
    path = r"C:\Users\monti\AppData\Local\PixelPolish3D\LHPO_Round_Polished_400_0002_11p7_2024-08-03.vtp"
    polydata = read_as_vtp(path)
    
    # Compute normals
    normal_generator = vtk.vtkPolyDataNormals()
    normal_generator.SetInputData(polydata)
    normal_generator.ComputePointNormalsOff()
    normal_generator.ComputeCellNormalsOn()
    normal_generator.Update()

    # Get the processed vtkPolyData
    polydata_with_normals = normal_generator.GetOutput()

    # Split the polydata into three surfaces
    s1, s2, s3 = PxlModeling.split_polydata_by_elevation(polydata_with_normals)

    if enable_print:
        print("PolyData split into three surfaces:")
        print(f"s1 (negative elevation): {s1.GetNumberOfCells()} cells")
        print(f"s2 (zero elevation): {s2.GetNumberOfCells()} cells")
        print(f"s3 (positive elevation): {s3.GetNumberOfCells()} cells")
    return s1
        
# *************************************************************
# *************************************************************

class QFrameMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the main window
        self.setWindowTitle("VTK in QFrame with PySide6")
        # self.setGeometry(100, 100, 800, 600)

        # Create a central widget
        self.widget = QFrame(self)
        self.setCentralWidget(self.widget)

        # Create a layout for the central widget
        self.layout = QVBoxLayout(self.widget)
        
        self.renderer = None



# Run the application
if __name__ == "__main__":
    
    # example_split_polydata_by_elevation()
    # exit()

    app = QApplication(sys.argv)

    # window app
    window = QFrameMainWindow()
    window.showMaximized()
    window.show()

    # vtk rendering    
    pxl_vis = PxlModeling(window.widget, window.layout)
    s1 = example_split_polydata_by_elevation()
    pxl_vis.set_surface(s1)
    pxl_vis.interactor.Start()
    sys.exit(app.exec())


