
import sys
import os
import time
import gc
import cv2

from PySide6 import QtWidgets, QtCore 
from PySide6.QtWidgets import QLabel, QVBoxLayout, QFrame
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow
from pixelpolish3d_ui_v01 import Ui_MainWindow  # Importa la clase generada

import pxl_tools
from pxl_vtk import PxlModeling, State


class AppMainWindow(QMainWindow): 

    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)  # Configure the interface
        self.Qt_MainWindow = self

        self.folderpath = None

        # Connecting button to callback
        self.ui.actionSelect_Folder.triggered.connect(self.OnSelectFolder)        
        self.ui.actionSelect_CSV_File.triggered.connect(self.OnSelectSameFolder)        
        
        self.ui.toolButton_2d_modeling.clicked.connect(self.OnModeling2D)
        self.ui.toolButton_3d_modeling.clicked.connect(self.OnFastModel3D)
        self.ui.toolButton_simulator.clicked.connect(self.OnPreciseModel3D)
        self.ui.toolButton_quit.clicked.connect(self._quit)
        
        # self.ui.toolButton_action_1.clicked.connect(self.OnGo)
        # self.ui.toolButton_action_3.clicked.connect(self.OnViewGroups)
        # self.ui.toolButton_action_4.clicked.connect(self.OnViewSurface)
        # self.ui.toolButton_action_2.clicked.connect(self.OnViewSplits)

        def set_frame_layout(frame:QFrame):
            layout = QtWidgets.QVBoxLayout()
            layout.setContentsMargins(3,3,3,3)
            frame.setLayout(layout)
            return layout

        self.vlayout_1 = set_frame_layout(self.ui.frame_1)

        ''' 
        STEP #3 FOR INTEGRATION (Processing pipeline):
        pxl_main.py represents your own application.
        The following code must be added to your application.
        A single PxlModeling() component must be added to your code.
        It will remain idle without displaying anything (actors disabled).
        pxl_vis is the processing and modeling pipeline. 
        '''     
        # Here is your image repository (list of numpy.ndarray files), 
        # It could be set to your component name.
        self.images = []
        # Here is created the main component for 2D and 3D processing and visualization (PxlModeling)
        # Very important: 
        #   Your code must provide the QFrame arguments and a QVBoxLayout, any other layout can be used.
        #   The PxlModeling visualization is displayed in this QFrame of your application.
        self.pxl_vis = PxlModeling(self.ui.frame_1, self.vlayout_1)
        # Set to None to select cube as initial surface
        self.pxl_vis.set_surface(None)
        # All actors are kept invisible except the 3D final result
        self.pxl_vis.actors_off()
        # Start the visualization machine (VTK) 
        self.pxl_vis.interactor.Start()
        # This is optional and is used to display the runtimes, the printQ() code must be added.
        # Create a component to display messages using the printQ() method.
        self.label = None
        self.BuildPrint()     
        # The following callback called OnGo() must be associated with a button in your application.   
        self.ui.toolButton_action_1.clicked.connect(self.OnGo)


    def BuildPrint(self):
        self.label = QLabel("", self.ui.frame_1_toolbar)
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.label.setStyleSheet("color: black; font-size: 14px; font-weight: bold;")
        self.label.move(5, 5) 
        layout = self.ui.frame_1_toolbar.layout()
        if layout is None:
            layout = QVBoxLayout(self.ui.frame_1_toolbar)
            self.ui.frame_1_toolbar.setLayout(layout)
        layout.addWidget(self.label)
        self.ui.frame_1_toolbar.setVisible(True)
        self.ui.frame_1_toolbar.update()

    def printQ(self, txt:str): 
        if self.label is None:
            return
        if hasattr(self, 'label'):
            self.label.setText(txt)
       
                
    '''
    internal use
    callback for "Select Folder" menu
    '''
    # def on_select_folder(self):
    def OnSelectFolder(self, same_folder=None):
        """
        Handles folder selection and image loading based on pipeline configuration.
        """
        if same_folder is None or same_folder is False:
            self.folderpath = pxl_tools.loader_folder_backup()
            self.folderpath = QtWidgets.QFileDialog.getExistingDirectory(
                self.Qt_MainWindow, 
                'Select a folder with diamond images', 
                self.folderpath)
            pxl_tools.loader_folder_backup(self.folderpath)

        # If a folder is selected, update the folder path and load images
        if self.folderpath != "" and os.path.exists(self.folderpath):

            # advtools.loader_folder_backup(self.folderpath)

            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            start_time = time.time()

            # Initialize output variables
            y_table = [-1] # Simulate an output parameter
            max_val = [-1] # Simulate an output parameter

            # Check if the folder exists and process it
            if os.path.exists(self.folderpath):

                image_paths = [os.path.join(self.folderpath, f)
                               for f in os.listdir(self.folderpath)
                               if os.path.splitext(f)[1].lower()==".bmp"
                               and os.path.isfile(os.path.join(self.folderpath, f))]

                if len(image_paths) == 0:
                    image_paths = [os.path.join(self.folderpath, f)
                                for f in os.listdir(self.folderpath)
                                if os.path.splitext(f)[1].lower()==".jpg"
                                and os.path.isfile(os.path.join(self.folderpath, f))]
                
                option = 2
                if option==1:
                    self.pxl_vis.free_memory()
                    self.pxl_vis.ImagesReader(image_paths, self.pxl_vis)
                elif option==2:
                    '''
                    STEP #4 FOR INTEGRATION (Images proividing):
                    The following code provides the images to the pxl_vis processing pipeline 
                    using the self.pxl_vis.SetInputs() method.
                    Instructions marked with *** can be replaced at your discretion to fit your own code.                                        
                    '''
                    # Frees all occupied memory space except for the input images provided by ImagesProvider()
                    self.pxl_vis.free_memory()
                    # *** Frees the memory space occupied by the images provided by ImagesProvider()
                    for image in self.images:
                        image = None
                    self.images.clear()
                    # All actors are kept invisible except the 3D final result
                    self.pxl_vis.actors_off()
                    # Update screen output to empty
                    self.pxl_vis.render_window.Render()
                    # Get your own images and code_id, 
                    # *** This line can be overridden by your own method for handling grayscale input images.
                    self.images, code_id = self.ImagesProvider(image_paths)
                    # 
                    self.pxl_vis.SetInputs(self.images, code_id)
                    # self.pxl_vis.actors_off()
                    # self.pxl_vis.render_window.Render()
                    # Force garbage collection to free memory
                    gc.collect()       

            # Calculate and display the time lapse
            lapse = round(time.time() - start_time, 1)
            QtWidgets.QApplication.restoreOverrideCursor()

            self.printQ("Loading: " + str(lapse) + " sec")
            self.pxl_vis.state = State.LOADED
            self.pxl_vis.ResetCameraObject()

            pass
    
    '''
    internal use
    callback for "Select Folder" menu
    '''
    # def on_select_folder(self):
    def OnSelectSameFolder(self):
        """
        Handles folder selection and image loading based on pipeline configuration.
        """
        self.folderpath = pxl_tools.loader_folder_backup()
        self.OnSelectFolder(same_folder=self.folderpath)
    
    def OnModeling2D(self):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        start_time = time.time()
        
        self.pxl_vis.modeling_2d(self.pxl_vis)
        self.pxl_vis.state = State.IMAGING
        self.pxl_vis.select_camera(0)
        self.pxl_vis.cleaned_polygons_calibration()
        
        lapse = round(time.time() - start_time, 1)
        self.printQ("Image processing: " + str(lapse) + " sec")
        QtWidgets.QApplication.restoreOverrideCursor()
        self.pxl_vis.ResetCameraObject()

    def OnFastModel3D(self):  # 
        start_time = time.time()

        self.pxl_vis.fast_3d_modeling() 
        self.pxl_vis.state = State.FAST_MODEL
        self.pxl_vis.set_surface(surface=self.pxl_vis.fast_surface)

        lapse = round(time.time() - start_time, 1)
        self.printQ("Fast modeling: " + str(lapse) + " sec")
    
    def OnPreciseModel3D(self):  # 
        start_time = time.time()

        is_above = self.pxl_vis.precise_3d_modeling()
        self.pxl_vis.state = State.PRECISE_MODEL
        self.pxl_vis.set_surface(surface=self.pxl_vis.precise_surface)        

        lapse = round(time.time() - start_time, 1)
        is_above = '' if is_above else '-' 
        self.printQ(f"Precise modeling{is_above}: {lapse} sec")
        # self.pxl_vis.girdle_spheres_actor.VisibilityOn()
    
    def _quit(self):
        exit(0)

    def OnSplit(self):
        self.pxl_vis.split_surface()
    
    def OnViewGroups(self):
        self.pxl_vis.ViewGroups()

    def OnViewSurface(self):
        self.pxl_vis.ViewSurface()
    
    def OnViewSplits(self):
        self.pxl_vis.ViewSplits()
        # self.pxl_vis.SwitchEdgeVisibility()
        
        
    ''' 
    STEP #2 FOR INTEGRATION (Two new methods):
    pxl_main.py represents your own application.
    The ImagesProvider() and OnGo() methods must be added to your application.
    
    NOTE: 
    The method PxlModeling.get_parent_of_images_folder() provides a code_id 
    and it can be overridden by your own method. 
    The code_id used in this application is the name of the parent 
    of the folder containing the images, for example 
    "LHPO_Certified_Round_Polished_400_0001_11p7_2024-08-03"
    '''     
        
    def ImagesProvider(self, image_paths:list):
        """
        Provides grayscale images and a code identifier based on the given image paths.
        
        NOTE:
        This method should be overridden by your own method of handling grayscale images.

        This method processes a list of image file paths, reads the images in grayscale format,
        and collects them into a list. Additionally, it retrieves a unique code identifier
        associated with the folder containing the first image in the list.

        Args:
            image_paths (list): A list of file paths to the images to be processed.

        Returns:
            tuple:
                - images (list): A list containing the grayscale images. If an image cannot be
                read, it is skipped.
                - code_id (str): A unique identifier obtained from the parent folder of the 
                first image path in the list.

        NOTE:
            - The method assumes all paths in `image_paths` are valid strings.
            - If the list is empty or the first path is invalid, the behavior of the `code_id` 
            generation may depend on the implementation of `PxlModeling.get_parent_of_images_folder`.

        Example:
            image_paths = ["path/to/image1.png", "path/to/image2.png"]
            images, code_id = instance.ImagesProvider(image_paths)
            print(f"Number of images: {len(images)}, Folder code: {code_id}")
            """        
        
        images = []
        for path in image_paths:
            gray_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if gray_image is not None:
                images.append(gray_image)
                
        code_id = PxlModeling.get_parent_of_images_folder(image_paths[0])
        return images, code_id
    
    
    def OnGo(self):
        """
        Executes the precise modeling process and logs the time taken.

        This method performs the following tasks:
        1. Sets the application cursor to a waiting state.
        2. Initiates the precise modeling process using `PxlModeling.Go()` with the provided
        `pxl_vis` attribute.
        3. Calculates and logs the time taken for the modeling process in seconds.
        4. Restores the application cursor to its default state.

        Args:
            None

        Returns:
            None

        Example:
            instance.OnGo()
            # Outputs a log similar to "Precise modeling: 3.2 sec" in a QFrame.
        """
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        start_time = time.time()

        is_above = PxlModeling.Go(self.pxl_vis)
        is_above = '' if is_above else '-' 

        lapse = round(time.time() - start_time, 1)
        self.printQ(f"Precise modeling{is_above}: {lapse} sec")
        QtWidgets.QApplication.restoreOverrideCursor()

        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # window app
    window = AppMainWindow()
    window.showMaximized()
    window.show()
    sys.exit(app.exec())



