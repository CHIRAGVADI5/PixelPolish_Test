import os
import numpy as np
import pandas as pd
import csv
import time
import matplotlib.pyplot as plt
from Pixel3D.pxl_tools import pxl_path_to_appdata, peak_detector
import vtk
from Pixel3D.pxl_girdle_proc import is_sequence_valid, correct_sequence

enable_print = False

class GirdleHandler():

    def __init__(self): # , code_id, filename = None, path = None):
        '''
        Step #1:
        # code_id example: "LHPO_Round_Polished_400_0002_11p7_2024-08-03"
        '''
        
        # if filename is None:
        #     filename = "more_values_" + code_id + ".csv"
        
        # if path is None:
        #     pxl_path = pxl_path_to_appdata()
        #     self.file_path = os.path.join(path, filename)
        
        self.df:pd.DataFrame = None

        self.peaks_lower_girdle_top = None
        self.peaks_lower_girdle_botton = None
        self.peaks_upper_girdle_top = None
        self.peaks_upper_girdle_botton = None

        self.y_lower_girdle_smooth = None 
        self.y_upper_girdle_smooth = None 

        self.lower_points_3d = None
        self.upper_points_3d = None
        
        self.append_filter:vtk.vtkAppendPolyData = None
        self.sphere_actor:vtk.vtkActor = None
        self.sphere_mapper:vtk.vtkPolyDataMapper = None
        
        self.sphere_source:vtk.vtkSphereSource = None
        # Create the cell data array to hold integer indices for each sphere's cells
        self.cell_data_array:vtk.vtkIntArray = None
        self.transform:vtk.vtkTransform = None
        self.transform_filter:vtk.vtkTransformPolyDataFilter = None
        
        text_actor_list = []
        self.points_lower_peaks:np.ndarray = None
        self.points_upper_peaks:np.ndarray = None

    def reset_components(self):
        if self.append_filter is None:
            return
        # Reiniciar vtkAppendPolyData
        self.append_filter.RemoveAllInputs()
        # Reiniciar vtkIntArray
        self.cell_data_array.Initialize()
        
    def load_data(self, code_id, filename = None, path = None):
        '''
        Step #2:
        code_id example: "LHPO_Round_Polished_400_0002_11p7_2024-08-03"
        '''
                
        if filename is None:
            filename = "more_values_" + code_id + ".csv"
        
        if path is None:
            path = pxl_path_to_appdata()

        file_path = os.path.join(path, filename)
        
        # Check the file extension
        if file_path.endswith('.xlsx'):
            self.df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            self.df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a .xlsx or .csv file.")

    def get_girdle_3d_positioning(self):
        '''
        Step #3:
        '''
        # angle delta is 360/n for n 400 and 800
        # angles in the range 0º to 179.1º for df.index in the ranges (0, 199) and (200, 399)     
        n = self.df.shape[0]
        self.df['Angle'] = np.where(self.df.index < n // 2, -self.df.index * (360.0 / n), -(self.df.index - n // 2) * (360.0 / n))

        self.df["c"] = np.cos(np.radians(self.df['Angle']))
        self.df["s"] = np.sin(np.radians(self.df['Angle']))

        self.df["x_new"] = self.df["c"] * self.df["x_lower"]  # + self.df["s"] * self.df["z_lower"]
        self.df["z_new"] = -self.df["s"] * self.df["x_lower"] # + self.df["c"] * self.df["z_lower"]
        
        # lower_points = np.array([self.df["x_new"], self.df["y_lower"], self.df["z_new"]])
        # upper_points = np.array([self.df["x_new"], self.df["y_upper"], self.df["z_new"]])
        
        self.lower_points_3d = np.column_stack((self.df["x_new"], self.df["y_lower"], self.df["z_new"]))
        self.upper_points_3d = np.column_stack([self.df["x_new"], self.df["y_upper"], self.df["z_new"]])        
        return self.lower_points_3d, self.upper_points_3d

    def all_peak_detection(self):
        '''
        Step #4:
        '''
        self.peaks_lower_girdle_top, self.y_lower_girdle_smooth = peak_detector(self.df['y_lower'].values, positive=True)
        self.peaks_lower_girdle_botton, self.y_lower_girdle_smooth = peak_detector(self.df['y_lower'].values, positive=False)        
        self.peaks_upper_girdle_top, self.y_upper_girdle_smooth = peak_detector(self.df['y_upper'].values, positive=True)
        self.peaks_upper_girdle_botton, self.y_upper_girdle_smooth = peak_detector(self.df['y_upper'].values, positive=False)
        
        return self.peaks_lower_girdle_top, self.peaks_lower_girdle_botton, self.peaks_upper_girdle_top, self.peaks_upper_girdle_botton

    def create_colored_spheres(self, xyz:np.ndarray, index = 0, radio=0.05):
        """
        Create a vtkPolyData with colored spheres distributed around the origin.
        
        Parameters:
        xyz: (16,3) np.ndarray
        index: color index
        radio: 50 microns default
        
        Returns:
        vtkPolyData: A polydata containing the spheres with assigned unique integer indices per cell.
        """
        # Create a vtkAppendPolyData to hold all the spheres
        if self.append_filter is None:
            self.append_filter = vtk.vtkAppendPolyData()

        if self.sphere_actor is None:
            # Color lookup table with 9 colors (index 0 to 8)
            colors = vtk.vtkNamedColors()
            color_table = [
                # (0.6, 0.3, 0),   # brown
                # (1, 0, 0),       # red
                # (1, 0.5, 0),     # orange
                # (1, 1, 0),       # yellow
                # (0, 1, 0),       # green
                # (0, 0, 1),       # blue
                # (234/255, 51/255, 188/255),     # violet
                # (0.6, 0.6, 0.6), # gray
                colors.GetColor3d("Brown"),  # marrón
                colors.GetColor3d("Red"),          # rojo
                colors.GetColor3d("Orange"),       # naranja
                colors.GetColor3d("Yellow"),       # amarillo
                colors.GetColor3d("Green"),        # verde
                colors.GetColor3d("Blue"),         # azul
                colors.GetColor3d("Violet"),       # violeta
                colors.GetColor3d("Gray")          # gris
            ]

            # Crear una tabla de colores (LUT)
            lut = vtk.vtkLookupTable()
            lut.SetNumberOfTableValues(8)
            lut.Build()

            # Asignar los colores a las entradas del LUT
            for i, color in enumerate(color_table):
                lut.SetTableValue(i, color[0], color[1], color[2], 1.0)  # RGBA, con A = 1.0 para opacidad

            # Create mapper and actor to visualize the colored spheres
            self.sphere_mapper = vtk.vtkPolyDataMapper()
            self.sphere_mapper.SetInputConnection(self.append_filter.GetOutputPort())
            self.sphere_mapper.SetScalarRange(0, 7)  # Scalar range between 0 and 8
            # Asignar el LUT al mapper
            self.sphere_mapper.SetLookupTable(lut)
            # surface.GetCellData().SetActiveScalars("Azimuth")
            self.sphere_mapper.ScalarVisibilityOn()
            self.sphere_actor = vtk.vtkActor()
            self.sphere_actor.SetMapper(self.sphere_mapper)

        # Función para crear un BillboardTextActor3D
        def create_billboard_text(text, position):
            text_actor = vtk.vtkBillboardTextActor3D()
            text_actor.SetPosition(position)
            text_actor.SetInput(text)
            text_actor.GetTextProperty().SetFontSize(16)  # Reducir el tamaño a la mitad
            text_actor.GetTextProperty().BoldOn()
            text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)  # Color blanco
            text_actor.VisibilityOn()
            return text_actor
        
        if self.sphere_source is None:
            # Sphere source
            self.sphere_source = vtk.vtkSphereSource()
            self.sphere_source.SetRadius(radio)  # Radius (diameter = 0.020)
            self.sphere_source.SetPhiResolution(16)
            self.sphere_source.SetThetaResolution(16)
            self.sphere_source.Update()
        
        if self.cell_data_array is None:
            # Create the cell data array to hold integer indices for each sphere's cells
            self.cell_data_array = vtk.vtkIntArray()
            self.cell_data_array.SetName("Indices")
        
        # Number of cells in the template sphere
        num_cells_per_sphere = self.sphere_source.GetOutput().GetNumberOfCells()

        # Set position for the current sphere
        if self.transform_filter is None:
            self.transform = vtk.vtkTransform()
            # transform.Translate(x, y, z)
            self.transform_filter = vtk.vtkTransformPolyDataFilter()
            self.transform_filter.SetInputConnection(self.sphere_source.GetOutputPort())
            self.transform_filter.SetTransform(self.transform)
            # transform_filter.Update()

        # Distribute spheres around the origin in a circular manner
        # radius = 0.1  # Distance from origin for each sphere
        # angle_step = 360.0 / num_spheres

        for i in range(xyz.shape[0]):
            # Positioning each sphere
            # angle = np.radians(i * angle_step)
            # x = radius * np.cos(angle)
            # y = radius * np.sin(angle)
            # z = 0  # Place all spheres in the XY plane
            
            x,y,z = xyz[i]

            # Determinar el desplazamiento según la posición Y
            # if y < yRef:
            #     position = (x, y - 2*radio, z)  # Desplazamiento en -Y
            # else:
            #     position = (x, y + 2*radio, z)  # Desplazamiento en +Y

            # Set position for the current sphere
            self.transform.Identity()
            self.transform.Translate(x, y, z)
            self.transform_filter.Update()

            polydata = vtk.vtkPolyData() 
            polydata.DeepCopy(self.transform_filter.GetOutput())
            # Append transformed sphere to polydata
            self.append_filter.AddInputData(polydata)

            # Assign a unique integer between 0 and 8 to the cells of the sphere
            for j in range(num_cells_per_sphere):
                self.cell_data_array.InsertNextValue(index)  # Ensure the indices are between 0 and 8

            # Crear el texto con el índice
            # text = str(i)  # El índice de la esfera
            # text_actor = create_billboard_text(text, position)
            # self.text_actor_list.append(text_actor)

        # Combine all the spheres into one polydata
        self.append_filter.Update()
        polydata = self.append_filter.GetOutput()
        polydata.GetCellData().SetScalars(self.cell_data_array)
        return self.sphere_actor

    @staticmethod
    def rendering_pipeline():

        # Create a renderer (container for the objects to render)
        renderer = vtk.vtkRenderer()

        # Create a render window (visualization window)
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)  # Associate the renderer with the window

        # Create a render window interactor (interaction controller)
        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window_interactor.SetRenderWindow(render_window)  # Associate the window with the interactor

        # Set the interactor style (Trackball Camera)
        interactor_style = vtk.vtkInteractorStyleTrackballCamera()
        render_window_interactor.SetInteractorStyle(interactor_style)

        # Set renderer properties
        renderer.SetBackground(0.1, 0.1, 0.2)  # Set background color (dark blue)
        render_window.SetSize(600, 400)  # Set window size

        # Add an example object (a cube) to the scene
        # cube = vtk.vtkCubeSource()
        # cube_mapper = vtk.vtkPolyDataMapper()
        # cube_mapper.SetInputConnection(cube.GetOutputPort())
        # cube_actor = vtk.vtkActor()
        # cube_actor.SetMapper(cube_mapper)
        # renderer.AddActor(cube_actor)

        # Start rendering process
        # render_window.Render()  # Render the initial scene
        # render_window_interactor.Start()  # Start the interaction
        return renderer, render_window, render_window_interactor
        

    @staticmethod
    def previous_peek_detection(filename, path = None):
        if path is None:
                pxl_path = pxl_path_to_appdata()
        file_path = os.path.join(path, filename)

        # Check the file extension
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a .xlsx or .csv file.")
        
        use_lower = True
        positive = use_lower
        
        if use_lower:
            y = df['y_lower'].values  # Extraemos los valores de la columna 'y'
        else:
            y = df['y_upper'].values  # Extraemos los valores de la columna 'y'

        peaks, y_smooth = peak_detector(y, positive=positive)

        peaks_lower_girdle_top, y_lower_girdle_smooth = peak_detector(df['y_lower'].values, positive=True)
        peaks_lower_girdle_botton, y_lower_girdle_smooth = peak_detector(df['y_lower'].values, positive=False)
        peaks_upper_girdle_top, y_upper_girdle_smooth = peak_detector(df['y_upper'].values, positive=True)
        peaks_upper_girdle_botton, y_upper_girdle_smooth = peak_detector(df['y_upper'].values, positive=False)

        # angle delta is 360/n for n 400 and 800
        # angles in the range 0º to 179.1º for df.index in the ranges (0, 199) and (200, 399)     
        n = df.shape[0]
        df['Angle'] = np.where(df.index < n // 2, -df.index * (360.0 / n), -(df.index - n // 2) * (360.0 / n))

        df["c"] = np.cos(np.radians(df['Angle']))
        df["s"] = np.sin(np.radians(df['Angle']))

        df["x_new"] = df["c"] * df["x_lower"]  # + df["s"] * df["z_lower"]
        df["z_new"] = -df["s"] * df["x_lower"] # + df["c"] * df["z_lower"]
        
        # lower_points = np.array([df["x_new"], df["y_lower"], df["z_new"]])
        # upper_points = np.array([df["x_new"], df["y_upper"], df["z_new"]])
        
        lower_points = np.column_stack((df["x_new"], df["y_lower"], df["z_new"]))
        upper_points = np.column_stack([df["x_new"], df["y_upper"], df["z_new"]])
        
        # falta aplicar el desplazamiento y la rotación, obtenida en el posicionamiento de la tabla. 
        
        angle_index = peaks.tolist()
        girdle_values = y_smooth[peaks].tolist()

        # Visualizar los resultados
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, y, label='Datos originales', alpha=0.7)
        plt.plot(df.index, y_smooth, label='Datos suavizados', alpha=0.7)
        plt.plot(df.index[peaks], y[peaks], "x", label='Picos detectados', markersize=10, color='red')
        plt.title('Detección de picos en la señal')
        plt.xlabel('Índice')
        plt.ylabel('Valores')
        plt.legend()
        plt.grid()
        plt.show()

        # Mostrar las posiciones y valores de los picos detectados
        picos_detectados = pd.DataFrame({'Índice': peaks, 'Valores': y[peaks]})
        if enable_print:
            print(picos_detectados)

    @staticmethod
    def previous_girdle_table_to_excel(base_name, option=1, path = r"G:\2024\1-Bhavesh\Documentación\UI"):
        
        # Leer el archivo de texto
        file_path = os.path.join(path, base_name + ".txt")

        # Leer los datos del archivo y convertir cada línea en tupla
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                # Limpiar la línea, eliminando cualquier prefijo no deseado y los paréntesis
                line = line.strip().split(']')[1]  # Eliminar el índice antes de la tupla
                line = line.strip()[1:-1]  # Eliminar los paréntesis
                values = line.split(',')  # Dividir la línea por comas
                data.append(tuple(map(float, values)))  # Convertir los valores a float y agregar la tupla

        # Crear el DataFrame
        if option==1:
            columns = ["index", "x_upper", "x_lower", "y_upper", "y_lower", "height"]
        elif option==2:
            columns = ["index", "x_lower", "y_lower", "z_lower", "nx", "ny", "nz"]
            
        df = pd.DataFrame(data, columns=columns)

        # Guardar el DataFrame en un archivo Excel
        output_file = os.path.join(path, base_name + ".xlsx")
        df.to_excel(output_file, index=False)
        return output_file

    @staticmethod
    def previous_draw_from_excel(base_name, option=1, path = r"G:\2024\1-Bhavesh\Documentación\UI"):

        excel_path = os.path.join(path, base_name + ".xlsx")

        # Leer el archivo Excel y crear el DataFrame
        df = pd.read_excel(excel_path)

        # Graficar las variables "y_upper" y "y_lower"
        plt.figure(figsize=(10, 6))
        if option==1:
            # plt.plot(df['y_upper'], label='y_upper', color='blue', marker='o', linestyle='-', markersize=3)
            plt.plot(df['y_lower'], label='y_lower', color='red', marker='x', linestyle='--', markersize=3)
        elif option==2:
            # Calcular el ángulo de cada vector con el plano XZ
            # df['angle'] = np.degrees(np.arctan2(df['ny'], np.sqrt(df['nx']**2 + df['nz']**2)))
            # plt.plot(df['index'], df['angle'], marker='o', linestyle='-', color='b')
            # plt.plot(df['ny'], label='ny', color='blue', marker='o', linestyle='-', markersize=3)
            # plt.plot(df['nx'], label='nx', color='red', marker='x', linestyle='--', markersize=3)
            plt.plot(df['y_lower'], label='y_lower', color='red', marker='x', linestyle='--', markersize=3)

        # Añadir etiquetas y título
        plt.title('Gráfico de y_upper y y_lower')
        plt.xlabel('Índice')
        plt.ylabel('Valores')
        plt.legend()

        # Mostrar la gráfica
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def true_girdle_peek_detection(filename, path = None):

        if path is None:
            path = pxl_path_to_appdata()
        file_path = os.path.join(path, filename)

        # Check the file extension
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a .xlsx or .csv file.")
        
        is_lower_girdle = 'lower' in filename
        positive = is_lower_girdle
        
        y = df['y'].values  # Extraemos los valores de la columna 'y'

        peaks, y_smooth = peak_detector(y, positive=positive)

        points = np.column_stack((df["x"], df["y"], df["z"]))
        
        # falta aplicar el desplazamiento y la rotación, obtenida en el posicionamiento de la tabla. 
        
        angle_index = peaks.tolist()
        girdle_values = y_smooth[peaks].tolist()

        # Visualizar los resultados
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, y, label='Datos originales', alpha=0.7)
        plt.plot(df.index, y_smooth, label='Datos suavizados', alpha=0.7)
        plt.plot(df.index[peaks], y[peaks], "x", label='Picos detectados', markersize=10, color='red')
        plt.title('Detección de picos en la señal')
        plt.xlabel('Índice')
        plt.ylabel('Valores')
        plt.legend()
        plt.grid()
        plt.show()

        # Mostrar las posiciones y valores de los picos detectados
        picos_detectados = pd.DataFrame({'Índice': peaks, 'Valores': y[peaks]})
        if enable_print:
            print(picos_detectados)

    def both_true_girdle_peek_detection(self, code_id, save = False, plot = False, path = None):

        if path is None:
            path = pxl_path_to_appdata()

        lower_filepath = os.path.join(path, "lower_girdle_coords_" + code_id + ".csv")
        upper_filepath = os.path.join(path, "upper_girdle_coords_" + code_id + ".csv")

        df_lower = pd.read_csv(lower_filepath)
        df_upper = pd.read_csv(upper_filepath)
        
        '''
        angle	    node_id	    x	        y	        z
        0.101368	1348	    3.52418	    0.992425	0.00623501
        2.71321	    1347	    3.52152	    1.0081	    0.166884
        2.76047	    1777	    3.52138	    1.00831	    0.169789
        3.1783	    1778	    3.51974	    1.00988	    0.195447
        '''        
        
        # y_smooth_lower is always None
        peaks_lower, y_smooth_lower = peak_detector(df_lower['y'].values, positive=True)
        
        if peaks_lower.shape[0] == 15:
            signal = df_lower['y'].values
            # Revisar si los puntos alrededor de la posición 0 y 360 forman picos, considerando la conexión circular
            for i in range(-3, 4):  # Tres puntos antes y tres puntos después de la posición 0
                idx = i % len(signal)  # Índice circular, permite acceder a signal[-3] como signal[len(signal) - 3]
                prev_idx = (idx - 1) % len(signal)  # Índice anterior circular
                next_idx = (idx + 1) % len(signal)  # Índice siguiente circular
                
                if signal[idx] > signal[prev_idx] and signal[idx] > signal[next_idx]:
                    peaks_lower = np.insert(peaks_lower, 0, idx)  # Inserta el índice en la lista de picos si es mayor que sus vecinos
                    break

            # Eliminar picos duplicados y ordenar
            peaks_lower = np.unique(peaks_lower)

            # # Revisar si el punto de la posición 0 es un pico, considerando la conexión circular con 360
            # if signal[0] > signal[1] and signal[0] > signal[-1]:
            #     peaks_lower = np.insert(peaks_lower, 0, 0)  # Inserta el índice 0 al inicio de la lista de picos
            # if signal[-1] > signal[-2] and signal[-1] > signal[0]:
            #     peaks_lower = np.append(peaks_lower, len(signal) - 1)  # Agregar el índice final como un pico
            # # Eliminar picos duplicados
            # peaks_lower = np.unique(peaks_lower)            
        
        peaks_upper, y_smooth_upper = peak_detector(df_upper['y'].values, positive=False)

        if peaks_upper.shape[0] == 15:
            signal = df_upper['y'].values
            # Revisar si los puntos alrededor de la posición 0 y 360 forman picos, considerando la conexión circular
            for i in range(-3, 4):  # Tres puntos antes y tres puntos después de la posición 0
                idx = i % len(signal)  # Índice circular, permite acceder a signal[-3] como signal[len(signal) - 3]
                prev_idx = (idx - 1) % len(signal)  # Índice anterior circular
                next_idx = (idx + 1) % len(signal)  # Índice siguiente circular
                
                if signal[idx] < signal[prev_idx] and signal[idx] < signal[next_idx]:
                    peaks_upper = np.insert(peaks_upper, 0, idx)  # Inserta el índice en la lista de picos si es mayor que sus vecinos
                    break

            # Eliminar picos duplicados y ordenar
            peaks_upper = np.unique(peaks_upper)

            # signal = df_upper['y'].values
            # # Revisar si el punto de la posición 0 es un pico, considerando la conexión circular con 360
            # if signal[0] < signal[1] and signal[0] < signal[-1]:
            #     peaks_upper = np.insert(peaks_upper, 0, 0)  # Inserta el índice 0 al inicio de la lista de picos
            # if signal[-1] < signal[-2] and signal[-1] < signal[0]:
            #     peaks_upper = np.append(peaks_upper, len(signal) - 1)  # Agregar el índice final como un pico
            # # Eliminar picos duplicados
            # peaks_upper = np.unique(peaks_upper)            


        # if not is_sequence_valid(peaks_upper, peaks_lower):
        #     peaks_lower = correct_sequence(peaks_upper, peaks_lower)
        #     pass

        # angles_lower = df_lower['angle'].values.tolist()
        # angles_upper = df_upper['angle'].values.tolist()

        '''
        df_upper
                angle  node_id        x        y         z
        0      0.66743     1255  3.11257  1.10617  0.036260
        1      3.39718     1261  3.10948  1.11895  0.184583
        2      5.46858     1262  3.10007  1.12219  0.296787
        3      5.64053     1682  3.09929  1.12260  0.306101
        4      7.01981     1683  3.09068  1.12370  0.380573
        ..         ...      ...      ...      ...       ...
        395  356.74700     1966  3.10771  1.12702 -0.176652
        396  357.37200     1670  3.11020  1.12455 -0.142742
        397  357.40200     1664  3.11024  1.12437 -0.141120
        398  358.88800     1258  3.11237  1.11634 -0.060438
        399  359.75400     1254  3.11360  1.11239 -0.013343
        '''

        if True:
            # get angle columns of the dataframes 
            angles_lower = df_lower['angle'].values
            angles_upper = df_upper['angle'].values

            # get the peak values
            angles_lower_peaks = angles_lower[peaks_lower]
            angles_upper_peaks = angles_upper[peaks_upper]

            # verify if the two last sequences has the same length and 5º of maximum differences between peaks 
            is_valid = is_sequence_valid(angles_upper_peaks, angles_lower_peaks)

            if not is_valid:
                if enable_print:
                    print("girdle points are revaluated", len(angles_upper_peaks), len(angles_lower_peaks))

                # girdle points are revaluated
                corrected_angles_lower_peaks = correct_sequence(angles_upper_peaks, angles_lower_peaks)
                
                lower_closet_indices = [np.argmin(np.abs(np.array(angles_lower) - target_angle)) for target_angle in corrected_angles_lower_peaks]
                lower_closet_indices = [int(index) for index in lower_closet_indices]

                upper_closet_indices = [np.argmin(np.abs(np.array(angles_upper) - target_angle)) for target_angle in angles_upper_peaks]
                upper_closet_indices = [int(index) for index in upper_closet_indices]

                peaks_lower = lower_closet_indices

        points_lower = np.column_stack((df_lower["x"], df_lower["y"], df_lower["z"]))
        points_upper = np.column_stack((df_upper["x"], df_upper["y"], df_upper["z"]))
        
        if save:
            '''
            # Data example
            point_id = np.array([1, 10, 20, 30, 40, 50, 60], dtype=np.int64)
            angle = np.array([0.425717, 0.669741, 0.83225, 1.63723, 3.82316, 4.29949, 6.88758], dtype=np.float64)
            x = np.array([3.11375, 3.11369, 3.11366, 3.11278, 3.10666, 3.10452, 3.09067], dtype=np.float64)
            y = np.array([1.39718, 1.39555, 1.39436, 1.38903, 1.37778, 1.37602, 1.3684], dtype=np.float64)
            z = np.array([0.0231361, 0.0363982, 0.0452306, 0.0889719, 0.207605, 0.233402, 0.373333], dtype=np.float64)
            '''
            
            def save_girdle_peaks(angle:np.ndarray, peak_id:int, node_id:int, x:np.ndarray, y:np.ndarray, z:np.ndarray, output_file:str):
                pass
                # Escribir la tabla en formato CSV
                with open(output_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    # Escribir la cabecera
                    writer.writerow(["angle", "peak_id", "node_id", "x", "y", "z"])
                    # Escribir los datos
                    for i in range(len(node_id)):
                        writer.writerow([angle[i], peak_id[i], node_id[i], x[i], y[i], z[i]])
                if enable_print:
                    print(f"Archivo CSV salvado como '{output_file}'")
                
            peaks_lower_filepath = os.path.join(path, "lower_girdle_peaks_" + code_id + ".csv")
            _angle = df_lower['angle'].values[peaks_lower]
            nodes_id = df_lower["node_id"].values[peaks_lower]
            _x = df_lower["x"].values[peaks_lower]
            _y = df_lower["y"].values[peaks_lower]
            _z = df_lower["z"].values[peaks_lower]
            save_girdle_peaks(_angle, peaks_lower, nodes_id, _x, _y, _z, peaks_lower_filepath)

            peaks_upper_filepath = os.path.join(path, "upper_girdle_peaks_" + code_id + ".csv")
            _angle = df_upper['angle'].values[peaks_upper]
            nodes_id = df_upper["node_id"].values[peaks_upper]
            _x = df_upper["x"].values[peaks_upper]
            _y = df_upper["y"].values[peaks_upper]
            _z = df_upper["z"].values[peaks_upper]
            save_girdle_peaks(_angle, peaks_upper, nodes_id, _x, _y, _z, peaks_upper_filepath)
                    
        if plot:
            # Visualizar los resultados
            plt.figure(figsize=(20, 10))
            # plt.plot(df.index, y, label='Datos originales', alpha=0.7)
            # plt.plot(df.index, y_smooth, label='Datos suavizados', alpha=0.7)
            # plt.plot(df.index[peaks], y[peaks], "x", label='Picos detectados', markersize=10, color='red')

            plt.plot(df_lower['angle'].values, df_lower["y"].values, label='Lower Girdlet', alpha=0.7)
            if y_smooth_lower is not None:
                plt.plot(df_lower['angle'].values, y_smooth_lower, label='Lower soft', alpha=0.7)
            plt.plot(df_lower['angle'].values[peaks_lower], df_lower["y"].values[peaks_lower], "x", label='Lower peaks', markersize=10, color='red')

            plt.plot(df_upper['angle'].values, df_upper["y"].values, label='Upper Girdlet', alpha=0.7)
            if y_smooth_upper is not None:
                plt.plot(df_upper['angle'].values, y_smooth_upper, label='Upper soft', alpha=0.7)
            plt.plot(df_upper['angle'].values[peaks_upper], df_upper["y"].values[peaks_upper], "x", label='Upper peaks', markersize=10, color='blue')

            plt.title('Diamond Girdle Analytics')
            plt.xlabel('Angle º')
            plt.ylabel('Values (mm)')
            plt.legend()
            plt.grid()
            plt.show()
            
        self.points_lower_peaks = points_lower[peaks_lower]
        self.points_upper_peaks = points_upper[peaks_upper]  
        return self.points_lower_peaks, self.points_upper_peaks          

        # Mostrar las posiciones y valores de los picos detectados
        # picos_detectados = pd.DataFrame({'Índice': peaks, 'Valores': y[peaks]})
        # print(picos_detectados)
    # end of both_true_girdle_peek_detection()

def girdle_statistics():

    def statistic(seq):
        # Calcular la secuencia diferencia entre términos consecutivos
        diff = np.diff(seq)
        # Calcular el valor medio y la desviación estándar de diff
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        return mean_diff, std_diff

    upper =     [0.539967, 22.2623, 43.9249, 67.1075, 89.4845, 111.268, 133.786, 156.567, 181.597, 204.058, 225.693, 247.933, 270.234, 292.408, 315.474, 337.26, 360.539967]
    upper_new = [0.539967, 22.2623, 43.9249, 66.9318, 89.4845, 111.268, 133.786, 156.450, 181.597, 204.058, 225.693, 248.212, 270.234, 292.408, 315.474, 337.26, 360.539967]
    lower =     [0.425717, 22.186, 44.5007, 67.6245, 90.9852, 111.699, 134.035, 155.184, 181.174, 203.542, 225.532, 247.495, 271.157, 291.495, 315.452, 338.039, 360.425717]
    lower_new = [0.425717, 22.186, 44.5007, 67.6245, 90.9852, 111.325, 132.982, 155.184, 181.149, 203.542, 225.532, 247.495, 271.227, 291.495, 315.720, 338.039, 360.425717]

    '''
    results:
    mean   std
    22.5   0.836000794481453   upper
    22.5   0.8528004746853896  upper_new
    22.5   1.3150044614510323  lower
    22.5   1.3524139376079058  lower_new
    '''
    
    all_seq = [upper, upper_new, lower, lower_new]
    for seq in all_seq:
        mean_diff, std_diff = statistic(seq)
        if enable_print:
            print(mean_diff, std_diff)

def test_from_c(code_id):
    # filename = "lower_girdle_coords_" + code_id + ".csv"
    gh = GirdleHandler()
    gh.both_true_girdle_peek_detection(code_id, save=True, plot=True, path=None)

import argparse

if __name__ == "__main__":
     
    # girdle_statistics()

    if False:    
        # Set up the argument parser
        parser = argparse.ArgumentParser(description="Pass a string argument to the main function.")
        
        # Add a string argument
        parser.add_argument("input_string", type=str, help="A string to pass to the main function")
        
        # Parse the arguments
        args = parser.parse_args()
        print(args.input_string)
        code_id = args.input_string
        
        gh = GirdleHandler()
        gh.both_true_girdle_peek_detection(code_id, save=False, plot=True, path=None)
    
    
    option = 2
    start_time = time.time()

    if option==1:  # old method
        base_name = "more_values_5"
        # excel_path = girdle_table_to_excel(base_name, option=2) 
        # draw_from_excel(base_name, option=2)

        path = r"G:\2024\1-Bhavesh\Documentación\UI"
        path = r"C:\Users\monti\AppData\Local\PixelPolish3D"
        filename = "more_values_LHPO_Round_Polished_400_0002_11p7_2024-08-03.csv"
        # peek_detection(filename, path=None)
        
    if option==2:  # true girdle 3D
        code_id = "LHPO_Round_Polished_800_0002_11p7_2024-08-03"
        code_id = "LHPO_Round_Polished_400_0002_11p7_2024-08-03"
        # code_id = "HPO_Round_Polished_400_0001_11p7_2024-08-03"   # falta de sincro en girdle peaks 0 and 1
        # code_id = "Round_SemiPolished_400_0001_11p7_2024-06-24"
        # filename = "lower_girdle_coords_" + code_id + ".csv"
        gh = GirdleHandler()
        gh.both_true_girdle_peek_detection(code_id, save=False, plot=True, path=None)
        
    if option==3:
        
        gh = GirdleHandler()
        code_id = "LHPO_Round_Polished_400_0002_11p7_2024-08-03"
        gh.load_data(code_id)

        girdle_lower_points_3d, girdle_upper_points_3d = gh.get_girdle_3d_positioning()

        # peaks_lower_girdle_top, peaks_lower_girdle_botton, peaks_upper_girdle_top, peaks_upper_girdle_botton
        lower_girdle_top_peaks, lower_girdle_botton_peaks, _, _ = gh.all_peak_detection()
        
        lower_girdle_top_3d_peaks = girdle_lower_points_3d[lower_girdle_top_peaks]
        lower_girdle_botton_3d_peaks = girdle_lower_points_3d[lower_girdle_botton_peaks]

        lapse = round(time.time() - start_time, 3)
        print("girdle peaks analysis:", lapse)

        # Add an example object (a cube) to the scene
        # cube = vtk.vtkCubeSource()
        # cube_mapper = vtk.vtkPolyDataMapper()
        # cube_mapper.SetInputConnection(cube.GetOutputPort())
        # cube_actor = vtk.vtkActor()
        # cube_actor.SetMapper(cube_mapper)

        radio = 11.72 / 2000.0
        sphere_actor = gh.create_colored_spheres(lower_girdle_botton_3d_peaks, 1, radio)
        sphere_actor = gh.create_colored_spheres(lower_girdle_top_3d_peaks, 2, radio)
        # Start rendering process
        renderer, render_window, render_window_interactor = GirdleHandler.rendering_pipeline()
        renderer.AddActor(sphere_actor)
        # renderer.AddActor(cube_actor)
        render_window.Render()  # Render the initial scene
        render_window_interactor.Start()  # Start the interaction

        pass

# "C:\Users\monti\AppData\Local\PixelPolish3D\more_values_LHPO_Round_Polished_400_0002_11p7_2024-08-03.txt"

'''
'tabla4-GOOD.txt' se obtiene haciendo parada en  
la línea 544 de "PolyhedralModeling.cpp", 
imprimiendo manualmente los valores de "more_values":

544 
	if (this->silhouettes_polygons.size() > 0)

Estos 
		// girdle and other points handling
		// index_upper_girdle, x_upper_girdle, x_lower_girdle, y_upper_girdle, y_lower_girdle, girdle_height  
		extern std::vector<std::tuple<int, float, float, float, float, float>> more_values;

'''





