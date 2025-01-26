import os
import numpy as np
import threading
import cv2
import warnings

import vtk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

using_cv2_coordinates = True

class KeyPressCallback(vtk.vtkCommand):

    def __init__(self):
        super().__init__()
        self.pxl_visualization = None

    @staticmethod
    def New():
        # Create an instance of KeyPressCallback and return it
        return KeyPressCallback(None)

    def Execute(self, caller, event):

        # Get the interactor
        interactor = caller

        # Check if the event is a KeyPressEvent
        if event == "KeyPressEvent":
        # Get the key pressed
            key = interactor.GetKeySym()
            self.pxl_visualization.key_events(key)

        # Re-render the scene
        # self.pxl_vis.render_window.Render()


def pxl_path_to_appdata():
    local_dir = os.getenv('LOCALAPPDATA')

    pxl_path = os.path.join(local_dir, "PixelPolish3D")

    if not os.path.exists(pxl_path):
        os.mkdir(pxl_path)

    return pxl_path

def loader_folder_backup(new_path = None):

    pxl_path = pxl_path_to_appdata()
    load_file = os.path.join(pxl_path, "load_folder.txt")
    line = "C:\\"

    if new_path == None:
        if os.path.exists(load_file):
            with open(load_file, 'r') as f:
                lines = f.readlines()
                # line = lines[0]
                if len(lines) > 0:
                    line = lines[0]
        else:
            with open(load_file, 'w') as f:
                f.write("C:\\")
                # line = "C:\\"        
    else:
        with open(load_file, 'w') as f:
            f.write(new_path)
            line = new_path        
    f.close()
    return line

def loader_file_backup(new_path = None):

    pxl_path = pxl_path_to_appdata()
    load_file = os.path.join(pxl_path, "load_file.txt")
    line = "C:\\"

    if new_path == None:
        if os.path.exists(load_file):
            with open(load_file, 'r') as f:
                lines = f.readlines()
                # line = lines[0]
                if len(lines) > 0:
                    line = lines[0]
        else:
            with open(load_file, 'w') as f:
                f.write("C:\\")
                # line = "C:\\"        
    else:
        with open(load_file, 'w') as f:
            f.write(new_path)
            line = new_path        
    f.close()
    return line

import cv2


# def read_bmp_image(file_path):

#     try:
        
#         # Carga la imagen usando cv2 para evaluar sus propiedades
#         # img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
#         img = cv2.imread("path_to_your_image.bmp", cv2.IMREAD_GRAYSCALE)
        
#         if img is None:
#             print("Error: Cannot load the image. Please check the file path.")
#             return None, None

#         # Obtener dimensiones
#         height, width = img.shape[:2]
        
#         # Evaluar el número de canales
#         # if len(img.shape) == 3:
#         #     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         #     num_channels = img.shape[2]  # Tres dimensiones indican una imagen con canales (BGR)
#         #     del img
#         # else:
#         #     img_gray = img
#         #     num_channels = 1  # Dos dimensiones indican una imagen en escala de grises

#         # print(f"Dimensions: {width}x{height}")
#         # print(f"Number of channels: {num_channels}")
        
#         return img
    
#     except Exception as e:
#         print(f"Error reading the image: {e}")
#         return None, None


def show_image(image, window_name="Image"):
    """
    Function to show an image in a separate thread, with the window resized to 600x400.
    :param image: Image to display (numpy array).
    :param window_name: Name of the OpenCV window.
    """
    def display():
        resized_image = cv2.resize(image, (600, 400))  # Resize the image to 600x400
        cv2.imshow(window_name, resized_image)
        # Keep the window open until 'q' is pressed
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyWindow(window_name)
                break

    # Run the display function in a separate thread
    thread = threading.Thread(target=display, daemon=True)
    thread.start()

def table_position(gray_image):
    """
    Detect the row representing the transition between the diamond and the platform
    in a grayscale image.
    
    :param gray_image: Grayscale image as a numpy.ndarray.
    :return: Index of the row with the maximum transition.
    """
    # Get the number of rows in the image
    rows = gray_image.shape[0]

    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, 50, 150)

    # show_image(edges, "Edges")
    
    # Find contours in the edges
    # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ******** Detect the top plane of the platform ********
    # Sum pixel values along each row
    sum_cols = np.sum(edges, axis=1, dtype=np.int32)

    # threshold = 2
    # for index, value in enumerate(sum_cols):
    #         if value >= threshold:
    #             break
            
    # Find the row with the maximum value
    max_row = np.argmax(sum_cols)

    # mas center of largest contour
    cx, cy = 0,0
    offset = 10
    selec_cxcy = False
    
    if True:
        # Find contours in the edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if any contours were found
        if contours:
            # Find the largest contour by the number of points
            largest_contour = max(contours, key=len)

            # Calculate the centroid (center of mass) of the largest contour
            M = cv2.moments(largest_contour)
            if M['m00'] != 0:  # Avoid division by zero
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                # print(f"Center of mass: ({cx}, {cy})")

            if not selec_cxcy:
                cx = cx + offset
                cy = max_row - offset
            
            value_1 = gray_image[cy, cx]
            # print("value at cx cy:", value_1)

            # Find the index of the first non-zero element
            top_row = next((i for i, x in enumerate(sum_cols) if x != 0), None)

            # Find the topmost point of the largest contour
            # This corresponds to the smallest y-coordinate in the contour
            # topmost_point = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
            # Row-column coordinates
            # cy, cx = topmost_point[1], topmost_point[0]

            # cy = top_row + 20
            # value_2 = gray_image[cy, cx]
            pass
    
    return int(max_row), cx, cy

def cv_to_vtk_image(cv_image, vtk_image):
    """
    Convert a grayscale OpenCV image to a VTK image.

    :param cv_image: Grayscale image (numpy.ndarray) read using cv2.
    :return: vtkImageData object.
    """
    # Get the dimensions of the OpenCV image
    height, width = cv_image.shape

    # Create a vtkImageData object
    # vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(width, height, 1)
    vtk_image.SetSpacing(1.0, 1.0, 1.0)
    vtk_image.SetOrigin(0.0, 0.0, 0.0)
    vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)  # 1 channel (grayscale)

    # Convert the OpenCV image (numpy array) to a VTK array
    flat_array = cv_image.flatten()  # Flatten the numpy array
    vtk_array = numpy_to_vtk(flat_array, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    vtk_image.GetPointData().SetScalars(vtk_array) 
    return vtk_image

def cv2_processing(gray_image, y_table, max_val):

    # Clone the input image to modify it
    # img = gray_image.copy()
    img = gray_image

    # Verify that the image is valid
    if img is None or img.size == 0:
        return

    # Calculate y_table if not already defined
    if y_table[0] < 0:
        y_table[0], y_table[1], y_table[2] = table_position(img)
        # y_table[2] = y_table[2] - 50  # PENDIENTE
        if y_table[0] < img.shape[0] / 2:
            y_table[0] = img.shape[0] - 2
        if y_table[2] >= y_table[0]:
            y_table[2] = y_table[0] - 50 
        # print(f"y_table: {y_table[0]}   cx: {y_table[1]}   cy: {y_table[2]}")

    # Calculate max_val if not already defined
    if max_val[0] < 0:
        max_val[0] = np.max(img)

    # Convert max_val to a valid scalar for OpenCV
    max_color = int(max_val[0])  # Ensure max_val is an integer for grayscale

    # Define points for clearing regions
    pt1 = [0, min(y_table[0], img.shape[0] - 2)]  # Avoid non-table condition
    pt2 = [img.shape[1] - 1, img.shape[0] - 1]    # [cols-1, rows-1]

    # Clear lower region of table
    cv2.rectangle(img, tuple(pt1), tuple(pt2), max_color, thickness=cv2.FILLED)

	# Clear first row that it is necessary for cylinders 
    pt1 = [0, 0]
    pt2 = [img.shape[1] - 1, 1]
    cv2.rectangle(img, tuple(pt1), tuple(pt2), max_color, thickness=cv2.FILLED)
    return img

# it is not more used
def processing_and_convert_to_vtk_image(gray_image_cut, vtk_image, y_table, max_val):
    """
    Process an input grayscale image and convert it to a VTK image.
    
    The function updates `vtk_image`, `y_table`, and `max_val` as output arguments.

    :param gray_image: Input grayscale image (numpy.ndarray).
    :param vtk_image: vtkImageData object to store the output (output argument).
    :param y_table: Output argument to store the table position (int, reference-like behavior).
    :param max_val: Output argument to store the maximum pixel value (float, reference-like behavior).
    
    Example of use:
    gray_image = cv2.imread("path_to_image.bmp", cv2.IMREAD_GRAYSCALE)
    vtk_image = vtk.vtkImageData()

    # Initialize output variables
    y_table = [-1] # Simulate an output parameter
    max_val = [-1] # Simulate an output parameter

    # Processing the image and converting it to VTK format
    processing_and_convert_to_vtk_image(gray_image, vtk_image, y_table, max_val)    
    """
    
    if False:    
        # Clone the input image to modify it
        # img = gray_image.copy()
        img = gray_image

        # Verify that the image is valid
        if img is None or img.size == 0:
            return

        # Calculate y_table if not already defined
        if y_table[0] < 0:
            y_table[0], y_table[1], y_table[2] = table_position(img)
            # y_table[2] = y_table[2] - 50  # PENDIENTE
            if y_table[0] < img.shape[0] / 2:
                y_table[0] = img.shape[0] - 2
            if y_table[2] >= y_table[0]:
                y_table[2] = y_table[0] - 50 
            # print(f"y_table: {y_table[0]}   cx: {y_table[1]}   cy: {y_table[2]}")

        # Calculate max_val if not already defined
        if max_val[0] < 0:
            max_val[0] = np.max(img)

        # Convert max_val to a valid scalar for OpenCV
        max_color = int(max_val[0])  # Ensure max_val is an integer for grayscale

        # Define points for clearing regions
        pt1 = [0, min(y_table[0], img.shape[0] - 2)]  # Avoid non-table condition
        pt2 = [img.shape[1] - 1, img.shape[0] - 1]    # [cols-1, rows-1]

        # Clear lower region of table
        cv2.rectangle(img, tuple(pt1), tuple(pt2), max_color, thickness=cv2.FILLED)

        # Clear first row that it is necessary for cylinders 
        pt1 = [0, 0]
        pt2 = [img.shape[1] - 1, 1]
        cv2.rectangle(img, tuple(pt1), tuple(pt2), max_color, thickness=cv2.FILLED)

    # # img = cv2_processing(gray_image, y_table, max_val)
    # img = gray_image_cut

    # # Vertical flip is necessary from OpenCV to VTK
    # img2 = img.copy()
    # img = cv2.flip(img, 0)

    # # Convert the processed image to VTK format
    # cv_to_vtk_image(img, vtk_image)
    
    # if using_cv2_coordinates:
    #     # img = None
    #     return img2

    # img2 = None
    # return img

    # Vertical flip is necessary from OpenCV to VTK
    img2 = gray_image_cut.copy()
    img2 = cv2.flip(img2, 0)

    # Convert the processed image to VTK format
    cv_to_vtk_image(img2, vtk_image)

def add_text(renderer, label, scale_text, vector_text=None, text_actor=None):
    """
    Add text to a VTK renderer with specified scale and properties.

    :param renderer: vtkRenderer instance where the text will be added.
    :param label: The text to display.
    :param scale_text: Scale of the text for better visibility.
    :param vector_text: vtkVectorText instance to define the text (optional, creates a new one if None).
    :param text_actor: vtkFollower instance to represent the text actor (optional, creates a new one if None).
    :return: Updated vector_text and text_actor.
    """
    
    if text_actor is None:
        text_actor = vtk.vtkFollower()
    
        # Initialize vtkVectorText if not provided
        if vector_text is None:
            vector_text = vtk.vtkVectorText()

        # Create a mapper for the text
        text_mapper = vtk.vtkPolyDataMapper()
        text_mapper.SetInputConnection(vector_text.GetOutputPort())

        # Initialize vtkFollower if not provided
        text_actor.SetMapper(text_mapper)
        text_actor.GetProperty().SetColor(1.0, 1.0, 0.0)  # Yellow color for the text

        # Associate the text actor with the renderer's active camera
        text_actor.SetCamera(renderer.GetActiveCamera())

        # Set position (example position; adjust as needed)
        text_actor.SetPosition(10, 10, 0)
        text_actor.VisibilityOn()

        # Add the text actor to the renderer
        renderer.AddActor(text_actor)

    vector_text.SetText(label)
    text_actor.SetScale(scale_text)  # Scale the text for better visibility

    return vector_text, text_actor

import numpy as np

def detect_contour_with_marching_squares(image, start_point, threshold):
    """
    Detecta el contorno de un objeto en una imagen en escala de grises utilizando Marching Squares.

    :param image: np.ndarray, imagen en escala de grises (2D).
    :param start_point: Tuple[int, int], punto inicial dentro del objeto.
    :param threshold: float, umbral para determinar el borde entre el objeto y el fondo.
    :return: List[Tuple[float, float]], lista de puntos subpixel del contorno.
    """
    # Direcciones para explorar vecinos (en sentido horario)
    directions = [
        (0, 1),  # Derecha
        (1, 0),  # Abajo
        (0, -1),  # Izquierda
        (-1, 0)  # Arriba
    ]

    def interpolate(p1, p2, v1, v2):
        """Interpolación lineal para encontrar puntos subpixel."""
        if abs(v2 - v1) < 1e-6:  # Evitar divisiones por cero
            return p1
        return p1 + (p2 - p1) * (threshold - v1) / (v2 - v1)

    # Inicializar variables
    x, y = start_point
    visited = set()  # Mantener seguimiento de celdas visitadas
    contour = []  # Almacenar puntos del contorno

    # Encontrar el borde inicial desplazándose hacia el borde
    while image[y, x] < threshold:
        y -= 1
    y += 1

    # Iniciar la detección del contorno
    current_direction = 0
    while True:
        # Evaluar los cuatro esquinas de la celda actual
        corners = [
            image[y, x],       # Esquina superior izquierda
            image[y, x + 1],   # Esquina superior derecha
            image[y + 1, x + 1],  # Esquina inferior derecha
            image[y + 1, x]    # Esquina inferior izquierda
        ]
        square_index = sum(1 << i for i, value in enumerate(corners) if value > threshold)

        # Evitar bucles infinitos
        if (x, y) in visited:
            break
        visited.add((x, y))

        # Determinar los bordes activos del contorno
        if square_index in {1, 14}:  # Borde entre 0 y 3
            px = interpolate(x, x + 1, corners[0], corners[1])
            py = interpolate(y, y + 1, corners[0], corners[3])
        elif square_index in {2, 13}:  # Borde entre 1 y 2
            px = interpolate(x, x + 1, corners[1], corners[2])
            py = interpolate(y, y + 1, corners[0], corners[1])
        elif square_index in {4, 11}:  # Borde entre 2 y 3
            px = interpolate(x, x + 1, corners[3], corners[2])
            py = interpolate(y, y + 1, corners[2], corners[3])
        elif square_index in {8, 7}:  # Borde entre 0 y 1
            px = interpolate(x, x + 1, corners[0], corners[3])
            py = interpolate(y, y + 1, corners[1], corners[0])
        else:
            break  # Caso no manejado o celda no activa

        # Agregar punto interpolado al contorno
        contour.append((px, py))

        # Moverse a la siguiente celda en la dirección actual
        for i in range(4):
            nx, ny = x + directions[(current_direction + i) % 4][0], y + directions[(current_direction + i) % 4][1]
            if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0] and (nx, ny) not in visited:
                x, y = nx, ny
                current_direction = (current_direction + i) % 4
                break
        else:
            break  # Contorno completo o sin salida

        # Parar si volvemos al punto inicial
        if len(contour) > 1 and np.allclose(contour[0], contour[-1], atol=1e-3):
            break

    return contour


def create_polydata_from_contour(contour_points):
    """
    Transforms a list of points into vtkPolyData with a single polyline cell.
    
    :param contour_points: List[Tuple[float, float]], list of contour points.
    :return: vtkPolyData, the contour represented as a polyline.

    # Example usage:
    contour_points = [(0, 0), (1, 0), (1, 1), (0, 1)]  # Example list of contour points
    polydata = create_polydata_from_contour(contour_points)

    # Verify the result (optional)
    print(f"Number of points: {polydata.GetNumberOfPoints()}")
    print(f"Number of cells: {polydata.GetNumberOfCells()}")
    """
    
    # Create vtkPoints to store the contour points
    points = vtk.vtkPoints()
    for point in contour_points:
        points.InsertNextPoint(point[0], point[1], 0)  # Z = 0 for 2D geometry

    # Create a vtkPolyLine to represent the polyline
    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(len(contour_points))  # Number of points
    for i, _ in enumerate(contour_points):
        polyline.GetPointIds().SetId(i, i)  # Assign point IDs sequentially

    # Create a vtkCellArray to store the polyline cell
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(polyline)

    # Create vtkPolyData and set points and cells
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)  # Assign points to polydata
    polydata.SetLines(cells)   # Assign the polyline as a cell

    return polydata


def detect_contour_subpixel_ORG(image:np.ndarray, start_point, threshold):
    """
    Detecta el contorno de un objeto en escala de grises con precisión subpixel 
    usando un enfoque basado en Marching Squares.

    :param image: np.ndarray, imagen en escala de grises.
    :param start_point: Tuple[int, int], punto inicial dentro del objeto.
    :param threshold: float, umbral para definir el borde.
    :return: List[Tuple[float, float]], puntos del contorno en coordenadas subpixel.
    """
    def interpolate(p1, p2, v1, v2):
        
        # if np.isinf(v1) or np.isinf(v2):
        #     raise ValueError("v1 or v2 has an infinite value.")

        # if np.isnan(v1) or np.isnan(v2):
        #     raise ValueError("v1 or v2 has a NaN value.")
        
        """Interpolación lineal para encontrar puntos subpixel."""
        diff = v2 - v1

        # if np.isinf(diff):
        #     raise ValueError("v1 or v2 has an infinite value.")

        # if np.isnan(diff):
        #     raise ValueError("v1 or v2 has a NaN value.")

        if abs(diff) < 1e-6:  # Evitar divisiones por cero
            print("Here there are overflow.")
            return p1
        return p1 + (p2 - p1) * (threshold - v1) / diff

    # Dimensiones de la imagen
    height, width = image.shape

    # Posición inicial
    x, y = start_point

    # Verificar que el punto inicial está dentro de los límites de la imagen
    if not (0 <= x < width and 0 <= y < height):
        return None

    # Direcciones para explorar vecinos en sentido horario
    directions = [
        (0, 1),  # Derecha
        (1, 0),  # Abajo
        (0, -1), # Izquierda
        (-1, 0)  # Arriba
    ]
    
    right, down, left, up = 0, 1, 2, 3
    
    s = -1
    # NOTE: use 1 for images in cv2 coordinates
    if using_cv2_coordinates:
        s = 1

    # Encontrar el borde inicial moviéndose hacia arriba hasta cruzar el umbral
    while image[y, x] < threshold:
        y += s
    y -= s

    # Iniciar el algoritmo Marching Squares
    current_direction = left 
    # NOTE: use right for images in cv2 coordinates
    if using_cv2_coordinates:
        current_direction = right 

    new_direction = None

    # start point (first square)   
    xo = x
    yo = y

    # first contour point
    px = x
    py = interpolate(yo, yo + 1, image[yo, xo], image[yo + 1, xo]) 

    # Puntos del contorno
    contour_points = []
    
    h = image.shape[0]
    contour_points.append((px, py))

    while True:
        # Valores de las esquinas de la celda actual
        corners = [
            image[y, x],       # Esquina superior izquierda
            image[y, x + 1],   # Esquina superior derecha
            image[y + 1, x + 1], # Esquina inferior derecha
            image[y + 1, x]    # Esquina inferior izquierda
        ]
        
        # Generar un índice para determinar los bordes activos
        square_index = sum(1 << i for i, value in enumerate(corners) if value > threshold)

        # Detectar los bordes cruzados y calcular puntos subpixel

        if square_index in {1, 14}:  # Esquina superior izquierda activa

            # 		        new direction
            # 1 0   1 / 14	l, u 
            # 0 0
                        
            if current_direction == right:
                new_direction = up
                # px = interpolate(x, x + 1, corners[0], corners[1])
                px = interpolate(x, x + 1, image[y, x], image[y, x+1])
                py = y
            elif current_direction == down:
                new_direction = left
                px = x
                # py = interpolate(y, y + 1, corners[0], corners[3])
                py = interpolate(y, y + 1, image[y, x], image[y+1, x])
        
        elif square_index in {2, 13}:  # Esquina superior derecha activa

            # 		        new direction
            # 0 1   2 / 13	r, u
            # 0 0

            if current_direction == left:
                new_direction = up
                # px = interpolate(x, x + 1, corners[1], corners[2])
                px = interpolate(x, x + 1, image[y, x], image[y, x+1])
                py = y
            elif current_direction == down:
                new_direction = right
                px = x + 1
                # py = interpolate(y, y + 1, corners[0], corners[1])
                py = interpolate(y, y + 1, image[y, x+1], image[y+1, x+1])

        elif square_index in {4, 11}:  # Esquina inferior derecha activa

            # 		        new direction
            # 0 0   4 / 11	r, d
            # 0 1

            if current_direction == left:
                new_direction = down
                px = interpolate(x, x + 1, image[y+1, x], image[y+1, x+1])
                # px = interpolate(x, x + 1, corners[3], corners[2])
                py = y + 1
            elif current_direction == up:
                new_direction = right
                px = x + 1
                # py = interpolate(y, y + 1, corners[2], corners[3])
                py = interpolate(y, y + 1, image[y, x+1], image[y+1, x+1])

        elif square_index in {8, 7}:  # Esquina inferior izquierda activa

            # 		        new direction
            # 1 1   7 / 8	l, d
            # 0 1
        
            if current_direction == right:
                new_direction = down
                # px = interpolate(x, x + 1, corners[0], corners[3])
                px = interpolate(x, x + 1, image[y+1, x], image[y+1, x+1])
                py = y + 1
            elif current_direction == up:
                new_direction = left
                px = x
                # py = interpolate(y, y + 1, corners[1], corners[0])
                py = interpolate(y, y + 1, image[y, x], image[y+1, x])



        elif square_index in {3, 12}:  # Esquinas superiores activas

            # 		        new direction
            # 1 1   3 / 12	l, r .  se mantiene la dirección previa
            # 0 0

            if current_direction == right:
                px = x+1
                # px = interpolate(x, x + 1, corners[0], corners[3])
                py = interpolate(y, y + 1, image[y, x+1], image[y+1, x+1])
            if current_direction == left:
                px = x
                # py = interpolate(y, y + 1, corners[1], corners[0])
                py = interpolate(y, y + 1, image[y, x], image[y+1, x])
            new_direction = current_direction

        elif square_index in {6, 9}:  # Esquina de la derecha activas
            
            # 		        new direction
            # 0 1   6 / 9	u, d .  se mantiene la dirección previa
            # 0 1

            if current_direction == up:
                # px = interpolate(x, x + 1, corners[0], corners[3])
                px = interpolate(x, x + 1, image[y, x], image[y, x+1])
                py = y
            if current_direction == down:
                # py = interpolate(y, y + 1, corners[1], corners[0])
                px = interpolate(x, x + 1, image[y+1, x], image[y+1, x+1])
                py = y+1
            new_direction = current_direction

        else:
            # 		        new direction
            # 1 0   5 / 10	----  ----
            # 0 1
            # No hay bordes activos o caso no manejado
            new_direction = current_direction
            # print("I'm here.")
            continue

        # Agregar punto interpolado al contorno
        # if using_cv2_coordinates:
        #     py = h - 1 - py
        contour_points.append((px, py))

        dy, dx = directions[new_direction]
        x, y = x + dx, y + dy
        current_direction = new_direction
    
        if x==xo and y==yo:
            break
    return contour_points



def interpolate_with_warning(p1, p2, v1, v2, threshold):
    """
    Interpolates a position along a line between two points based on intensity values.
    Captures warnings like overflow and handles them gracefully.

    Parameters:
        p1 (int): Pixel coordinate 1.
        p2 (int): Pixel coordinate 2.
        v1 (int): Intensity value at p1.
        v2 (int): Intensity value at p2.
        threshold (float): Threshold intensity for interpolation.        
    Returns:
        float: Interpolated position, or midpoint if v1 == v2 or warning occurs.
    """
    
    p1=int(p1)
    p2=int(p2)
    v1=int(v1)
    v2=int(v2)
    
    # Ensure inputs are integers for p1, p2, v1, and v2
    # if not all(isinstance(x, int) for x in [p1, p2, v1, v2]):
    #     raise ValueError("p1, p2, v1, and v2 must be integers.")
    # if not isinstance(threshold, (int, float)):
    #     raise ValueError("threshold must be a number.")

    # Use a warnings filter to catch overflow warnings
    with warnings.catch_warnings(record=True) as _w:
        warnings.simplefilter("always")  # Convert warnings into manageable objects

        # Compute the difference in intensity
        diff = v2 - v1

        # Avoid division by zero if v1 == v2
        if diff == 0:
            print("Warning: Avoided division by zero, returning midpoint.")
            return (p1 + p2) / 2

        # Perform interpolation
        result = p1 + ((threshold - v1) / diff) * (p2 - p1)

        # Check for any warnings raised during the computation
        if _w:
            for warning in _w:
                print(f"Warning detected: {warning.message}")
            return (p1 + p2) / 2  # Return midpoint as a fallback

        return result

import random

def detect_contour_subpixel(image:np.ndarray, start_point, threshold, invert=True, image_number=-1):
    """
    Detecta el contorno de un objeto en escala de grises con precisión subpixel usando un enfoque basado en Marching Squares.

    :param image: np.ndarray, imagen en escala de grises.
    :param start_point: Tuple[int, int], punto inicial dentro del objeto.
    :param threshold: float, umbral para definir el borde.
    :return: List[Tuple[float, float]], puntos del contorno en coordenadas subpixel.
    """
    # show_image(image)

    def interpolate(p1, p2, v1, v2, threshold):
        p1=int(p1)
        p2=int(p2)
        v1=int(v1)
        v2=int(v2)
        diff = float(v2 - v1)
        if abs(diff) < 1e-6:  # Evitar divisiones por cero
            # print("Here v1 equals v2.")
            # print("v1,v2,p1,p2", v1, v2, p1, p2)
            return (p1+p2)/2
        return float(p1 +  (float(threshold - v1) / diff) * (p2 - p1))


    # Dimensiones de la imagen
    height, width = image.shape

    # Posición inicial
    x, y = start_point

    # Verificar que el punto inicial está dentro de los límites de la imagen
    if not (0 <= x < width and 0 <= y < height):
        return None

    s = 1
    # Encontrar el borde inicial moviéndose hacia abajo hasta cruzar el umbral
    while image[y, x] < threshold:
        y += s
    y -= s

    # start point (first square)   
    xo = x
    yo = y

    # first contour point
    px = xo
    py = interpolate(yo, yo + 1, image[yo, xo], image[yo + 1, xo], threshold) 
    
    change_to_left = False

    if change_to_left:
        xo -= 1
        x = xo
    
    # Puntos del contorno
    if invert:
        contour_points = [(px, height-1-py)]
    else:
        contour_points = [(px, py)]
    
    links = [[0,1], [1,2], [2,3], [3,0]]
    link_number = 3 # go right
    
    if change_to_left:
        link_number = 1

    dxdy = [[0,-1], [1,0], [0,1], [-1,0]]

    echo = image_number == 91
    indices_ = [0, 1, 2, 3]
    first_link_number = -1
    
    while True:
        if x < 0 or y < 0:
            break
        if x >= image.shape[1]-2 or y >= image.shape[0]-2:
            break
        
        # Valores de las esquinas de la celda actual
        vert_value = [
            image[y, x],       # Esquina superior izquierda
            image[y, x + 1],   # Esquina superior derecha
            image[y + 1, x + 1], # Esquina inferior derecha
            image[y + 1, x]    # Esquina inferior izquierda
        ]
        
        vert_activity = [v >= threshold for v in vert_value]  
        _vert_activity = [1 if v >= threshold else 0 for v in vert_value]  
        _vert_activity_1 = [0, 1, 0, 1]
        _vert_activity_2 = [1, 0, 1, 0]

        def link_activity(j):
            link = links[j]
            # XOR operation
            return vert_activity[link[0]] ^ vert_activity[link[1]]

        def select_random_other_index(indices, current_index):
            # Crea una nueva lista sin el índice actual
            other_indices = [i for i in indices if i != current_index]
            # Selecciona un índice aleatorio de la lista restante
            return random.choice(other_indices)
        
        def select_random_excluding(indices, exclude_indices):
            # Crea una nueva lista excluyendo los índices especificados
            other_indices = [i for i in indices if i not in exclude_indices]
            # Selecciona un índice aleatorio de la lista restante
            return random.choice(other_indices)        

        # for i_link, link in enumerate(links):
        #     if i_link == link_number:
        #         continue 
        #     if link_activity(i_link):                
        #         break

        loop_condition = _vert_activity == _vert_activity_1 or _vert_activity == _vert_activity_2
        
        if loop_condition:
            return None
            if first_link_number == -1:
                first_link_number = link_number
            i_link = select_random_excluding(indices_, [link_number, first_link_number])
            print("echo:", link_number, first_link_number, "   ", _vert_activity, i_link, "   ", x, y)
        else:
            first_link_number = -1
            for i_link, link in enumerate(links):
                if i_link == link_number:
                    continue 
                if link_activity(i_link):                
                    break
        
        if False:
            if sum(_vert_activity) != 2:                                 
                for i_link, link in enumerate(links):
                    if i_link == link_number:
                        continue 
                    if link_activity(i_link):                
                        break
            else:
                indices_ = [0, 1, 2, 3]
                i_link = select_random_other_index(indices_, link_number)
                print("echo:", link_number, "   ", _vert_activity, i_link, "   ", x, y)
            
        # if echo:
        #     print("echo:", link_number, "   ", _vert_activity, i_link, "   ", x, y)
    
        # definición de los limks
        # links = [[0,1], [1,2], [2,3], [3,0]]

        if i_link == 0:
            px = interpolate(x, x + 1, vert_value[0], vert_value[1], threshold)
            py = y
            link_number = 2 # go up
        elif i_link == 1:            
            px = x+1
            py = interpolate(y, y + 1, vert_value[1], vert_value[2], threshold)
            link_number = 3 # go right
        elif i_link == 2:
            px = interpolate(x + 1, x, vert_value[2], vert_value[3], threshold)
            # px = interpolate(x, x+1, vert_value[2], vert_value[3])
            py = y+1
            link_number = 0 # go down
        elif i_link == 3:
            px = x
            py = interpolate(y + 1, y, vert_value[3], vert_value[0], threshold)
            link_number = 1 # go left
            
        if invert:
            contour_points.append((px, height-1-py))
        else:
            contour_points.append((px, py))

        dx, dy = dxdy[i_link]
        x, y = x + dx, y + dy
        
        if x==xo and y==yo:
            break
        
    return contour_points



def test_input_output_arrays():
    
    import os
    import ctypes
    import numpy as np

    # from pxl_tools import pxl_path_to_appdata
    # app_data_path = pxl_path_to_appdata()

    # Cargar la DLL
    dll_path = r"C:\Aplicaciones\Diamantes\Test2\PxlSense\x64\Release\PixelPolish3DPyV02.dll"
    native_processor = ctypes.CDLL(dll_path)


    '''
        this dll method is used:
    	PIXELPOLISH3DPYV02_API const double* test_input_output_arrays(
            const double* decimated_points, 
            size_t length_decimated, 
            size_t* length_cleaned, 
            size_t images_qtty, 
            size_t imageNumber);
    '''
    
    # *** Configurate the prototipe funtion ***

    native_processor.test_input_output_arrays.argtypes = [ 
        ctypes.POINTER(ctypes.c_double),    # pointer to c_double (double array)
        ctypes.c_size_t,                    # input array size (size_t)
        ctypes.POINTER(ctypes.c_size_t),    # output array size as pointer to c_size_t
        ctypes.c_size_t,                    # image number (size_t)
        ctypes.c_size_t                     # image qtty (size_t)
    ]

    native_processor.test_input_output_arrays.restype = ctypes.POINTER(ctypes.c_double)

    
    decimated_point_array: np.ndarray = np.array([100.0, 200.0, 300.0])
    decimated_lenght = decimated_point_array.shape[0]
    
    array_type = ctypes.c_double * decimated_lenght  # Crear el tipo array
    c_array = array_type(*decimated_point_array)  # Convertir la lista a array
    
    cleaned_length = ctypes.c_size_t()
    double_ptr = native_processor.test_input_output_arrays( \
                    c_array, \
                    decimated_lenght, \
                    ctypes.byref(cleaned_length), 1, 400)
    
    cleaned_array = [double_ptr[i] for i in range(cleaned_length.value)]
    pass



def get_pp3d_data_path() -> str:
    """
    Get the full path to the 'PixelPolish3D' folder inside the user's AppData/Local directory.

    :return: Full path to the 'PixelPolish3D' folder.
    """
    # Get the path to the AppData\Local folder
    appdata_local_path = os.getenv('LOCALAPPDATA')  # Windows-specific environment variable
    
    if not appdata_local_path:
        raise EnvironmentError("The LOCALAPPDATA environment variable is not set.")
    
    # Append the PixelPolish3D folder to the AppData\Local path
    pixelpolish3d_folder = os.path.join(appdata_local_path, 'PixelPolish3D')
    
    return pixelpolish3d_folder


def create_vtk_polydata(points_array, cells):
    """
    Create a vtkPolyData object from points and cells.

    Parameters:
    points_array (list of numpy.ndarray): List of 3D points as numpy arrays.
    cells (list of list of int): List of cells where each cell is a list of indices 
                                 corresponding to the points that form the cell.

    Returns:
    vtk.vtkPolyData: The resulting vtkPolyData object.
    """
    # Create a vtkPoints object to store point coordinates
    vtk_points = vtk.vtkPoints()
    for point in points_array:
        vtk_points.InsertNextPoint(point[0], point[1], point[2])

    ij = 0
    # Create a vtkCellArray to store the cells
    vtk_cells = vtk.vtkCellArray()
    for cell in cells:
        len_ij = len(cell) 
        vtk_cells.InsertNextCell(len(cell))
        ij += 1
        for index in cell:
            vtk_cells.InsertCellPoint(index)

    # Create a vtkPolyData object and set its points and cells
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(vtk_points)
    poly_data.SetPolys(vtk_cells)

    return poly_data

# Example usage
def test_create_vtk_polydata():

    points_array = [np.array([0.0, 0.0, 0.0]), 
                    np.array([1.0, 0.0, 0.0]), 
                    np.array([1.0, 1.0, 0.0]), 
                    np.array([0.0, 1.0, 0.0])]
                    
    cells = [[0, 1, 2, 3]]  # Single quadrilateral cell

    poly_data = create_vtk_polydata(points_array, cells)

    # Optional: Write to file for inspection
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName("output.vtp")
    writer.SetInputData(poly_data)
    writer.Write()


def transform_polydata(polydata, x0, y0, z0, y_rotation_angle):
    """
    Translates and rotates a vtkPolyData.
    
    Parameters:
    polydata (vtk.vtkPolyData): The input vtkPolyData to be transformed.
    x0, y0, z0 (float): The translation values for x, y, and z axes.
    rotation_angle (float): The rotation angle in degrees around the Y axis.
    
    Returns:
    vtk.vtkPolyData: The transformed vtkPolyData.
    """
    # Create a transform object
    transform = vtk.vtkTransform()
    
    # Step 1: Translate the polydata
    transform.Translate(x0, y0, z0)
    
    # Step 2: Rotate the polydata around the Y axis
    transform.RotateY(y_rotation_angle)  # Rotation in degrees
    
    # Create a filter to apply the transformation
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(polydata)
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    
    # Return the transformed polydata
    return transform_filter.GetOutput()


def rotate_around_y(point_coordinates, alpha):
    """
    Rotate a list of 3D points around the Y axis.

    Parameters:
    point_coordinates (list of np.ndarray): List of 3D points as numpy arrays.
    alpha (float): Angle in degrees to rotate around the Y axis.
    """
    factor = np.pi / 180.0
    cos_alpha = np.cos(factor * alpha)
    sin_alpha = np.sin(factor * alpha)
    
    for point in point_coordinates:
        x = point[0]
        z = point[2]
        point[0] = x * cos_alpha + z * sin_alpha
        point[2] = -x * sin_alpha + z * cos_alpha

def move_xyz(point_coordinates, x=0.0, y=0.0, z=0.0):
    """
    Move a list of 3D points by (x, y, z) displacement.

    Parameters:
    point_coordinates (list of np.ndarray): List of 3D points as numpy arrays.
    x (float): Displacement along the X axis.
    y (float): Displacement along the Y axis.
    z (float): Displacement along the Z axis.
    """
    for point in point_coordinates:
        point[0] += x
        point[1] += y
        point[2] += z

def translate_points(points, translation_vector):
    """
    Apply a translation to all points in the array.
    
    Args:
        points (np.ndarray): Array of points with shape (N, 3).
        translation_vector (tuple): Translation vector (xp, yp, zp).
    
    Returns:
        np.ndarray: Translated points with shape (N, 3).
    """
    xp, yp, zp = translation_vector
    translated_points = points + np.array([xp, yp, zp])
    return translated_points


def rotate_points_around_y(points, theta):
    """
    Rotate points around the Y-axis by a given angle theta (in radians).
    
    Args:
        points (np.ndarray): Array of points with shape (N, 3).
        theta (float): Angle in degrees.
    
    Returns:
        np.ndarray: Rotated points with shape (N, 3).
    """
    theta = np.radians(theta)  # to radians

    # Create the rotation matrix around the Y-axis
    rotation_matrix = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    
    # Apply the rotation to each point
    rotated_points = np.dot(points, rotation_matrix.T)  # Transpose to align with (x, y, z)
    return rotated_points

def translate_rotate_example():
    # Ejemplo de datos
    np.random.seed(42)  # Para reproducibilidad
    lower_points = np.random.rand(10, 3)  # 10 puntos con shape (N, 3)

    # Parámetros de traslación y rotación
    translation_vector = (10, 20, 30)  # Traslación (xp, yp, zp)
    theta_degrees = 45  # Ángulo de rotación en grados
    # theta_radians = np.radians(theta_degrees)  # Convertir a radianes

    # Paso 1: Aplicar la traslación
    translated_points = translate_points(lower_points, translation_vector)

    # Paso 2: Aplicar la rotación en torno al eje Y
    rotated_points = rotate_points_around_y(translated_points, theta_radians)

    # Mostrar los resultados
    # print("Original points (first 5):\n", lower_points[:5])
    # print("\nTranslated points (first 5):\n", translated_points[:5])
    # print("\nRotated points (first 5):\n", rotated_points[:5])



def peak_detector(y, positive=True):

    y_smooth = None
    if positive:
        # Suavizar los datos para reducir ruido
        # y_smooth = gaussian_filter1d(y, sigma=2)

        # Detectar picos con scipy
        # Ajustar 'distance' según la cantidad esperada de picos (16 en este caso)
        peaks, _ = find_peaks(y, distance=10, prominence=0.005)
        # peaks, _ = find_peaks(y_smooth, distance=10, prominence=0.005)
        # peaks, _ = find_peaks(y_smooth, distance=len(y)//16, prominence=0.01)

        # Si no se detectan exactamente 16 picos, ajustamos
        if len(peaks) != 16:
            peaks, _ = find_peaks(y, distance=5, prominence=0.002)
            # peaks, _ = find_peaks(y_smooth, distance=5, prominence=0.002)
    else:
        # Trabajar con el negativo de 'y'
        y_neg = -y

        # Suavizar los datos para reducir ruido (trabajando con el negativo)
        # y_neg_smooth = gaussian_filter1d(y_neg, sigma=2)

        # Detectar picos en el negativo de 'y'
        peaks, _ = find_peaks(y_neg, distance=10, prominence=0.005)
        # peaks, _ = find_peaks(y_neg_smooth, distance=10, prominence=0.005)

        # Si no se detectan exactamente 16 picos, ajustamos
        if len(peaks) != 16:
            peaks, _ = find_peaks(y_neg, distance=5, prominence=0.002)
            # peaks, _ = find_peaks(y_neg_smooth, distance=5, prominence=0.002)

        # Regresar las variables afectadas al estado original
        # y_smooth = -y_neg_smooth  # Invertir el suavizado        
    
    return peaks, y_smooth



def create_coordinate_axes(length):
    """
    Creates a VTK actor representing coordinate axes with X (red), Y (green), and Z (blue) lines.

    Parameters:
    length (float): The length of each axis.

    Returns:
    vtk.vtkActor: The actor containing the coordinate axes.
    """
    # Create points for the axes
    points = vtk.vtkPoints()
    points.InsertNextPoint(0, 0, 0)  # Origin
    points.InsertNextPoint(length, 0, 0)  # X-axis end
    points.InsertNextPoint(0, length, 0)  # Y-axis end
    points.InsertNextPoint(0, 0, length)  # Z-axis end

    # Create the lines for the axes
    lines = vtk.vtkCellArray()

    # X-axis line
    line_x = vtk.vtkLine()
    line_x.GetPointIds().SetId(0, 0)  # Origin
    line_x.GetPointIds().SetId(1, 1)  # X-axis end
    lines.InsertNextCell(line_x)

    # Y-axis line
    line_y = vtk.vtkLine()
    line_y.GetPointIds().SetId(0, 0)  # Origin
    line_y.GetPointIds().SetId(1, 2)  # Y-axis end
    lines.InsertNextCell(line_y)

    # Z-axis line
    line_z = vtk.vtkLine()
    line_z.GetPointIds().SetId(0, 0)  # Origin
    line_z.GetPointIds().SetId(1, 3)  # Z-axis end
    lines.InsertNextCell(line_z)

    # Create a polydata to store the points and lines
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.SetLines(lines)

    # Create an array to store line indices (0 for X, 1 for Y, 2 for Z)
    line_indices = vtk.vtkIntArray()
    line_indices.SetName("LineIndex")
    line_indices.SetNumberOfComponents(1)
    line_indices.SetNumberOfTuples(3)  # We have 3 lines (X, Y, Z)
    line_indices.InsertTuple1(0, 0)  # Index for X-axis
    line_indices.InsertTuple1(1, 1)  # Index for Y-axis
    line_indices.InsertTuple1(2, 2)  # Index for Z-axis

    # Attach the array to the cell data
    polyData.GetCellData().SetScalars(line_indices)

    # Create a mapper for the polydata
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polyData)
    mapper.SetScalarModeToUseCellData()
    mapper.SetColorModeToMapScalars()
    mapper.SetScalarRange(0, 2)  # Match range to the LUT range

    # Set up the color lookup table (LUT) for the three axes
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(3)  # Only 3 values (X, Y, Z)
    lut.SetTableRange(0, 2)  # Map indices 0, 1, 2 to the LUT
    lut.Build()
    lut.SetTableValue(0, 1, 0, 0)  # Red for X-axis
    lut.SetTableValue(1, 0, 1, 0)  # Green for Y-axis
    lut.SetTableValue(2, 0, 0, 1)  # Blue for Z-axis

    mapper.SetLookupTable(lut)  # Use the lookup table in the mapper

    # Create an actor for the axes
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor




def detect_contour_subpixel_COPY(image:np.ndarray, start_point, threshold):
    """
    Detecta el contorno de un objeto en escala de grises con precisión subpixel usando un enfoque basado en Marching Squares.

    :param image: np.ndarray, imagen en escala de grises.
    :param start_point: Tuple[int, int], punto inicial dentro del objeto.
    :param threshold: float, umbral para definir el borde.
    :return: List[Tuple[float, float]], puntos del contorno en coordenadas subpixel.
    """

    def interpolate(p1, p2, v1, v2, threshold):
        p1=int(p1)
        p2=int(p2)
        v1=int(v1)
        v2=int(v2)
        diff = float(v2 - v1)
        if abs(diff) < 1e-6:  # Evitar divisiones por cero
            # print("Here v1 equals v2.")
            # print("v1,v2,p1,p2", v1, v2, p1, p2)
            return (p1+p2)/2
        return float(p1 +  (float(threshold - v1) / diff) * (p2 - p1))

    # Dimensiones de la imagen
    height, width = image.shape

    # Posición inicial
    x, y = start_point

    # Verificar que el punto inicial está dentro de los límites de la imagen
    if not (0 <= x < width and 0 <= y < height):
        return None

    s = 1
    # Encontrar el borde inicial moviéndose hacia abajo hasta cruzar el umbral
    while image[y, x] < threshold:
        y += s
    y -= s

    # start point (first square)   
    xo = x
    yo = y

    # first contour point
    px = xo
    py = interpolate(yo, yo + 1, image[yo, xo], image[yo + 1, xo], threshold) 
    
    change_to_left = False
    invert = True

    if change_to_left:
        xo -= 1
        x = xo
    
    # Puntos del contorno
    if invert:
        contour_points = [(px, height-1-py)]
    else:
        contour_points = [(px, py)]
    
    links = [[0,1], [1,2], [2,3], [3,0]]
    link_number = 3 # go right
    
    if change_to_left:
        link_number = 1

    dxdy = [[0,-1], [1,0], [0,1], [-1,0]]

    while True:
        if x < 0 or y < 0:
            break
        if x >= image.shape[1]-2 or y >= image.shape[0]-2:
            break
        
        # Valores de las esquinas de la celda actual
        vert_value = [
            image[y, x],       # Esquina superior izquierda
            image[y, x + 1],   # Esquina superior derecha
            image[y + 1, x + 1], # Esquina inferior derecha
            image[y + 1, x]    # Esquina inferior izquierda
        ]
        
        vert_activity = [v >= threshold for v in vert_value]  

        def link_activity(j):
            link = links[j]
            # XOR operation
            return vert_activity[link[0]] ^ vert_activity[link[1]]
                                 
        for i_link, link in enumerate(links):
            if i_link == link_number:
                continue 
            if link_activity(i_link):                
                break

        if i_link == 0:
            px = interpolate(x, x + 1, vert_value[0], vert_value[1], threshold)
            py = y
            link_number = 2 # go up
        elif i_link == 1:            
            px = x+1
            py = interpolate(y, y + 1, vert_value[1], vert_value[2], threshold)
            link_number = 3 # go right
        elif i_link == 2:
            px = interpolate(x + 1, x, vert_value[2], vert_value[3], threshold)
            # px = interpolate(x, x+1, vert_value[2], vert_value[3])
            py = y+1
            link_number = 0 # go down
        elif i_link == 3:
            px = x
            py = interpolate(y + 1, y, vert_value[3], vert_value[0], threshold)
            link_number = 1 # go left
            
        if invert:
            contour_points.append((px, height-1-py))
        else:
            contour_points.append((px, py))

        dx, dy = dxdy[i_link]
        x, y = x + dx, y + dy
    
        if x==xo and y==yo:
            break
        
    return contour_points


def read_vector_from_file(filename):
    """
    Reads integers from a text file and stores them in a list.

    :param filename: Name of the text file to read from.
    :return: A list of integers read from the file.

    # Example
    pxl_path = pxl_path_to_appdata()
    filepath = os.path.join(pxl_path, "A_codes.txt")
    vector = read_vector_from_file(filename=filepath)
    """
    try:
        # Open the file in read mode
        with open(filename, 'r') as file:
            # Read each line, strip whitespace, and convert to integer
            vector = [int(line.strip()) for line in file]
        
        # Print the loaded vector
        # print(f"Vector read from the file: {vector}")
        return vector
    except FileNotFoundError:
        # Handle the case where the file does not exist
        print(f"Error: File {filename} not found.")
        return []
    except ValueError:
        # Handle the case where the file contains non-integer values
        print("Error: The file contains non-integer data.")
        return []

def adjust_points_to_same_plane_save(vertices):
    """
    Ajusta los puntos no compartidos (v3 y v4) para que los triángulos sean coplanares.

    :param vertices: Lista de 4 puntos en 3D que forman dos triángulos.
                     Ejemplo: [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]
    :return: Nueva lista de vértices con los puntos ajustados.
    """
    v1, v2, v3, v4 = map(np.array, vertices)

    # Crear vectores eliminando la componente Y
    n3 = np.array([v3[0], 0, v3[2]])
    n4 = np.array([v4[0], 0, v4[2]])

    # Normalizar los vectores
    n3 /= np.linalg.norm(n3)
    n4 /= np.linalg.norm(n4)

    # Calcular los parámetros t3 y t4 para coplanaridad
    normal_v1v2 = np.cross(v2 - v1, np.array([0, 1, 0]))  # Normal del plano v1, v2
    normal_v1v2 /= np.linalg.norm(normal_v1v2)

    A = np.dot(normal_v1v2, n3)
    B = np.dot(normal_v1v2, n4)

    C = np.dot(normal_v1v2, v1 - v3)
    D = np.dot(normal_v1v2, v1 - v4)

    t3 = -C / A if A != 0 else 0
    t4 = -D / B if B != 0 else 0

    # Desplazar los puntos v3 y v4
    v3f = v3 + t3 * n3
    v4f = v4 + t4 * n4

    return [v1, v2, v3f, v4f]

##
def tri_points_angle(v1:np.array, v2:np.array, v3:np.array):
    """
    Analyzes the relationship between three points in 3D space.
    Calculates the angle between segments v1-v2 and v2-v3,
    and determines if v2 is above the line formed by v1 and v3 based on specific conditions.

    Parameters:
        v1, v2, v3: np.array
            Coordinates of the points (x, y, z).

    Returns:
        dict: A dictionary with:
            - 'angle': The angle in degrees between the two segments.
            - 'is_above': Boolean indicating if v2 is above the line v1-v3.
            - 'case': Explanation of the behavior when all Z coordinates are zero or other conditions are met.
            - 'projected_point': The point on the line v1-v3 with the same XZ coordinates as v2.
    """
    '''
        # Example usage:
        v1 = np.array([0, 0, 0])
        v2 = np.array([1, 1, 1])
        v3 = np.array([2, 2, 2])

        result = analyze_points(v1, v2, v3)
        print(f"Angle between segments: {result['angle']:.2f} degrees")
        print(f"Is v2 above the line v1-v3? {'Yes' if result['is_above'] else 'No'}")
        print(f"Case: {result['case']}")
        if result['projected_point'] is not None:
        print(f"Projected point on the line v1-v3 with the same XZ as v2: {result['projected_point']}")    
    '''

    def angle_between_vectors(vec1, vec2):
        """Calculates the angle in degrees between two vectors."""
        dot_product = np.dot(vec1, vec2)
        magnitude_vec1 = np.linalg.norm(vec1)
        magnitude_vec2 = np.linalg.norm(vec2)
        cos_theta = dot_product / (magnitude_vec1 * magnitude_vec2)
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clipping to avoid numerical errors
        return np.degrees(angle)

    def project_point_on_line(v1, v3, v2):
        """
        Estimates the point on the line v1-v3 that has the same XZ coordinates as v2.
        Assumes v1, v2, v3 are collinear in the XZ plane.
        """
        t = (v2[0] - v1[0]) / (v3[0] - v1[0]) if v3[0] != v1[0] else (v2[2] - v1[2]) / (v3[2] - v1[2])
        projected_y = v1[1] + t * (v3[1] - v1[1])
        return np.array([v2[0], projected_y, v2[2]])

    # Vectors representing the segments
    segment1 = v2 - v1
    segment2 = v3 - v2

    # Calculate the angle between the segments
    angle = angle_between_vectors(segment1, segment2)

    # Check if v2 is above the line v1-v3
    # is_above_1 = is_point_above_line(v1, v2, v3)

    # Special case: all Z coordinates are zero
    # if v1[2] == v2[2] == v3[2] == 0:
    #     special_case = "All points lie in the XY plane. Analysis limited to the Y-axis."  
    # else:
    #     special_case = "Points have non-zero Z coordinates. Full 3D analysis performed."

    # Calculate the projected point on the line v1-v3
    projected_point = project_point_on_line(v1, v3, v2)
    is_above = v2[1] > projected_point[1]

    return {
        'angle': float(np.round(angle,2)),
        'is_above': bool(is_above),
    }


# Final method
def calculate_angle_between_triangles(vertices):
    """
    Calcula el ángulo entre dos triángulos en 3D en el rango [0, 360] grados.

    :param vertices: Lista de 4 puntos en 3D que forman dos triángulos.
                     Los dos triángulos comparten un lado (dos vértices).
                     Ejemplo: [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]
    :return: Ángulo entre los dos triángulos en grados en el rango [0, 360].
    """
    v1, v2, v3, v4 = map(np.array, vertices)

    # Calcular los vectores de los lados de los triángulos
    tri1_vec1 = v2 - v1
    tri1_vec2 = v3 - v1

    tri2_vec1 = v2 - v1
    tri2_vec2 = v4 - v1

    # Calcular los vectores normales de los triángulos usando el producto cruzado
    normal1 = np.cross(tri1_vec1, tri1_vec2)
    normal2 = np.cross(tri2_vec1, tri2_vec2)

    # Normalizar los vectores normales
    normal1_unit = normal1 / np.linalg.norm(normal1)
    normal2_unit = normal2 / np.linalg.norm(normal2)

    # Calcular el ángulo entre los vectores normales usando el producto escalar
    cos_theta = np.clip(np.dot(normal1_unit, normal2_unit), -1.0, 1.0)
    angle_radians = np.arccos(cos_theta)

    # Determinar el signo del ángulo usando el producto mixto
    direction = np.dot(np.cross(normal1_unit, normal2_unit), tri1_vec1)
    if direction < 0:
        angle_radians = 2 * np.pi - angle_radians

    # Convertir el ángulo a grados
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


# Final method
def adjust_points_to_same_plane(vertices, dt = 0.005, optimization=True, tolerance=0.1, max_iter=200):
    """
    Ajusta los puntos no compartidos (v3 y v4) para que los triángulos sean coplanares.

    :param vertices: Lista de 4 puntos en 3D que forman dos triángulos.
    :param tolerance: Tolerancia para considerar los triángulos coplanares.
    :param max_iter: Número máximo de iteraciones para encontrar los valores óptimos.
    :return: Nueva lista de vértices con los puntos ajustados.
    """
    v1, v2, v3, v4 = map(np.array, vertices)

    # Crear vectores eliminando la componente Y
    n3 = np.array([v3[0]-v2[0], 0, v3[2]]-v2[2])
    n4 = np.array([v4[0]-v2[0], 0, v4[2]]-v2[2])

    # Normalizar los vectores
    n3 /= np.linalg.norm(n3)
    n4 /= np.linalg.norm(n4)

    axis = v1 - v2
    axis /= np.linalg.norm(axis)
    
    # Inicializar valores de t3 y t4
    iter = 0
    angles = []
    ts = []
    
    if optimization:
        t3, t4 = 0.05, 0.05   # Valores iniciales cercanos a los observados
    else:
        t3, t4 = 0.00, 0.00

    for _ in range(max_iter):

        # Calcular nuevos puntos ajustados
        v3f = v3 + t3 * n3
        v4f = v4 + t4 * n4

        angle = calculate_angle_between_triangles([v1, v2, v3f, v4f])
        angle = angle - 180

        if optimization:
            # print("angle:", angle)
            if iter == 0:
                v3fo = v3f
                v4fo = v4f
                # print("v3f, v4f: ", v3f, v4f, iter)                
            # Verificar si el ángulo está dentro de la tolerancia
            if abs(angle) < tolerance:
                # print("t3, t4, iter: ", t3, t4, iter)
                # print("v3fo, v4fo: ", v3fo, v4fo)
                # print("v3f, v4f: ", v3f, v4f, iter)
                return [v1, v2, v3f, v4f]
            # Actualizar t3 y t4 de forma iterativa para reducir el ángulo
            gradient = -angle * 0.001  # Pequeños ajustes
            t3 += gradient
            t4 += gradient
        else:
            max_iter = 50
            angles.append(float(angle))
            ts.append(t3)
            t3 += dt
            t4 += dt        
        iter += 1

    for angle in angles:
        # print(angle)
        if angle >-5 and angle < 5:
            print(angle)
    # print("No se alcanzó la coplanaridad dentro del número máximo de iteraciones.")
    # print("t3, t4, iter: ", t3, t4, iter)
    return [v1, v2, v3f, v4f]

def show_image(image, window_name="Test"):            
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Permitir ajuste dinámico
    cv2.resizeWindow(window_name, 800, 502) 
    cv2.moveWindow(window_name, 50, 50)
        
    # Mostrar la imagen usando OpenCV
    cv2.imshow(window_name, image)
    cv2.waitKey(0)  # Esperar a que se presione una tecla
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Ejemplo de uso
    # pxl_path = pxl_path_to_appdata()
    # filepath = os.path.join(pxl_path, "A_codes.txt")
    # vector = read_vector_from_file(filename=filepath)
    # test_input_output_arrays()
    pass




