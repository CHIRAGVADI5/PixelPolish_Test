
import os
import numpy as np
import time

import ctypes
from ctypes import POINTER, c_int, c_float, c_double, c_char_p, c_long

import vtk
import cv2
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk

# import platform
# print(platform.architecture())
enable_print = False

from Pixel3D.pxl_tools import pxl_path_to_appdata
app_data_path = pxl_path_to_appdata()

# STEP #1 FOR INTEGRATION (DLL relocalization)
'''
Adjust the DLL location to your selection. 
It is currently located in the relative path "libs".
'''
# DLL Loading
dll_path = os.path.join(os.path.dirname(__file__), "libs", "PixelPolish3DPyV02.dll")
if os.path.exists(dll_path):
    native_processor = ctypes.CDLL(dll_path)



#  get cleaned points (12 to 16 points) from decimated points (51 points)
def get_cleaned_points_from_decimated(decimated_point_array:np.ndarray, image_number, image_qtty):
    
    '''
        this dll method is used:
    	PIXELPOLISH3DPYV02_API const double* get_cleaned_points_from_decimated(
            const double* decimated_points, 
            size_t length_decimated, 
            size_t* length_cleaned, 
            size_t images_qtty, 
            size_t imageNumber);
    '''
    
    # PIXELPOLISH3DPYV02_API const double* get_cleaned_points_from_decimated
    #   (const double* decimated_points, size_t length_decimated, 
    #       size_t* length_cleaned, size_t images_qtty, size_t imageNumber);

    # *** Configurate the prototipe funtion ***

    native_processor.get_cleaned_points_from_decimated.argtypes = [ 
        ctypes.POINTER(ctypes.c_double),    # pointer to c_double (double array)
        ctypes.c_size_t,                    # input array size (size_t)
        ctypes.POINTER(ctypes.c_size_t),    # output array size as pointer to c_size_t
        ctypes.c_size_t,                    # image number (size_t)
        ctypes.c_size_t                     # image qtty (size_t)
    ]

    native_processor.get_cleaned_points_from_decimated.restype = ctypes.POINTER(ctypes.c_double)

    # decimated points to ndarray, 51 points
    # points = decimated_edges.GetPoints()
    # decimated_point_array = vtk_to_numpy(points.GetData())
    # decimated_point_array:np.ndarray = decimated_point_array.flatten()
    decimated_lenght = decimated_point_array.shape[0]
    
    # Convertir la lista a un array de C de tipo double
    
    array_type = ctypes.c_double * decimated_lenght  # Crear el tipo array
    c_array = array_type(*decimated_point_array)  # Convertir la lista a array
    
    cleaned_length = ctypes.c_size_t()
    double_ptr = native_processor.get_cleaned_points_from_decimated( \
                    c_array, \
                    decimated_lenght, \
                    ctypes.byref(cleaned_length), image_number, image_qtty)
    
    cleaned_array = [double_ptr[i] for i in range(cleaned_length.value)]
    return cleaned_array # , decimated_point_array.tolist()
    pass


def push_one_image_no_memory_optimized(image_path):
    
    # Definición de la función process_image de la DLL
    # int process_image(unsigned char* imageData, int width, int height)
    native_processor.push_one_image.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int]
    native_processor.push_one_image.restype = ctypes.c_int  # La función ahora devuelve un int (0 = éxito, -1 = error)

    """Lee una imagen BW de 8 bits y la envía a la DLL en formato unsigned char*."""
    # Leer la imagen en escala de grises (BW) con OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        if enable_print:
            print(f"Error: No se pudo leer la imagen en la ruta {image_path}")
        return

    # Obtener las dimensiones de la imagen
    height, width = image.shape

    # Asegurar que la imagen esté en formato 8 bits (por si acaso)
    image_8bit = image.astype(np.uint8)

    # Convertir la imagen en un array de bytes (unsigned char)
    image_bytes = image_8bit.tobytes()

    # Convertir a un tipo adecuado para ctypes
    image_ptr = ctypes.cast(ctypes.create_string_buffer(image_bytes), ctypes.POINTER(ctypes.c_ubyte))

    # Llamar a la función de la DLL
    result = native_processor.push_one_image(image_ptr, width, height, 1, 400)
    
    # Verificar la confirmación recibida
    if enable_print:
        if result == 0:
            print("✔️ La imagen fue recibida y procesada correctamente.")
        else:
            print("❌ Hubo un error al procesar la imagen.")


# memory optimized code for sending one image to DLL
def push_one_image_to_dll(image_path):

    # Definición de la función push_one_image de la DLL
    native_processor.push_one_image.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    native_processor.push_one_image.restype = ctypes.c_int

    """Envía una imagen BW de 8 bits a la DLL."""
    # 1️⃣ Leer la imagen en escala de grises y asegurarse de que está en formato uint8
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"❌ Error: No se pudo leer la imagen en la ruta {image_path}")
        return

    # Obtener las dimensiones de la imagen
    height, width = image.shape

    # 2️⃣ Obtener los bytes directamente de la imagen usando ravel() para evitar duplicados
    image_bytes = image.ravel().tobytes()

    start_time = time.time()

    # 3️⃣ Crear un buffer de C (memoria contigua) para pasar la imagen a la DLL
    image_data = ctypes.create_string_buffer(image_bytes)  # Solo una copia final

    # 4️⃣ Llamar a la función de la DLL con el puntero de la imagen
    result = native_processor.push_one_image(ctypes.cast(image_data, ctypes.POINTER(ctypes.c_ubyte)), width, height, 1, 400)
    
    # Calculate the time lapse
    lapse = round(time.time() - start_time, 3)
    print("Loading lapse:", lapse)

    if result == 0:
        print("✔️ La imagen fue recibida y procesada correctamente.")
    else:
        print("❌ Hubo un error al procesar la imagen.")




def push_all_images_to_dll(image_paths):

    """Sends a list of N (400, 800) 8-bit BW images to the C++ DLL."""

    # Define the function signature for push_all_images in the DLL
    native_processor.push_all_images.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int, ctypes.c_int]
    native_processor.push_all_images.restype = ctypes.c_int
    
    # 1️⃣ Create a single contiguous memory block to store all images
    total_images = len(image_paths)  # Number of images to send
    
    offset = 0  # Position in the bytearray where the next image will be copied
    height, width = 0, 0
    total_bytes = 0  # Total memory required for all images
    all_image_bytes = None  # Create a contiguous array of bytes
    offset = 0  # Position in the bytearray where the next image will be copied
    
    for i, image_path in enumerate(image_paths):
        # 2️⃣ Read the image in grayscale (8-bit format) 
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"❌ Error: Could not read the image at {image_path}")
            return

        if i == 0:
            # get the image dimensions
            height, width = image.shape
            total_bytes = total_images * width * height  # Total memory required for all images
            all_image_bytes = bytearray(total_bytes)  # Create a contiguous array of bytes
            
        # Check if the image has the correct size
        if image.shape != (height, width):
            print(f"❌ Error: Image {image_path} does not have the expected size ({width}x{height}), it has {image.shape}")
            return
        
        # 3️⃣ Flatten the image into a 1D view (no memory copy) and copy it into the final bytearray
        image_bytes = image.ravel()  # Create a 1D view of the image (no data copy)
        all_image_bytes[offset:offset + len(image_bytes)] = image_bytes.tobytes()  # Copy the image as bytes
        offset += len(image_bytes)  # Move the position for the next image

    start_time = time.time()

    # 4️⃣ Create a C-style contiguous buffer from all images (6.25 MB for 400 images of 128x128)
    image_data = ctypes.create_string_buffer(bytes(all_image_bytes))  # Single final buffer
    
    # 5️⃣ Call the DLL function with the pointer to the contiguous memory buffer
    result = native_processor.push_all_images(ctypes.cast(image_data, ctypes.POINTER(ctypes.c_ubyte)), width, height, total_images)
    
    # Calculate the time lapse
    lapse = round(time.time() - start_time, 3)
    print("Loading lapse:", lapse)
    
    if result == 0:
        print("✔️ All images were successfully received and processed.")
    else:
        print("❌ An error occurred while processing the images.")



# Llamar a la función
def save_polygons(case_id: str, w: int, h: int, pixel_size: float) -> int:
    """
    Wrapper para la función save_polygons en la DLL.
    
    Usage:
    case_id = "example_case"
    width = 1936
    height = 1260
    pixel_size = 0.01172
    status = save_polygons(case_id, width, height, pixel_size)
    print(f"Result of save_polygons: {status}")    
    """
    # Configurar el prototipo de la función
    native_processor.save_polygons.argtypes = [c_char_p, c_int, c_int, c_float]
    native_processor.save_polygons.restype = c_int

    case_id_bytes = case_id.encode('utf-8')  # Convertir string a bytes
    result = native_processor.save_polygons(case_id_bytes, w, h, pixel_size)
    return result


def transfer_fast_3d_model_arrays(points_array: list, normals_array: list, cell_type_array: list, vertice_facetas: dict, faceta_vertices: dict, N: int, code_id: str):

    # Define the argument types for the DLL function
    native_processor.transfer_fast_3d_model_arrays.argtypes = [
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # points_array
        ctypes.POINTER(ctypes.c_int),  # sizes of sub-arrays for points_array
        ctypes.c_int,  # total number of sub-arrays in points_array

        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # normals_array
        ctypes.POINTER(ctypes.c_int),  # sizes of sub-arrays for normals_array
        ctypes.c_int,  # total number of sub-arrays in normals_array

        ctypes.POINTER(ctypes.c_int),  # cell_type_array
        ctypes.c_int,  # size of cell_type_array

        ctypes.POINTER(ctypes.c_int),  # vertice_facetas keys
        ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),  # vertice_facetas values
        ctypes.POINTER(ctypes.c_int),  # sizes of sub-arrays for vertice_facetas
        ctypes.c_int,  # total number of keys in vertice_facetas
        
        ctypes.POINTER(ctypes.c_int),  # vertice_facetas keys
        ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),  # vertice_facetas values
        ctypes.POINTER(ctypes.c_int),  # sizes of sub-arrays for vertice_facetas
        ctypes.c_int,  # total number of keys in vertice_facetas

        ctypes.c_int,  # N (new argument)
        c_char_p       # const char* code_id
    ]

    # Convert points_array to ctypes
    points_array_ctypes = (ctypes.POINTER(ctypes.c_double) * len(points_array))()
    points_sizes = (ctypes.c_int * len(points_array))()
    for i, sub_array in enumerate(points_array):
        points_array_ctypes[i] = sub_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        points_sizes[i] = len(sub_array)

    # Convert normals_array to ctypes
    normals_array_ctypes = (ctypes.POINTER(ctypes.c_double) * len(normals_array))()
    normals_sizes = (ctypes.c_int * len(normals_array))()
    for i, sub_array in enumerate(normals_array):
        normals_array_ctypes[i] = sub_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        normals_sizes[i] = len(sub_array)

    # Convert cell_type_array to ctypes
    cell_type_array_ctypes = (ctypes.c_int * len(cell_type_array))(*cell_type_array)

    # Convert vertice_facetas to ctypes
    vertice_keys_ctypes = (ctypes.c_int * len(vertice_facetas))(*vertice_facetas.keys())
    vertice_values_ctypes = (ctypes.POINTER(ctypes.c_int) * len(vertice_facetas))()
    vertice_sizes = (ctypes.c_int * len(vertice_facetas))()
    for i, (key, value) in enumerate(vertice_facetas.items()):
        array = (ctypes.c_int * len(value))(*value)
        vertice_values_ctypes[i] = ctypes.cast(array, ctypes.POINTER(ctypes.c_int))
        vertice_sizes[i] = len(value)

    # Convert faceta_vertices to ctypes
    facet_keys_ctypes = (ctypes.c_int * len(faceta_vertices))(*faceta_vertices.keys())
    facet_values_ctypes = (ctypes.POINTER(ctypes.c_int) * len(faceta_vertices))()
    facet_sizes = (ctypes.c_int * len(faceta_vertices))()
    for i, (key, value) in enumerate(faceta_vertices.items()):
        array = (ctypes.c_int * len(value))(*value)
        facet_values_ctypes[i] = ctypes.cast(array, ctypes.POINTER(ctypes.c_int))
        facet_sizes[i] = len(value)

    code_id = code_id.encode('utf-8')

    # Call the DLL function
    native_processor.transfer_fast_3d_model_arrays(
        points_array_ctypes, points_sizes, len(points_array),
        normals_array_ctypes, normals_sizes, len(normals_array),
        cell_type_array_ctypes, len(cell_type_array),
        vertice_keys_ctypes, vertice_values_ctypes, vertice_sizes, len(vertice_facetas), 
        facet_keys_ctypes, facet_values_ctypes, facet_sizes, len(faceta_vertices), N, code_id
    )
    

    # +++++++++++++++++++++++++++++++++++++++++++++++++++OnRealModel3D++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def transfer_precise_3d_model(code_id: str):

    '''
    DLL side using C++:
    
        __declspec(dllexport) void transfer_precise_3d_model(
                const char* code_id,
                int** flattened_vector,         // Salida: vector aplanado
                int* flattened_size,            // Salida: tamaño total del vector aplanado
                int** subvector_sizes,          // Salida: array de longitudes de los sub-vectores
                int* num_subvectors,            // Salida: número total de sub-vectores
                float** tuple_array,            // Salida: array de tuplas
                int* tuple_size,                // Salida: tamaño total del array de tuplas
                double* array_of_doubles,       // Salida: array de 3 doubles
                double* single_double)          // Salida: single double
    '''

    # Definir los tipos de argumentos de la función
    native_processor.transfer_precise_3d_model.argtypes = [
        c_char_p,                         # const char* code_id
        POINTER(POINTER(c_int)),          # int** flattened_vector
        POINTER(c_int),                   # int* flattened_size
        POINTER(POINTER(c_int)),          # int** subvector_sizes
        POINTER(c_int),                   # int* num_subvectors
        POINTER(POINTER(c_float)),        # float** tuple_array
        POINTER(c_int),                   # int* tuple_size
        POINTER(c_double),                # double* array_of_doubles
        POINTER(c_double)                 # double* single_double
    ]

    # 1. Variables de salida
    flattened_vector = POINTER(c_int)()
    flattened_size = c_int()

    subvector_sizes = POINTER(c_int)()
    num_subvectors = c_int()

    tuple_array = POINTER(c_float)()
    tuple_size = c_int()

    array_of_doubles = (c_double * 3)()
    single_double = c_double()

    # code_id = b"example_code_id"

    code_id = code_id.encode('utf-8')

    # 2. Llamar a la función DLL
    native_processor.transfer_precise_3d_model(
        code_id,
        ctypes.byref(flattened_vector),
        ctypes.byref(flattened_size),
        ctypes.byref(subvector_sizes),
        ctypes.byref(num_subvectors),
        ctypes.byref(tuple_array),
        ctypes.byref(tuple_size),
        array_of_doubles,
        ctypes.byref(single_double)
    )

    # 3. Reconstruir vector_of_vectors
    flattened_data = [flattened_vector[i] for i in range(flattened_size.value)]
    subvector_sizes_data = [subvector_sizes[i] for i in range(num_subvectors.value)]

    vector_of_vectors_result = []
    offset = 0
    for size in subvector_sizes_data:
        sub_vector = flattened_data[offset:offset + size]
        vector_of_vectors_result.append(sub_vector)
        offset += size

    if enable_print:
        print("Vector of Vectors of Ints:", vector_of_vectors_result[:5])
        print("Number of vectors of Ints:", len(vector_of_vectors_result))

    # 4. Extraer tuple_array
    tuple_array_result = []
    for i in range(tuple_size.value // 3):
        start = i * 3
        int_val = int(tuple_array[start])
        float_val1 = float(tuple_array[start + 1])
        float_val2 = float(tuple_array[start + 2])
        tuple_array_result.append((int_val, float_val1, float_val2))
    if enable_print:
        print("Vector of Tuples (int, float, float):", tuple_array_result)

    # 5. Extraer array_of_doubles
    array_of_doubles_result = [array_of_doubles[i] for i in range(3)]
    if enable_print:
        print("Array of 3 Doubles:", array_of_doubles_result)

    # 6. Extraer single_double
    single_double_result = single_double.value
    if enable_print:
        print("Single Double Value:", single_double_result)
    
    # returns the following:
    # cells, pavilion_locations, table_center, minAngle
    return vector_of_vectors_result, tuple_array_result, array_of_doubles_result, single_double_result
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++


# def execute_precise_3d_model_BAD():

#     # Define the argument types for Execute_
#     native_processor.execute_precise_3d_model.argtypes = [
#         c_char_p,                        # const char* code_id
#         POINTER(POINTER(c_int)),         # int*** vector_of_vectors
#         POINTER(POINTER(c_int)),         # int** vector_sizes
#         POINTER(c_int),                  # int* num_vectors
#         POINTER(POINTER(c_float)),       # float** tuple_array
#         POINTER(c_int),                  # int* tuple_size
#         POINTER(c_double),               # double* array_of_doubles
#         POINTER(c_double)                # double* single_double
#     ]

#     # Variables to hold the output from C++
#     vector_of_vectors = POINTER(POINTER(c_int))()
#     vector_sizes = POINTER(c_int)()
#     num_vectors = c_int()

#     tuple_array = POINTER(c_float)()
#     tuple_size = c_int()

#     array_of_doubles = (c_double * 3)()
#     single_double = c_double()

#     # Call the function
#     code_id = b"example_code_id"  # Convert string to bytes
#     native_processor.execute_precise_3d_model(
#         code_id,
#         ctypes.byref(vector_of_vectors),
#         ctypes.byref(vector_sizes),
#         ctypes.byref(num_vectors),
#         ctypes.byref(tuple_array),
#         ctypes.byref(tuple_size),
#         array_of_doubles,
#         ctypes.byref(single_double)
#     )

#     # Extract vector of vectors of ints
#     vector_of_vectors_result = []
#     for i in range(num_vectors.value):
#         sub_array = [vector_of_vectors[i][j] for j in range(vector_sizes[i])]
#         vector_of_vectors_result.append(sub_array)
#     print("Vector of Vectors of Ints:", vector_of_vectors_result)

#     # Extract tuple array (flattened into a single array)
#     tuple_array_result = []
#     for i in range(tuple_size.value // 3):
#         start = i * 3
#         int_val = int(tuple_array[start])  # Extract integer
#         float_val1 = float(tuple_array[start + 1])  # Extract first float
#         float_val2 = float(tuple_array[start + 2])  # Extract second float
#         tuple_array_result.append((int_val, float_val1, float_val2))
#     print("Vector of Tuples (int, float, float):", tuple_array_result)

#     # Extract array of 3 doubles
#     array_of_doubles_result = [array_of_doubles[i] for i in range(3)]
#     print("Array of 3 Doubles:", array_of_doubles_result)

#     # Extract single double
#     single_double_result = single_double.value
#     print("Single Double Value:", single_double_result)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# def execute_precise_3d_model_LAST():
#     pass

#     # Define the argument types for the function
#     # native_processor.execute_precise_3d_model.argtypes = [
#     # from ctypes import POINTER, c_int, c_float, c_double, c_char_p, c_long

#     dll = native_processor

#     # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     # ++++++++++++++++++++++++++++++++++++++++++++++++++++++


#     # Definir los tipos de argumentos de la función
#     dll.execute_precise_3d_model.argtypes = [
#         c_char_p,                         # const char* code_id
#         POINTER(POINTER(c_long)),         # int*** vector_of_vectors
#         POINTER(POINTER(c_int)),          # int** vector_sizes
#         POINTER(c_int),                   # int* num_vectors
#         POINTER(POINTER(c_float)),        # float** tuple_array
#         POINTER(c_int),                   # int* tuple_size
#         POINTER(c_double),                # double* array_of_doubles
#         POINTER(c_double)                 # double* single_double
#     ]

#     # 1. Crear la variable de salida vector_of_vectors
#     num_vectors = c_int(3)  # Supongamos que tenemos 3 sub-vectores
#     vector_sizes = (c_int * num_vectors.value)(3, 2, 4)  # Tamaños de cada sub-vector

#     # Crear la estructura de vector_of_vectors
#     vector_of_vectors = (POINTER(c_long) * num_vectors.value)()

#     # Asignar sub-vectores
#     vector_data_1 = (c_long * 3)(1, 2, 3)  # Primer sub-vector
#     vector_data_2 = (c_long * 2)(4, 5)     # Segundo sub-vector
#     vector_data_3 = (c_long * 4)(6, 7, 8, 9)  # Tercer sub-vector

#     # Asignar la dirección de cada sub-vector
#     vector_of_vectors[0] = ctypes.cast(vector_data_1, POINTER(c_long))
#     vector_of_vectors[1] = ctypes.cast(vector_data_2, POINTER(c_long))
#     vector_of_vectors[2] = ctypes.cast(vector_data_3, POINTER(c_long))

#     # 2. Variables de salida adicionales
#     tuple_size = c_int()
#     tuple_array = POINTER(c_float)()

#     array_of_doubles = (c_double * 3)()
#     single_double = c_double()

#     # 3. Llamar a la función DLL
#     code_id = b"example_code_id"
#     dll.execute_precise_3d_model(
#         code_id,
#         ctypes.cast(vector_of_vectors, POINTER(POINTER(c_long))),
#         ctypes.cast(vector_sizes, POINTER(c_int)),
#         ctypes.byref(num_vectors),
#         ctypes.byref(tuple_array),
#         ctypes.byref(tuple_size),
#         array_of_doubles,
#         ctypes.byref(single_double)
#     )

#     # 4. Extraer los valores de vector_of_vectors
#     vector_of_vectors_result = []
#     for i in range(num_vectors.value):
#         sub_vector_ptr = ctypes.cast(vector_of_vectors[i], POINTER(c_long))
#         sub_array = [sub_vector_ptr[j] for j in range(vector_sizes[i])]
#         vector_of_vectors_result.append(sub_array)
#     print("Vector of Vectors of Ints:", vector_of_vectors_result)

#     # 5. Extraer tuple_array
#     tuple_array_result = []
#     for i in range(tuple_size.value // 3):
#         start = i * 3
#         int_val = int(tuple_array[start])
#         float_val1 = float(tuple_array[start + 1])
#         float_val2 = float(tuple_array[start + 2])
#         tuple_array_result.append((int_val, float_val1, float_val2))
#     print("Vector of Tuples (int, float, float):", tuple_array_result)

#     # 6. Extraer array_of_doubles
#     array_of_doubles_result = [array_of_doubles[i] for i in range(3)]
#     print("Array of 3 Doubles:", array_of_doubles_result)

#     # 7. Extraer single_double
#     single_double_result = single_double.value
#     print("Single Double Value:", single_double_result)


if __name__ == "__main__":
    
    folderpath = r"C:\Datos\PixelPolish\Round Polished\LHPO_Round_Polished_400_0002_11p7_2024-08-03\Images"
    
    option = 5
    
    if option==1:
        # Prueba la función con una imagen
        image_path = os.path.join(folderpath, "Photo001.bmp")
        push_one_image_to_dll(image_path)
    
    elif option==2:
        image_paths = [os.path.join(folderpath, f)
                        for f in os.listdir(folderpath)
                        if os.path.splitext(f)[1].lower()==".bmp"
                        and os.path.isfile(os.path.join(folderpath, f))]
        push_all_images_to_dll(image_paths)
        
    elif option==3:
        case_id = "example_case"
        width = 1936
        height = 1260
        pixel_size = 0.01172
        status = save_polygons(case_id, width, height, pixel_size)
        print(f"Result of save_polygons: {status}")    
        
    elif option==4:
        # Example data
        points_array = [np.array([1.1, 2.2, 3.3]), np.array([4.4, 5.5, 6.6])]
        normals_array = [np.array([0.05, 0.1, 0.9]), np.array([0.2, 0.7, 0.3])]
        cell_type_array = [5, 7, 5, 7, 7]
        vertice_facetas = {0: [1, 2, 3, 4, 5, 6], 1: [7, 8, 9, 10], 2: [11, 12, 13]}
        transfer_fast_3d_model_arrays(points_array, normals_array, cell_type_array, vertice_facetas)
        pass
    
    elif option==5:
        transfer_precise_3d_model()
        # transfer_precise_3d_model()
        pass    
        

