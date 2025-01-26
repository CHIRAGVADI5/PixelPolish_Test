import os
import numpy as np
import pandas as pd
import csv
# import time
from scipy.signal import find_peaks

'''
This release includes:
- The both_true_girdle_peek_detection method.
- Plots using matplotlib were deleted.
- Saved girdle peak data.

pyinstaller --onedir pxl_girdle3.py

The following does not work:
pyinstaller --onedir --exclude-module=pytest --exclude-module=numpy.testing --exclude-module=matplotlib.testing --exclude-module=scipy.spatial --exclude-module=unittest --exclude-module=logging --exclude-module=pandas.tests pxl_girdle2.py
'''

def wrap_angle(angle):
    """Envuelve un ángulo dentro del rango [0, 360)."""
    return angle % 360

def calculate_circular_difference(a, b):
    """Calcula la diferencia circular mínima entre dos ángulos."""
    diff = abs(a - b)
    return min(diff, 360 - diff)

def circular_interpolation(prev, next):
    """Realiza una interpolación circular entre dos ángulos."""
    if next > prev:
        interpolated = prev + (next - prev) / 2
    else:
        interpolated = prev + (next + 360 - prev) / 2
    return wrap_angle(interpolated)

def is_sequence_valid(A, B, threshold=5.0):
    """
    Verifica si la secuencia B es válida en comparación con A.

    Args:
        A (list of float): Secuencia de referencia.
        B (list of float): Secuencia a validar.
        threshold (float): Umbral de diferencia aceptable entre elementos.

    Returns:
        bool: True si B es válida, False en caso contrario.
    """
    if len(B) != len(A):
        return False
    for a, b in zip(A, B):
        if calculate_circular_difference(a, b) > threshold:
            return False
    return True

def correct_sequence(A, B):
    """
    Corrige la secuencia B basada en la secuencia A.

    Args:
        A (list of float): Secuencia de referencia.
        B (list of float): Secuencia a corregir.

    Returns:
        list: Secuencia B corregida.
    """
    if is_sequence_valid(A, B):
        return B  # Si B es válida, no hacer cambios

    A = np.array([wrap_angle(a) for a in A])
    B = np.array([wrap_angle(b) for b in B])

    # Detectar errores en B
    paired_indices = []
    for i, a in enumerate(A):
        differences = [calculate_circular_difference(a, b) for b in B]
        min_diff_index = np.argmin(differences)
        if differences[min_diff_index] <= 22.5:  # Umbral basado en separación promedio
            paired_indices.append(min_diff_index)
        else:
            paired_indices.append(None)

    # Construir B corregida
    corrected_B = []
    used_indices = set()
    for i, a in enumerate(A):
        if paired_indices[i] is not None and paired_indices[i] not in used_indices:
            interpolated_value = circular_interpolation(A[i - 1], A[(i + 1) % len(A)])
            if calculate_circular_difference(interpolated_value, B[paired_indices[i]]) > 5.0:
                corrected_B.append(a)  # Reemplazar con el valor de A si el error supera el umbral
            else:
                corrected_B.append(B[paired_indices[i]])
                used_indices.add(paired_indices[i])
        else:
            corrected_B.append(a)  # Si no hay par, usar el valor de A

    # Agregar términos faltantes
    if len(corrected_B) < len(A):
        corrected_B += [a for i, a in enumerate(A) if i not in paired_indices]

    # Asegurar que la longitud sea consistente
    corrected_B = corrected_B[:len(A)]  # Truncar si hay más elementos

    # Interpolar si es necesario
    for i in range(len(corrected_B)):
        if corrected_B[i] == A[i]:
            prev_idx = (i - 1) % len(corrected_B)
            next_idx = (i + 1) % len(corrected_B)
            corrected_B[i] = circular_interpolation(corrected_B[prev_idx], corrected_B[next_idx])

    return [float(wrap_angle(b)) for b in corrected_B]

# **************************************************************************************

def peak_detector(y, positive=True):

    y_smooth = None
    if positive:
        # Detectar picos con scipy
        # Ajustar 'distance' según la cantidad esperada de picos (16 en este caso)
        peaks, _ = find_peaks(y, distance=10, prominence=0.005)

        # Si no se detectan exactamente 16 picos, ajustamos
        if len(peaks) != 16:
            peaks, _ = find_peaks(y, distance=5, prominence=0.002)
    else:
        # Trabajar con el negativo de 'y'
        y_neg = -y

        # Detectar picos en el negativo de 'y'
        peaks, _ = find_peaks(y_neg, distance=10, prominence=0.005)

        # Si no se detectan exactamente 16 picos, ajustamos
        if len(peaks) != 16:
            peaks, _ = find_peaks(y_neg, distance=5, prominence=0.002)

    return peaks, y_smooth


def pxl_path_to_appdata():
    local_dir = os.getenv('LOCALAPPDATA')

    pxl_path = os.path.join(local_dir, "PixelPolish3D")

    if not os.path.exists(pxl_path):
        os.mkdir(pxl_path)

    return pxl_path


def both_true_girdle_peek_detection(code_id, save = False, path = None):

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

        # verify if the two last sequences has the same length and 5º maximum differences 
        is_valid = is_sequence_valid(angles_upper_peaks, angles_lower_peaks)

        if not is_valid:
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
            # print(f"Archivo CSV salvado como '{output_file}'")
            
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
                
    # Mostrar las posiciones y valores de los picos detectados
    # picos_detectados = pd.DataFrame({'Índice': peaks, 'Valores': y[peaks]})
    # print(picos_detectados)
    # end of both_true_girdle_peek_detection()


def get_latest_code_id(folder_path, prefix, extension):
    """
    Identifies the most recently written file in a folder matching a specific prefix and extension,
    and extracts the code_id from its name.

    :param folder_path: Path to the folder where files are located.
    :param prefix: The prefix of the file name.
    :param extension: The file extension.
    :return: The code_id of the most recent file or None if no files match.
    """
    # List all files in the folder that match the given prefix and extension
    files = [
        f for f in os.listdir(folder_path)
        if f.startswith(prefix) and f.endswith(extension)
    ]
    
    # If no files match, return None
    if not files:
        return None

    # Find the most recently modified file
    latest_file = max(
        files,
        key=lambda f: os.path.getmtime(os.path.join(folder_path, f))
    )

    # Extract the code_id by removing the prefix and extension
    code_id = latest_file[len(prefix):-len(extension)]
    
    return code_id


if __name__ == "__main__":
     
    # start_time = time.time()

    folder_path = pxl_path_to_appdata()
    prefix = "lower_girdle_coords_"
    extension = ".csv"

    # Get the code_id of the most recent file
    code_id = get_latest_code_id(folder_path, prefix, extension)

    if code_id is None:
        exit

    # code_id = "LHPO_Round_Polished_400_0002_11p7_2024-08-03"
    both_true_girdle_peek_detection(code_id, save=True, path=None)

    # lapse = round(time.time() - start_time, 3)
    # print("girdle peaks analysis:", lapse)
        
