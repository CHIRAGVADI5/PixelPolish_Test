import numpy as np

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

def example():
    # Secuencias de ejemplo
    A = [21.9, 43.1, 67.4, 90.6, 112.7, 135.5, 157.6, 180.2, 202.3, 225.2, 247.4, 272.0, 293.8, 314.9, 338.4, 359.8]
    B = [20.9, 43.3, 69.5, 88.8, 118.4, 133.6, 158.8, 180.7, 202.5, 224.5, 247.5, 272.8, 293.6, 315.3, 338.5, 359.7]
    # B = [20.9, 43.3, 69.5, 88.8, 95.1, 113.4, 133.6, 158.8, 180.7, 202.5, 224.5, 247.5, 272.8, 293.6, 315.3, 338.5]

    if is_sequence_valid(A, B):
        print("La secuencia B es válida.")
    else:
        corrected_B = correct_sequence(A, B)
        print("Secuencia B corregida:", corrected_B)


