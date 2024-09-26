import numpy as np
import cv2

def convert_markers_to_pixels(markers_cm, workspace_cm, workspace_px):
    """
    Converte i marker da coordinate in cm a coordinate in pixel utilizzando una trasformazione affine.

    INPUT:
    - markers_cm: array di marker in centimetri (Nx2)
    - workspace_cm: array di punti angolari del workspace in cm (4x2)
    - workspace_px: array di punti angolari del workspace in pixel (4x2)

    OUTPUT:
    - markers_px: array di marker convertiti in pixel (Nx2)
    """
    # Calcola la matrice di trasformazione affine
    M, _ = cv2.findHomography(workspace_cm, workspace_px)

    # Converti i marker in pixel usando la trasformazione
    markers_px = cv2.perspectiveTransform(np.array([markers_cm], dtype='float32'), M)
    
    return markers_px[0]  # Elimina la dimensione in più per facilità di utilizzo

# Definizione dei marker in centimetri
markers_cm = np.array([[0.0, 0.0], [21.5, 0.0], [43.0, 0.0], 
                       [0.0, 15.0], [21.5, 15.0], [43.0, 15.0],
                       [0.0, 30.0], [21.5, 30.0], [43.0, 30.0]])

# Punti angolari del workspace in centimetri (ad esempio, 43x30 cm)
workspace_cm = np.array([[0, 0], [100, 0], [0, 70], [100, 70]])

# Punti angolari del workspace in pixel
workspace_px = np.array([(153, 629), (985, 631), (126, 37), (1004, 23)])

# Converti i marker in pixel
markers_px = convert_markers_to_pixels(markers_cm, workspace_cm, workspace_px)

# Stampa i marker convertiti in pixel
for i, marker in enumerate(markers_px):
    print(f"Marker {i+1}: {marker}")
