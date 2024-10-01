import numpy as np

# Definisci i punti in pixel
saved_points = np.array([[1424, 271, 1], [1135, 263, 1], [854, 253, 1],
                          [1408, 472, 1], [1124, 461, 1], [848, 449, 1],
                          [1395, 668, 1], [1114, 652, 1], [842, 640, 1]])

# Posizioni desiderate
positions = np.array([[0.25, -0.2275, 0.6], [0.25, 0.0, 0.6], 
                      [0.25, 0.2275, 0.6], [0.25, -0.2275, 0.45], 
                      [0.25, 0.0, 0.45], [0.25, 0.2275, 0.45], 
                      [0.25, -0.2275, 0.3], [0.25, 0.0, 0.3], 
                      [0.25, 0.2275, 0.3]])[::-1]  # Inverti le posizioni

# Matrice della telecamera K
K = np.array([[4953.408647187136, 0.0, 973.6655217744913],
              [0.0, 5484.191359255874, 542.1833720711198],
              [0.0, 0.0, 1.0]])

# Vettore di traduzione t
t = np.array([351.0406152926042, -193.0561916697725, 3728.635914013687])

# Converti i punti in pixel in metri
def px2meters(pt, K, t):
    # Converti il punto in array NumPy se non lo è già
    pt = np.array(pt[:2], dtype=np.float64)  # Usa solo x e y
    K = np.array(K, dtype=np.float64)
    t = np.array(t, dtype=np.float64)

    # Aggiungi una dimensione a pt per il calcolo
    pt_h = np.array([pt[0], pt[1], 1.0])  # Coordinate omogenee

    # Calcola il punto proiettato in coordinate della fotocamera
    projected = K @ pt_h  # Proietta il punto usando la matrice della fotocamera
    projected /= projected[2]  # Normalizza per ottenere coordinate cartesiane (x, y)

    # Converti le coordinate proiettate in metri
    meters = projected[:2] - t[:2]  # Sottrai solo le prime due componenti di t
    return meters


# Calcolare i punti in metri
meters_points = np.array([px2meters(pt, K, t) for pt in saved_points])

# Calcola R_H2W
def compute_rotation(points_h, points_w):
    if points_h.shape[0] != points_w.shape[0]:
        raise ValueError("Incompatible dimensions: points_h and points_w must have the same number of points.")

    # Calcola i centroidi
    centroid_h = np.mean(points_h[:, 1:3], axis=0)
    centroid_w = np.mean(points_w[:, 1:3], axis=0)

    # Centra i punti
    centered_h = points_h[:, 1:3] - centroid_h
    centered_w = points_w[:, 1:3] - centroid_w

    # Calcola la matrice di covarianza
    H = centered_h.T @ centered_w

    # Calcola la decomposizione SVD
    U, S, Vt = np.linalg.svd(H)

    # Calcola la matrice di rotazione
    R = Vt.T @ U.T

    # Assicurati che R sia una matrice di rotazione valida
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1  # Inverti l'orientamento se necessario
        R = Vt.T @ U.T

    return R


# Assicurati che la dimensione dei punti in metri e le posizioni siano compatibili
if meters_points.shape[0] == positions.shape[0]:
    R_H2W = compute_rotation(meters_points, positions)  # Usa y e z
    print("Matrice di Rotazione R_H2W:")
    print(R_H2W)
else:
    print("Errore: il numero di punti in meters_points e positions deve essere uguale.")
