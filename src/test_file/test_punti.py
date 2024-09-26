import cv2
import numpy as np
import utils  # Assicurati che la tua classe Kinect sia nel file utils.py o cambi il percorso

def detect_black_squares(image):
    # Convertiamo l'immagine in scala di grigi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Soglia per binarizzare l'immagine (invertiamo i colori per rilevare i quadrati neri)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Troviamo i contorni nell'immagine binarizzata
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    squares_centers = []

    # Loop su ogni contorno per identificare quadrati
    for contour in contours:
        # Approssimiamo il contorno a un poligono
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Se il poligono ha 4 lati ed è convesso, possiamo considerarlo un quadrato
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > 500:  # Consideriamo solo quadrati di una certa dimensione
                # Calcoliamo il centro del quadrato
                M = cv2.moments(approx)
                if M['m00'] != 0:
                    cX = int(M['m10'] / M['m00'])
                    cY = int(M['m01'] / M['m00'])
                    
                    # Verifica se l'interno del quadrato è nero
                    mask = np.zeros_like(gray)
                    cv2.drawContours(mask, [approx], -1, 255, -1)
                    mean_val = cv2.mean(gray, mask=mask)[0]
                    if mean_val < 50:  # Se il valore medio è basso (nero)
                        squares_centers.append((cX, cY))
                        # Disegniamo il contorno del quadrato sull'immagine
                        cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
                        # Disegniamo il centro del quadrato
                        cv2.circle(image, (cX, cY), 5, (255, 0, 0), -1)

    return squares_centers

def main():
    # Inizializziamo la Kinect con il solo RGB abilitato
    kinect = utils.Kinect(enable_rgb=True, enable_depth=False, need_bigdepth=False, need_color_depth_map=False)
    
    print("Premi INVIO per scattare una foto con la Kinect")

    # Punto da mostrare in rosso
    red_point = (int(1.44846084e+03), int(2.57053426e+02))

    # Acquisiamo un'immagine RGB dalla Kinect
    while True:
        kinect.acquire(correct=False)
        frame = kinect.color_new

        # Verifica che l'immagine sia correttamente formattata per OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convertiamo l'immagine da RGB a BGR (se necessario)

        # Disegniamo il punto in rosso
        cv2.circle(frame, red_point, 5, (0, 0, 255), -1)

        cv2.imshow('Kinect Camera - Press Enter to Capture', frame)

        if cv2.waitKey(1) & 0xFF == 13:  # INVIO
            cv2.imwrite("kinect_image.png", frame)
            print("Immagine salvata!")
            break

    # Rileviamo i quadrati neri e otteniamo i centri
    squares_centers = detect_black_squares(frame)

    # Stampiamo e numeriamo i centri dei quadrati trovati
    if squares_centers:
        for i, center in enumerate(squares_centers):
            print(f"Quadrato {i+1}: Centro a {center}")
            # Aggiungiamo il numero del quadrato sull'immagine
            cv2.putText(frame, f"{i+1}", (center[0] - 20, center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    else:
        print("Nessun quadrato nero rilevato.")

    # Salviamo l'immagine con i quadrati numerati
    cv2.imwrite("kinect_squares_detected.png", frame)
    print("Immagine con quadrati numerati salvata!")

    # Mostriamo l'immagine con i quadrati rilevati e il punto rosso
    cv2.imshow("Quadrati Rilevati", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Rilasciamo la Kinect
    kinect.stop()

if __name__ == "__main__":
    main()
