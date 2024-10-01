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

def detect_a2_paper(image):
    # Convertiamo l'immagine nello spazio colore HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Mostriamo l'immagine HSV per controllo
    cv2.imshow("Immagine HSV", hsv)
    cv2.waitKey(0)

    # Definiamo l'intervallo per il colore verde (sfondo)
    lower_green = np.array([35, 40, 40])  # Prova a cambiare questi valori
    upper_green = np.array([85, 255, 255])

    # Creiamo una maschera per rilevare il verde
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Mostriamo la maschera verde per controllare se è corretta
    cv2.imshow("Maschera Verde", mask_green)
    cv2.waitKey(0)

    # Invertiamo la maschera per isolare il bianco (foglio A2)
    mask_white = cv2.bitwise_not(mask_green)

    # Troviamo i contorni del foglio bianco
    contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    a2_corners = None

    for contour in contours:
        # Approssimiamo il contorno a un poligono
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Se il contorno ha 4 lati ed è abbastanza grande, è probabilmente il foglio A2
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if 10000 < area < 100000:  # Aggiusta i limiti per l'area del foglio A2
                a2_corners = approx.reshape((4, 2))
                # Disegniamo il contorno del foglio sull'immagine
                cv2.drawContours(image, [approx], -1, (0, 0, 255), 3)

                # Disegniamo i vertici del foglio
                for corner in a2_corners:
                    cv2.circle(image, tuple(corner), 5, (255, 0, 0), -1)
                break

    return a2_corners


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

    # Rileviamo il foglio A2 bianco e otteniamo gli estremi
    a2_corners = detect_a2_paper(frame)

    if a2_corners is not None:
        print("Foglio A2 rilevato con i seguenti estremi:")
        for i, corner in enumerate(a2_corners):
            print(f"Estremo {i+1}: {corner}")
    else:
        print("Foglio A2 non rilevato.")

    # Salviamo l'immagine con i quadrati numerati e il foglio A2 riquadrato
    cv2.imwrite("kinect_squares_and_a2_detected.png", frame)
    print("Immagine con quadrati numerati e foglio A2 riquadrato salvata!")

    # Mostriamo l'immagine con i quadrati rilevati, il foglio A2 e il punto rosso
    cv2.imshow("Rilevamento Quadrati e Foglio A2", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Rilasciamo la Kinect
    kinect.stop()

if __name__ == "__main__":
    main()
