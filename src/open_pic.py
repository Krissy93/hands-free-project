import cv2

def open_image():

    # Carica l'immagine usando OpenCV
    img = cv2.imread('/home/jacopo/URProject/src/hands-free-project/src/calib_img/master0.png')

    # Controlla se l'immagine è stata caricata correttamente
    if img is None:
        print("Errore: Immagine non trovata o percorso non valido.")
        return

    # Mostra l'immagine
    cv2.imshow('Immagine Aperta', img)

    # Aspetta fino a che l'utente prema un tasto qualsiasi
    cv2.waitKey(0)

    # Chiude tutte le finestre create da OpenCV
    cv2.destroyAllWindows()

if __name__ == "__main__":
    open_image()
