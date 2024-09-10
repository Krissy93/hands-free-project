import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Carica la webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands() as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Non riesco a catturare l'immagine.")
            continue

        # Conversione in RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Rileva mani
        result = hands.process(image_rgb)

        # Se ci sono mani rilevate, disegna i landmarks
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Mostra l'immagine
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
