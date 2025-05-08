import cv2
import mediapipe as mp
import numpy as np

# Initialiser MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Définir la fonction pour compter les doigts levés
def count_fingers(hand_landmarks):
    # Liste des indices des points de repère pour chaque doigt (tips des doigts)
    finger_tips = [8, 12, 16, 20]  # 8: Index, 12: Middle, 16: Ring, 20: Pinky
    count = 0
    for tip in finger_tips:
        # Si le point de repère est plus haut que le point précédant dans le sens Y, alors le doigt est levé
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1
    return count

# Capture vidéo depuis la caméra
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inverser l'image pour la vision miroir
    frame = cv2.flip(frame, 1)

    # Convertir l'image de BGR (OpenCV) en RGB (MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processus de la détection des mains avec MediaPipe
    result = hands.process(rgb_frame)

    # Vérifier si des mains ont été détectées
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Dessiner les points de repère de la main
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Compter les doigts levés
            finger_count = count_fingers(hand_landmarks)

            # Affichage des résultats sur l'écran
            if finger_count == 0:
                label = "✊ Signification: A"
            elif finger_count == 1:
                label = "☝️ Signification: D"
            elif finger_count == 2:
                label = "✌️ Signification: V"
            elif finger_count == 5:
                label = "🖐️ Signification: Salut"
            else:
                label = f"🤟 Nombre de doigts levés: {finger_count}"

            # Afficher le label sur l'image
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Afficher l'image avec les résultats
    cv2.imshow("Main Détectée", frame)

    # Quitter si l'utilisateur appuie sur la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()






