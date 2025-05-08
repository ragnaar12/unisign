import streamlit as st
import mediapipe as mp
import cv2
import numpy as np

# Initialiser les objets de Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

st.title("🖐️ UniSign - MVP")
st.write("نظام ترجمة بسيط للإشارات بلغة الإشارة")

# Téléchargement de l'image ou de la vidéo
video_file = st.file_uploader("Télécharger une vidéo", type=["mp4", "avi", "mov"])

# Fonction pour compter les doigts levés
def count_fingers(hand_landmarks):
    # L'index des points de chaque doigt, en fonction de l'index de Mediapipe
    # 0: base de la paume, 4: petit doigt, 8: index, etc.
    # On vérifie si les doigts sont au-dessus de la paume (doigt levé)
    
    # Doigts à vérifier : [1, 2, 3, 4] (généralement les bouts des doigts)
    finger_tips = [8, 12, 16, 20]
    count = 0
    
    for tip in finger_tips:
        # Si la position du bout du doigt est plus haute que le bas du doigt, il est levé
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1

    return count

# Traitement de la vidéo téléchargée
if video_file:
    video_bytes = video_file.read()
    video_array = np.asarray(bytearray(video_bytes), dtype=np.uint8)
    video_cap = cv2.VideoCapture(video_array)
    
    ret, frame = video_cap.read()

    if ret:
        # Conversion de l'image de couleur BGR à RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Détection des mains
        result = hands.process(rgb_frame)
        
        label = "👋 Aucune main détectée"
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Compte des doigts levés
                finger_count = count_fingers(hand_landmarks)
                
                # Logique des gestes
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

        # Retourner les données au frontend
        st.write(label)
        
        # Affichage de la vidéo traitée
        st.image(frame, channels="RGB")

    video_cap.release()

