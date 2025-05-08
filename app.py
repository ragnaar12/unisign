import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

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
    finger_tips = [8, 12, 16, 20]  # Index des points des doigts
    count = 0
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:  # Vérification si le doigt est levé
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
