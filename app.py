import streamlit as st
import mediapipe as mp
import cv2
import numpy as np

# Titre de l'application Streamlit
st.title("🖐️ UniSign - MVP")
st.write("نظام ترجمة بسيط للإشارات بلغة الإشارة")

# Initialisation de MediaPipe pour la détection des mains
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Case à cocher pour démarrer la caméra
run = st.checkbox('تشغيل الكاميرا')

# Espaces pour afficher les images et les labels
frame_placeholder = st.empty()
label_placeholder = st.empty()

# Capture vidéo de la caméra
cap = cv2.VideoCapture(0)

# Fonction pour compter le nombre de doigts levés
def count_fingers(hand_landmarks):
    finger_tips = [8, 12, 16, 20]  # Index des poignées pour les doigts
    count = 0
    for tip in finger_tips:
        # Vérifie si chaque doigt est levé en comparant la position des articulations
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1
    return count

# Boucle pour afficher la vidéo et détecter les gestes
while run:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Retourner l'image pour la rendre miroir (comme devant un miroir)
    frame = cv2.flip(frame, 1)
    
    # Convertir l'image en RGB pour MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Processus de détection des mains
    result = hands.process(rgb_frame)

    # Par défaut, afficher un message indiquant qu'aucune main n'est détectée
    label = "👋 لا يوجد يد"
    
    # Si une main est détectée, compter les doigts et associer un signe
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Compter le nombre de doigts levés
            finger_count = count_fingers(hand_landmarks)
            
            # Association simple de chaque nombre de doigts à un signe de langue des signes
            if finger_count == 0:
                label = "✊ إشارة: A"
            elif finger_count == 1:
                label = "☝️ إشارة: D"
            elif finger_count == 2:
                label = "✌️ إشارة: V"
            elif finger_count == 5:
                label = "🖐️ إشارة: Salut"
            else:
                label = f"🤟 عدد أصابع: {finger_count}"

    # Afficher la signification sur l'interface Streamlit
    label_placeholder.markdown(f"### {label}")
    
    # Convertir l'image pour l'afficher dans Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Afficher la vidéo en temps réel
    frame_placeholder.image(frame, channels="RGB")

# Libérer la capture vidéo une fois terminé
cap.release()







