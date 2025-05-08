import streamlit as st
import mediapipe as mp
import cv2
import numpy as np

st.title("🖐️ UniSign - MVP")
st.write("نظام ترجمة بسيط للإشارات بلغة الإشارة")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

run = st.checkbox('تشغيل الكاميرا')

frame_placeholder = st.empty()
label_placeholder = st.empty()

cap = cv2.VideoCapture(0)

# Fonction pour détecter si la main est présente et donner des résultats
def count_fingers(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    count = 0
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1
    return count

# Fonction pour déterminer les signes
def detect_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]

    # Si la main n'est pas complètement ouverte ou bien positionnée, ne pas afficher "Salut"
    if index_tip.y < middle_tip.y < ring_tip.y < pinky_tip.y and thumb_tip.x < index_tip.x:
        return "👋 إشارة: Salut"
    elif index_tip.y < middle_tip.y and ring_tip.y > pinky_tip.y and thumb_tip.x < index_tip.x:
        return "✌️ إشارة: V"
    elif index_tip.y > middle_tip.y > ring_tip.y > pinky_tip.y:
        return "☝️ إشارة: D"
    elif thumb_tip.y < index_tip.y and middle_tip.y < index_tip.y and ring_tip.y < index_tip.y and pinky_tip.y < index_tip.y:
        return "✊ إشارة: A"
    else:
        return None  # Pas de signe reconnu

while run:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    label = "👋 لا يوجد يد"  # Message par défaut si aucune main n'est détectée

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = detect_gesture(hand_landmarks)
            
            # Si aucun geste n'est reconnu, retourner un message d'erreur
            if gesture is None:
                label = "🤷‍♂️ إشارة غير معروفة"
            else:
                label = gesture

    label_placeholder.markdown(f"### {label}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame, channels="RGB")

cap.release()




