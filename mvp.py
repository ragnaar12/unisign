# install necessary libraries:
# pip install mediapipe opencv-python streamlit

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

# نموذج بسيط جدا: التعرف على عدد الأصابع المرفوعة فقط
def count_fingers(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    count = 0
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1
    return count

while run:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    label = "👋 لا يوجد يد"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            finger_count = count_fingers(hand_landmarks)
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

    label_placeholder.markdown(f"### {label}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame, channels="RGB")

cap.release()
