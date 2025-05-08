# install necessary libraries:
# pip install streamlit mediapipe opencv-python numpy

import streamlit as st
import mediapipe as mp
import cv2
import numpy as np

# Title for the web app
st.title("🖐️ UniSign - MVP")
st.write("نظام ترجمة بسيط للإشارات بلغة الإشارة")

# MediaPipe hand detection setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Checkbox to toggle the camera on and off
run = st.checkbox('تشغيل الكاميرا')

# Placeholders for webcam feed and label
frame_placeholder = st.empty()
label_placeholder = st.empty()

# Set up webcam capture
cap = cv2.VideoCapture(0)

# Function to count the number of raised fingers
def count_fingers(hand_landmarks):
    finger_tips = [8, 12, 16, 20]  # Indices for finger tips
    count = 0
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1
    return count

# Main loop for running the webcam and gesture detection
while run:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe
    result = hands.process(rgb_frame)

    label = "👋 لا يوجد يد"  # Default label for no hand detected

    # If hands are detected, count the fingers
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            finger_count = count_fingers(hand_landmarks)

            # Map the number of raised fingers to a sign
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

    # Display the label on the Streamlit app
    label_placeholder.markdown(f"### {label}")
    
    # Convert the frame to RGB and display it on Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame, channels="RGB")

# Release the webcam when done
cap.release()








