import streamlit as st
import mediapipe as mp
import cv2
import requests
import numpy as np

# Initialisation de Streamlit
st.title("🖐️ UniSign - MVP")
st.write("نظام ترجمة بسيط للإشارات بلغة الإشارة")

# Initialisation de Mediapipe pour la détection des mains
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

run = st.checkbox('تشغيل الكاميرا')

frame_placeholder = st.empty()
label_placeholder = st.empty()

cap = cv2.VideoCapture(0)

# Fonction pour envoyer l'image à DeepAI pour l'analyse
def analyze_image_with_deepai(frame):
    # Convertir l'image en format approprié pour DeepAI
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()

    # Clé API DeepAI (remplace avec la tienne)
    api_key = 'YOUR_DEEP_AI_API_KEY'

    # URL de l'API DeepAI
    api_url = 'https://api.deepai.org/api/image-recognition'

    headers = {
        'api-key': api_key
    }

    # Envoi de l'image à DeepAI
    response = requests.post(api_url, headers=headers, files={'image': img_bytes})

    if response.status_code == 200:
        result = response.json()
        return result
    else:
        return None

# Fonction pour interpréter les résultats de l'API
def interpret_result(result):
    if result:
        # Par exemple, si DeepAI renvoie des étiquettes de la reconnaissance d'image
        labels = result.get('output', {}).get('labels', [])
        if labels:
            return f"📜 Signification: {', '.join(labels)}"
        else:
            return "🤷‍♂️ Aucune signification trouvée"
    else:
        return "🤷‍♂️ Erreur lors de l'analyse"

# Détection et interprétation des gestes
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

            # Envoi de l'image pour analyse avec DeepAI
            deepai_result = analyze_image_with_deepai(frame)
            label = interpret_result(deepai_result)

    label_placeholder.markdown(f"### {label}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame, channels="RGB")

cap.release()





