# app/asl_app.py

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sys
import os
import time
from collections import deque

# Add path to scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from label_map import get_label_map

# Load model with caching
@st.cache_resource
def load_asl_model():
    return load_model('models/sign_mnist_cnn_best.keras', compile=False)

model = load_asl_model()
label_map = get_label_map()

# Streamlit UI
st.title("🤟 Real-time ASL Recognition")
st.markdown("Place one hand inside the box and press **Start Webcam** to begin.")

# Sidebar controls
st.sidebar.title("🔧 Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.5, 1.0, 0.8, 0.01)
roi_size = st.sidebar.slider("ROI Box Size", 100, 400, 300, step=50)
show_gray = st.sidebar.checkbox("Show Grayscale ROI")
st.sidebar.markdown("### 🧠 Live Prediction")

# Webcam controls
run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])
roi_preview = st.sidebar.empty()

# Prediction smoothing
recent_preds = deque(maxlen=5)

# Start webcam
camera = cv2.VideoCapture(0)
prev_time = time.time()

while run:
    success, frame = camera.read()
    if not success:
        st.error("⚠️ Cannot access webcam.")
        break

    # Define ROI
    h, w = frame.shape[:2]
    x = (w - roi_size) // 2
    y = (h - roi_size) // 2
    cv2.rectangle(frame, (x, y), (x + roi_size, y + roi_size), (0, 255, 0), 2)
    roi = frame[y:y+roi_size, x:x+roi_size]

    # Preprocess ROI
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized / 255.0
    processed = normalized.reshape(1, 28, 28, 1)

    # Optional grayscale preview
    if show_gray:
        roi_preview.image(resized, caption="Grayscale ROI", width=200, channels="GRAY")

    # Predict
    prediction = model.predict(processed, verbose=0)
    confidence = np.max(prediction)
    class_index = np.argmax(prediction)

    # Smooth prediction
    if confidence > confidence_threshold:
        recent_preds.append(label_map[class_index])
        letter = max(set(recent_preds), key=recent_preds.count)
    else:
        letter = "No Sign"

    # FPS counter
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Annotate frame
    cv2.putText(frame, f'Prediction: {letter}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, f'Confidence: {confidence:.2f}', (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Convert and display frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame_rgb)

# Release camera
camera.release()
