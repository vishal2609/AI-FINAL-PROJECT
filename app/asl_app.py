# app/asl_app.py

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

from label_map import get_label_map

# Load trained CNN model
model = load_model("models/asl_cnn_model.keras")

# Get label mapping dictionary
label_map = get_label_map()

# Preprocessing function for each webcam frame
def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(gray, (28, 28))
    roi = roi.reshape(1, 28, 28, 1) / 255.0
    return roi

# Streamlit UI
st.title("🧠 ASL Alphabet Recognition App")
st.write("Place one hand sign inside the blue box. Press 'Start Webcam' to begin.")

run = st.checkbox('Start Webcam')
FRAME_WINDOW = st.image([])

# Open webcam
camera = cv2.VideoCapture(0)

while run:
    success, frame = camera.read()
    if not success:
        st.error("Camera not found or cannot open webcam.")
        break

    # Define region of interest (ROI) box
    x, y, w, h = 200, 150, 200, 200
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi = frame[y:y+h, x:x+w]
    processed = preprocess(roi)

    # Predict ASL letter
    prediction = model.predict(processed)
    confidence = np.max(prediction)
    class_index = np.argmax(prediction)

    # Only show prediction if confidence is high
    if confidence > 0.80:
        letter = label_map[class_index]
    else:
        letter = "No Sign"

    # Optional: print the confidence on screen
    cv2.putText(frame, f'Confidence: {confidence:.2f}', (x, y + h + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


    # Show prediction on frame
    cv2.putText(frame, f'Prediction: {letter}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert BGR to RGB for Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

# Release camera
camera.release()
