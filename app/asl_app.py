import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from label_map import get_label_map

# Load model
model = load_model("models/asl_cnn_model_upd.keras")  # or your new model
label_map = get_label_map()

# Preprocessing function
def preprocess(frame):
    roi = cv2.resize(frame, (64, 64))
    roi = roi.reshape(1, 64, 64, 3) / 255.0
    return roi

# Streamlit UI
st.title("🤟 ASL Alphabet Recognition (Fixed ROI)")
run = st.checkbox('Start Webcam')
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    success, frame = camera.read()
    if not success:
        st.error("Camera not found.")
        break

    frame = cv2.flip(frame, 1)  # Mirror image

    h, w, _ = frame.shape

    # Define fixed bounding box
    box_size = 200
    center_x, center_y = w // 2, h // 2
    x_min = center_x - box_size // 2
    y_min = center_y - box_size // 2
    x_max = center_x + box_size // 2
    y_max = center_y + box_size // 2

    # Draw fixed rectangle
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    # Crop and preprocess ROI
    roi = frame[y_min:y_max, x_min:x_max]
    processed = preprocess(roi)

    # Predict
    prediction = model.predict(processed)
    confidence = np.max(prediction)
    class_index = np.argmax(prediction)

    if confidence > 0.50:
        letter = label_map[class_index]
        cv2.putText(frame, f'{letter} ({confidence:.2f})', (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, f'No Sign ({confidence:.2f})', (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

camera.release()
