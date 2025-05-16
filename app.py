import cv2
import streamlit as st
import numpy as np
import tensorflow as tf

# Load your trained model (adjust path as needed)
model = tf.keras.models.load_model('fashion_mnist_advanced_model.h5')

# Function to preprocess frame for model prediction
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)
    return reshaped

# Overlay example function (for specs, caps, shirts)
def overlay_image(background, overlay, x, y):
    h, w, _ = overlay.shape
    background[y:y+h, x:x+w] = cv2.addWeighted(background[y:y+h, x:x+w], 0.5, overlay, 0.5, 0)
    return background

st.title("Virtual Try-On App")

run = st.checkbox('Start Webcam')

if run:
    cap = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame")
            break

        # Preprocess frame and predict (dummy example)
        processed = preprocess_frame(frame)
        prediction = model.predict(processed)
        pred_label = np.argmax(prediction)

        # Example: overlay a transparent image (you need to load your overlays properly)
        # frame = overlay_image(frame, specs_img, 100, 50)  # sample coordinates

        FRAME_WINDOW.image(frame, channels="BGR")

        # To break loop (optional)
        if not run:
            break

    cap.release()

# Note: no cv2.destroyAllWindows() or cv2.waitKey() calls here
