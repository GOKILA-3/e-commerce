import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model('fashion_mnist_advanced_model.h5')

# Class names for Fashion MNIST
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def preprocess_image(img):
    # Convert to grayscale, resize to 28x28, normalize and reshape for model
    img = img.convert('L').resize((28, 28))
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

st.title("Virtual Try-On Snapshot App")

# Take a picture from the webcam
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # Open image and preprocess
    img = Image.open(img_file_buffer)
    st.image(img, caption="Input Image", use_column_width=True)

    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    pred_label = np.argmax(prediction)

    st.write(f"**Predicted Class:** {class_names[pred_label]}")
