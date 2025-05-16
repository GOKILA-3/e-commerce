import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load your trained model
model = tf.keras.models.load_model('fashion_mnist_advanced_model.h5')

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Load overlay images with transparency (PNG) for specs, cap, tshirt
overlay_specs = Image.open("overlays/specs.png").convert("RGBA")
overlay_cap = Image.open("overlays/cap.png").convert("RGBA")
overlay_tshirt = Image.open("overlays/tshirt.png").convert("RGBA")

def preprocess_image(img):
    img = img.convert('L').resize((28, 28))
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

def overlay_image(base_img, overlay_img, position=(0,0), scale=1.0):
    # Resize overlay according to scale
    w, h = overlay_img.size
    overlay_img = overlay_img.resize((int(w*scale), int(h*scale)), Image.ANTIALIAS)

    base_img = base_img.convert("RGBA")
    base_img.paste(overlay_img, position, overlay_img)
    return base_img

st.title("Virtual Try-On Snapshot App")

img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    img = Image.open(img_file_buffer)
    st.image(img, caption="Input Image", use_column_width=True)

    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    pred_label = np.argmax(prediction)
    st.write(f"**Predicted Class:** {class_names[pred_label]}")

    # Overlay logic depending on predicted class
    # For simplicity, overlay cap if predicted is "T-shirt/top" or "Pullover"
    # overlay specs if "Shirt", overlay tshirt if "T-shirt/top"
    img_overlay = img.copy()

    if class_names[pred_label] in ["T-shirt/top", "Pullover"]:
        img_overlay = overlay_image(img_overlay, overlay_cap, position=(50, 10), scale=0.3)
        img_overlay = overlay_image(img_overlay, overlay_tshirt, position=(20, 150), scale=0.5)
    elif class_names[pred_label] == "Shirt":
        img_overlay = overlay_image(img_overlay, overlay_specs, position=(100, 80), scale=0.3)

    st.image(img_overlay, caption="Virtual Try-On Result", use_column_width=True)
