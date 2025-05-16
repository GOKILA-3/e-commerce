import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import pandas as pd

# Load the trained model
model = tf.keras.models.load_model("fashion_mnist_advanced_model.h5")

# Class labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# UI Design
st.set_page_config(page_title="Fashion Category Predictor", layout="wide")
st.markdown(
    """
    <style>
    .title {
        font-size:48px;
        color:#FF4B4B;
        font-weight:bold;
        text-align: center;
        margin-bottom: 0;
    }
    .sub {
        font-size:24px;
        color:#1E90FF;
        text-align: center;
        margin-top: 0;
        margin-bottom: 20px;
    }
    .confidence-label {
        font-weight: bold;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/892/892458.png", width=100)
st.sidebar.title("🛒 App Info")
st.sidebar.write("""
This AI-powered tool classifies fashion items into 10 categories from the Fashion MNIST dataset using a trained deep learning model.

### 💡 Tips:
- Use 28x28 grayscale images for best accuracy.
- Clothing images will be resized automatically.
- Works well for top-view product photos.
""")

st.markdown('<div class="title">🛍️ AI-Powered E-Commerce Personalization</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">👗 Fashion Category Predictor</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📷 Upload a clothing image (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

def display_confidence_bar(confidence, label):
    st.markdown(f'<div class="confidence-label">{label}</div>', unsafe_allow_html=True)
    st.progress(int(confidence * 100))

if uploaded_file:
    with st.spinner("Processing Image..."):
        image = Image.open(uploaded_file).convert("L")

        # Fix compatibility for resizing
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.ANTIALIAS

        image_resized = ImageOps.fit(image, (28, 28), method=resample)
        img_array = np.array(image_resized).reshape(1, 28, 28, 1).astype(np.float32) / 255.0

        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        predicted_class = class_names[predicted_index]
        confidence = predictions[0][predicted_index]

        # Show results
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image_resized, caption="🖼️ Processed Image", width=150)
        with col2:
            st.success(f"✅ Predicted Category: **{predicted_class}**")
            st.write(f"🎯 Confidence: **{confidence:.2%}**")
            if confidence < 0.7:
                st.warning("⚠️ Confidence is low. Try a clearer or closer image.")
            elif confidence > 0.9:
                st.balloons()

            display_confidence_bar(confidence, f"Confidence in {predicted_class}")

        # Confidence Scores Bar Chart
        st.markdown("### 📊 Confidence Scores for All Categories")
        conf_df = pd.DataFrame({
            "Category": class_names,
            "Confidence": predictions[0]
        })
        st.bar_chart(conf_df.set_index("Category"))
else:
    st.info("👆 Upload an image to begin fashion category prediction.")
