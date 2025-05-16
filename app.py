import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import pandas as pd
import os

# Load trained model
model = tf.keras.models.load_model("fashion_mnist_advanced_model.h5")

# Class labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# UI setup
st.set_page_config(page_title="Fashion Predictor with Try-On", layout="wide")
st.markdown(
    """
    <style>
    .title { font-size:48px; color:#FF4B4B; font-weight:bold; }
    .sub { font-size:24px; color:#1E90FF; }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/892/892458.png", width=100)
st.sidebar.title("🛒 App Info")
st.sidebar.write("""
This AI tool classifies clothing items and overlays try-on items like:
- Fashionwear (T-shirt, dress, coat, etc.)
- 👓 Specs
- 🧢 Caps

Add transparent PNGs into `tryon_assets/`.
""")

st.markdown('<div class="title">🛍️ AI E-Commerce Smart Try-On</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">👗 Fashion Prediction + Virtual Try-On (Clothes + Specs + Cap)</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📷 Upload a clothing image (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

# Add checkboxes for specs and cap
add_specs = st.checkbox("👓 Add Specs")
add_cap = st.checkbox("🧢 Add Cap")

if uploaded_file:
    with st.spinner("Processing Image..."):
        image = Image.open(uploaded_file).convert("L")

        # Resize image with compatible method
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

        # Show prediction
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image_resized, caption="🖼️ Processed Image", width=150)
        with col2:
            st.success(f"✅ Predicted Category: **{predicted_class}**")
            st.write(f"🎯 Confidence: **{confidence:.2%}**")
            if confidence < 0.7:
                st.warning("⚠️ Low confidence. Use a better image.")
            elif confidence > 0.9:
                st.balloons()

        # Confidence chart
        st.markdown("### 📊 Confidence Scores for All Categories")
        conf_df = pd.DataFrame({
            "Category": class_names,
            "Confidence": predictions[0]
        })
        st.bar_chart(conf_df.set_index("Category"))

        # Try-on section
        st.markdown("### 🧍 Virtual Try-On Preview")

        # Compose try-on items
        tryon_items = []

        # Main fashion item
        fashion_filename = predicted_class.lower().replace(" ", "").replace("-", "") + ".png"
        fashion_path = os.path.join("tryon_assets", fashion_filename)
        if os.path.exists(fashion_path):
            tryon_items.append(("🧥 Clothing", fashion_path))
        else:
            st.warning(f"🔍 No try-on overlay for: {predicted_class}")

        # Optional accessories
        if add_specs:
            specs_path = os.path.join("tryon_assets", "specs.png")
            if os.path.exists(specs_path):
                tryon_items.append(("👓 Specs", specs_path))
            else:
                st.warning("❌ Specs image not found.")

        if add_cap:
            cap_path = os.path.join("tryon_assets", "cap.png")
            if os.path.exists(cap_path):
                tryon_items.append(("🧢 Cap", cap_path))
            else:
                st.warning("❌ Cap image not found.")

        # Display overlays
        for label, path in tryon_items:
            image_overlay = Image.open(path).resize((150, 150))
            st.image(image_overlay, caption=label, width=150)
else:
    st.info("👆 Upload an image to begin fashion category prediction.")
