import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import pandas as pd
import altair as alt

# Load the trained model
model = tf.keras.models.load_model("fashion_mnist_advanced_model.h5")

# Class labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# --- UI Design ---
st.set_page_config(page_title="Fashion Category Predictor", layout="wide", page_icon="🛍️")

# Custom CSS for styling
st.markdown("""
<style>
    .title {
        font-size: 56px;
        font-weight: 800;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 24px;
        color: #1E90FF;
        text-align: center;
        margin-top: 0;
        margin-bottom: 30px;
        font-weight: 600;
    }
    .confidence-bar {
        background: linear-gradient(90deg, #FF4B4B, #FFAA00);
        height: 24px;
        border-radius: 12px;
    }
    .confidence-label {
        font-weight: 700;
        color: #444444;
        margin-top: 5px;
        margin-bottom: 5px;
    }
    .stProgress > div > div > div > div {
        background-color: #FF4B4B;
    }
    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/892/892458.png", width=100)
st.sidebar.title("🛒 App Info")
st.sidebar.markdown("""
This AI-powered tool classifies fashion items into 10 categories from the Fashion MNIST dataset using a deep learning model.

### 💡 Tips:
- Use clear product images.
- Images will be resized to 28x28 pixels.
- Works best for front-view clothing photos.
""")

st.markdown('<h1 class="title">🛍️ AI-Powered E-Commerce Personalization</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="subtitle">👗 Fashion Category Predictor</h2>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📷 Upload a clothing image (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

def display_confidence_bar(confidence, label):
    st.markdown(f'<div class="confidence-label">{label}</div>', unsafe_allow_html=True)
    st.progress(confidence)

if uploaded_file:
    with st.spinner("🔍 Analyzing your image..."):
        image = Image.open(uploaded_file).convert("L")

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

    # Show input image and prediction side by side
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image_resized, caption="🖼️ Processed Image (28x28)", width=180, use_column_width=False)

    with col2:
        st.success(f"✅ Predicted Category: **{predicted_class}**")
        st.markdown(f"### Confidence: <span style='color:#FF4B4B'>{confidence:.2%}</span>", unsafe_allow_html=True)
        if confidence < 0.7:
            st.warning("⚠️ Confidence is low. Try a clearer or closer image.")
        elif confidence > 0.9:
            st.balloons()

        # Animated confidence bar
        display_confidence_bar(confidence, f"Confidence in {predicted_class}")

    # Show confidence scores for all classes using Altair chart
    st.markdown("### 📊 Confidence Scores for All Categories")

    conf_df = pd.DataFrame({
        "Category": class_names,
        "Confidence": predictions[0]
    })

    chart = alt.Chart(conf_df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X('Confidence:Q', axis=alt.Axis(format='.0%')),
        y=alt.Y('Category:N', sort='-x'),
        color=alt.Color('Confidence:Q', scale=alt.Scale(scheme='redyellowgreen'), legend=None),
        tooltip=[alt.Tooltip('Category:N'), alt.Tooltip('Confidence:Q', format='.1%')]
    ).properties(height=300, width=700)

    st.altair_chart(chart, use_container_width=True)

    # Option to show raw prediction scores
    with st.expander("🔍 Show raw prediction scores"):
        st.write(predictions)

else:
    st.info("👆 Upload an image to begin fashion category prediction.")
