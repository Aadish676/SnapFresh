import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model (assumes you have a trained model at this path)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/snapfresh_model.h5")
    return model

model = load_model()

# Define class labels (example labels)
class_labels = ['Fresh', 'Moderate', 'Spoiled']

st.set_page_config(page_title="SnapFresh - Food Freshness Predictor", layout="centered")
st.title("SnapFresh \U0001F957")
st.subheader("Predict food freshness from an image")

uploaded_file = st.file_uploader("Upload an image of the food item", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image_resized = image.resize((224, 224))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)[0]
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"**Prediction:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    if predicted_class == 'Spoiled':
        st.warning("This item may not be safe to eat. Dispose responsibly.")
    elif predicted_class == 'Moderate':
        st.warning("Consume soon. May be nearing spoilage.")
    else:
        st.success("Looks fresh and good to consume!")

st.markdown("---")
st.caption("SnapFresh © 2025 | AI-powered food freshness detection")
