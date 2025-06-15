import streamlit as st
from PIL import Image
import random

# Title
st.title("SnapFresh - Food Freshness Predictor")

# Description
st.markdown("""
SnapFresh helps you predict the freshness and expected spoilage date of food based on uploaded images.
This version uses a mock algorithm for demonstration purposes.
""")

# Image Upload
uploaded_file = st.file_uploader("Upload an image of food", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Simulate prediction
    freshness_levels = ["Fresh", "Stale", "Almost Spoiled", "Spoiled"]
    predicted_freshness = random.choice(freshness_levels)
    spoilage_days = {
        "Fresh": random.randint(5, 7),
        "Stale": random.randint(3, 4),
        "Almost Spoiled": random.randint(1, 2),
        "Spoiled": 0
    }

    # Display Results
    st.subheader("Predicted Freshness Level:")
    st.success(predicted_freshness)

    st.subheader("Estimated Days Until Spoilage:")
    st.info(f"{spoilage_days[predicted_freshness]} day(s)")
