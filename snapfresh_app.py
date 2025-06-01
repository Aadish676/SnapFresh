import streamlit as st
import os
import tempfile
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from streamlit.components.v1 import html

# Setup temp directory
DATASET_DIR = os.path.join(tempfile.gettempdir(), "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

# Custom styling
st.set_page_config(page_title="SnapFresh", layout="centered")
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    .stImage > img {
        border-radius: 1em;
        margin: auto;
        max-width: 300px;
        display: block;
    }
    .step-title {
        font-size: 1.25em;
        font-weight: bold;
        margin-top: 1.5em;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🍏 SnapFresh: AI Food Freshness Detector</div>", unsafe_allow_html=True)

# Preprocess function
def preprocess_image(image_path):
    image = Image.open(image_path).resize((64, 64))
    return np.array(image).flatten() / 255.0

# Download images
def download_images(query, label, count=50):
    url = f"https://source.unsplash.com/640x480/?{query}"
    for i in range(count):
        response = requests.get(url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img.save(os.path.join(DATASET_DIR, f"{label}_{i}.jpg"))

# Step 1: Input food item
st.markdown("<div class='step-title'>Step 1: Enter a food item</div>", unsafe_allow_html=True)
food_item = st.text_input("e.g., banana, apple, tomato")

# Step 2: Download images
if st.button("🔽 Download Images"):
    if not food_item:
        st.warning("Please enter a food item before downloading.")
    else:
        st.info("Downloading images... Please wait (~1 minute)")
        stages = ['rotten', 'moderate', 'good']
        for stage in stages:
            download_images(f"{food_item} {stage}", stage)
        st.success("✅ Images downloaded successfully!")

# Step 3: Train model and predict
if st.button("🧠 Train Model & Predict"):
    st.info("Training model with downloaded images...")
    X, y = [], []

    for filename in os.listdir(DATASET_DIR):
        filepath = os.path.join(DATASET_DIR, filename)
        try:
            label = filename.split("_")[0]
            features = preprocess_image(filepath)
            X.append(features)
            y.append(label)
        except:
            continue

    if len(X) < 10:
        st.warning("Not enough images to train. Try downloading again.")
    else:
        X = np.array(X)
        y = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train, y_train)

        st.success("✅ Model trained successfully!")

        st.markdown("<div class='step-title'>Step 4: Upload an image to predict freshness</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            img = Image.open(uploaded_file).resize((64, 64))
            st.image(img, caption="Uploaded Image", width=250)

            img_array = np.array(img).flatten().reshape(1, -1) / 255.0
            prediction = model.predict(img_array)

            st.markdown(f"""
                <h3 style='text-align: center;'>🌟 Predicted Freshness: <span style='color: green;'>{prediction[0].capitalize()}</span></h3>
            """, unsafe_allow_html=True)


