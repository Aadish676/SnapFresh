import streamlit as st
import os
import tempfile
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Use temp directory to avoid permission issues
DATASET_DIR = os.path.join(tempfile.gettempdir(), "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

# Preprocess function
def preprocess_image(image_path):
    image = Image.open(image_path).resize((64, 64))
    return np.array(image).flatten() / 255.0

# Function to fetch and save images
def download_images(query, label, count=50):
    url = f"https://source.unsplash.com/640x480/?{query}"
    for i in range(count):
        response = requests.get(url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img.save(os.path.join(DATASET_DIR, f"{label}_{i}.jpg"))

# UI: Select food and trigger download
st.title("SnapFresh: Food Freshness Predictor")

food_item = st.text_input("Enter a food item (e.g. banana, apple, tomato):")
if st.button("Download Images"):
    st.info("Downloading images... Please wait (~1 minute)")
    stages = ['rotten', 'moderate', 'good']
    for stage in stages:
        download_images(f"{food_item} {stage}", stage)
    st.success("Images downloaded successfully!")

# Training phase
if st.button("Train Model and Predict"):
    st.info("Processing images and training model...")
    X = []
    y = []

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

        st.success("Model trained successfully!")

        uploaded_file = st.file_uploader("Upload an image of the food item:")
        if uploaded_file is not None:
            img = Image.open(uploaded_file).resize((64, 64))
            st.image(img, caption="Uploaded Image", use_column_width=True)
            img_array = np.array(img).flatten().reshape(1, -1) / 255.0
            prediction = model.predict(img_array)
            st.subheader(f"🧠 Predicted Freshness: {prediction[0].capitalize()}")

