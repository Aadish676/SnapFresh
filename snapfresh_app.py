# SnapFresh: Accurate Food Freshness Detection using MobileNetV2

import streamlit as st
from PIL import Image
import numpy as np
import os
import shutil
import requests
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras import layers, models
import datetime
import random

# Set page config
st.set_page_config(page_title="SnapFresh", layout="centered")

# Step 1: Prepare directories
DATA_DIR = "snapfresh_data"
MODEL_PATH = "snapfresh_model.h5"
CATEGORIES = ["fresh", "moderate", "rotten"]

os.makedirs(DATA_DIR, exist_ok=True)
for cat in CATEGORIES:
    os.makedirs(os.path.join(DATA_DIR, cat), exist_ok=True)

# Step 2: UI - Header
st.markdown("<h1 style='text-align: center;'>\U0001F4F1 SnapFresh</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>AI-powered Food Freshness Detection</h4>", unsafe_allow_html=True)
st.divider()

# Step 3: Train Section
with st.expander("\U0001F52C Train Model (One-Time)", expanded=True):
    food_item = st.text_input("Enter a food item (e.g., apple, tomato)", value="apple")

    def download_sample_images(food):
        search_urls = {
            "fresh": [f"https://source.unsplash.com/224x224/?{food},fresh" for _ in range(10)],
            "moderate": [f"https://source.unsplash.com/224x224/?{food},aging" for _ in range(10)],
            "rotten": [f"https://source.unsplash.com/224x224/?{food},rotten" for _ in range(10)],
        }
        for cat in CATEGORIES:
            for i, url in enumerate(search_urls[cat]):
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        img = Image.open(BytesIO(response.content)).convert("RGB")
                        img.save(os.path.join(DATA_DIR, cat, f"{cat}_{i}.jpg"))
                except Exception as e:
                    continue

    if st.button("Download & Train Model"):
        with st.spinner("Downloading images and training model..."):
            download_sample_images(food_item)
            datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.2,
                horizontal_flip=True,
                validation_split=0.2
            )

            train_gen = datagen.flow_from_directory(DATA_DIR, target_size=(224, 224), class_mode='categorical', subset='training')
            val_gen = datagen.flow_from_directory(DATA_DIR, target_size=(224, 224), class_mode='categorical', subset='validation')

            base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
            base_model.trainable = False
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(3, activation='softmax')
            ])

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(train_gen, validation_data=val_gen, epochs=5, verbose=1)
            model.save(MODEL_PATH)
        st.success("Model trained and saved!")

# Step 4: Prediction Section
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)

    uploaded_file = st.file_uploader("Upload a food image for prediction", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        resized = img.resize((224, 224))
        img_array = np.array(resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]
        label_idx = np.argmax(prediction)
        label = CATEGORIES[label_idx]

        spoilage_days_map = {"fresh": 5, "moderate": 2, "rotten": 0}
        spoilage_date = datetime.datetime.now() + datetime.timedelta(days=spoilage_days_map[label])

        st.image(img, caption="Uploaded Image", width=300)
        st.markdown(f"### Prediction: **{label.upper()}**")
        st.markdown(f"**Estimated Spoilage Date:** {spoilage_date.strftime('%Y-%m-%d')}")

# Step 5: Footer
st.divider()
st.caption("SnapFresh • 2025 Prototype")
