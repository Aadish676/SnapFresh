import streamlit as st
import os
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import base64
import random

# Create a directory to store images
os.makedirs("dataset", exist_ok=True)

def fetch_images(food_name, label, count=15):
    search_url = f"https://www.google.com/search?q={food_name}+{label}+site:unsplash.com&tbm=isch"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(search_url, headers=headers)
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        img_tags = soup.find_all("img")

        downloaded = 0
        for img_tag in img_tags:
            img_url = img_tag.get("src")
            if img_url and img_url.startswith("http"):
                try:
                    img_data = requests.get(img_url).content
                    img = Image.open(BytesIO(img_data)).convert("RGB")
                    img = img.resize((64, 64))
                    img.save(f"dataset/{label}_{downloaded}.jpg")
                    downloaded += 1
                    if downloaded >= count:
                        break
                except Exception:
                    continue
    except Exception as e:
        st.error(f"Image fetch error: {e}")

def load_data():
    X, y = [], []
    for file in os.listdir("dataset"):
        if file.endswith(".jpg"):
            label = file.split("_")[0]
            path = os.path.join("dataset", file)
            img = Image.open(path).resize((64, 64))
            X.append(np.array(img).flatten())
            y.append(label)
    return np.array(X), np.array(y)

def spoilage_date_prediction(label):
    days_map = {
        "fresh": random.randint(4, 7),
        "moderate": random.randint(2, 4),
        "rotten": random.randint(0, 1)
    }
    return days_map.get(label, 3)

st.set_page_config(page_title="SnapFresh", layout="centered")
st.title("🍎 SnapFresh - Food Freshness Predictor")

# Ask for food name first
food = st.text_input("Enter the name of a food item (e.g., Apple, Banana):")

if food:
    if st.button("Train Model"):
        st.info("Fetching images and training model...")

        for label in ["fresh", "moderate", "rotten"]:
            fetch_images(food, label, count=15)

        X, y = load_data()
        if len(X) == 0:
            st.error("No images were loaded. Try another food.")
        else:
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
            model = KNeighborsClassifier(n_neighbors=3)
            model.fit(X_train, y_train)
            st.success("Model trained successfully! 🎉")

            uploaded_image = st.file_uploader("Upload a food image to check freshness", type=["jpg", "png"])
            if uploaded_image:
                image = Image.open(uploaded_image).resize((64, 64))
                st.image(image, caption="Uploaded Image", width=150)
                img_array = np.array(image).flatten().reshape(1, -1)
                prediction = model.predict(img_array)
                label = le.inverse_transform(prediction)[0]

                days = spoilage_date_prediction(label)
                st.subheader(f"🟢 Predicted Freshness: **{label.upper()}**")
                st.write(f"🕒 Expected spoilage in **{days}** days.")
