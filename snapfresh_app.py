import streamlit as st
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import os
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import datetime

st.set_page_config(page_title="SnapFresh", layout="centered")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "model" not in st.session_state:
    st.session_state.model = None
if "labels" not in st.session_state:
    st.session_state.labels = []

# UI Header
st.markdown("<h1 style='text-align: center;'>📱 SnapFresh</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Lightweight AI for Food Freshness Detection</h4>", unsafe_allow_html=True)
st.divider()

# Input for food item
food_item = st.text_input("Enter a food item (e.g., apple, banana)")

# URLs for stages
STAGES = {
    "fresh": f"https://source.unsplash.com/224x224/?fresh,{food_item}",
    "moderate": f"https://source.unsplash.com/224x224/?moderate,{food_item}",
    "rotten": f"https://source.unsplash.com/224x224/?rotten,{food_item}"
}

@st.cache_data(show_spinner=False)
def download_stage_images():
    data = []
    labels = []
    for label, url_template in STAGES.items():
        for _ in range(10):
            response = requests.get(url_template)
            try:
                img = Image.open(BytesIO(response.content)).convert("RGB")
                img = img.resize((64, 64))
                hist = extract_features(img)
                data.append(hist)
                labels.append(label)
            except:
                continue
    return data, labels

def extract_features(img):
    # Simple color histogram (flattened)
    hist = np.array(img).mean(axis=(0, 1))  # RGB mean
    return hist

@st.cache_resource(show_spinner=True)
def train_lightweight_model():
    data, labels = download_stage_images()
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(data, labels)
    return model, labels

# Train model
if food_item:
    st.info(f"Training AI to detect {food_item} freshness...")
    model, label_list = train_lightweight_model()
    st.session_state.model = model
    st.session_state.labels = label_list
    st.success("Model trained successfully!")
    st.divider()

# Upload image for prediction
uploaded_file = st.file_uploader("Upload a food image to check freshness", type=["jpg", "png", "jpeg"])

def predict_spoilage(freshness):
    today = datetime.date.today()
    if freshness == "fresh":
        return today + datetime.timedelta(days=5)
    elif freshness == "moderate":
        return today + datetime.timedelta(days=2)
    else:
        return today

# Prediction
if uploaded_file and st.session_state.model:
    image = Image.open(uploaded_file).convert("RGB")
    resized = image.resize((64, 64))
    feat = extract_features(resized).reshape(1, -1)

    prediction = st.session_state.model.predict(feat)[0]
    spoil_date = predict_spoilage(prediction)

    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.success(f"**Freshness Verdict:** {prediction.upper()}")
    st.info(f"**Expected Spoilage Date:** {spoil_date.strftime('%Y-%m-%d')}")

    st.session_state.history.append({
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "result": prediction,
        "spoilage": spoil_date.strftime("%Y-%m-%d")
    })

    st.divider()

# History
if st.session_state.history:
    st.markdown("### 📂 Scan History")
    for entry in reversed(st.session_state.history[-5:]):
        st.markdown(f"- 🕒 {entry['timestamp']} — **{entry['result'].capitalize()}**, Spoils by {entry['spoilage']}")

st.caption("SnapFresh • Lightweight Prototype • 2025")
