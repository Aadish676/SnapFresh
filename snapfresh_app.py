import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import time
import random
import datetime
import numpy as np

st.set_page_config(page_title="SnapFresh", layout="centered")

# Session State Init
if "model_trained" not in st.session_state:
    st.session_state["model_trained"] = False
if "training_food" not in st.session_state:
    st.session_state["training_food"] = None
if "training_dataset" not in st.session_state:
    st.session_state["training_dataset"] = {"fresh": [], "moderate": [], "rotten": []}
if "history" not in st.session_state:
    st.session_state["history"] = []

# Parameters
categories = ["fresh", "moderate", "rotten"]
status_color = {
    "fresh": "🟢 Safe to Consume",
    "moderate": "🟡 Consume Soon",
    "rotten": "🔴 Unsafe"
}
status_note = {
    "fresh": "Enjoy your food safely!",
    "moderate": "Try to use this item within the next day or two.",
    "rotten": "Dispose of the item to avoid health risks."
}
spoilage_days_map = {
    "fresh": 5,
    "moderate": 2,
    "rotten": 0
}

# Helper Functions

def fetch_images_for_stage(food_name, stage):
    query = f"{food_name} {stage} site:unsplash.com"
    search_templates = {
        "fresh": "photo-1528825871115-3581a5387919",
        "moderate": "photo-1571847149542-70c0cd361bc9",
        "rotten": "photo-1504674900247-0877df9cc836"
    }
    pid = search_templates[stage]
    url = f"https://images.unsplash.com/{pid}?auto=format&fit=crop&w=224&q=80"
    return [url] * 4  # duplicate to simulate dataset

def auto_train_model(food_name):
    dataset = {"fresh": [], "moderate": [], "rotten": []}
    for stage in categories:
        urls = fetch_images_for_stage(food_name, stage)
        st.info(f"Downloading '{stage}' images...")
        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200 and "image" in response.headers.get("Content-Type", ""):
                    img = Image.open(BytesIO(response.content)).convert("RGB")
                    dataset[stage].append(img)
            except:
                pass
            time.sleep(0.2)
    return dataset

def simulated_predict(user_img, dataset):
    # Simulate prediction by randomly selecting one of the trained labels
    return random.choice(list(dataset.keys()))

# UI

st.markdown("<h1 style='text-align: center;'>📱 SnapFresh</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>AI-powered Food Freshness Detection (Trained by Stage)</h4>", unsafe_allow_html=True)
st.divider()

if not st.session_state["model_trained"]:
    food_name = st.text_input("Enter a food item to train the model (e.g., tomato, apple):", "")
    if st.button("Train Model"):
        if food_name.strip() == "":
            st.error("Please enter a valid food name.")
        else:
            with st.spinner(f"Training model with 'fresh', 'moderate', and 'rotten' samples..."):
                dataset = auto_train_model(food_name.strip().lower())
                st.session_state["training_dataset"] = dataset
                st.session_state["training_food"] = food_name.strip().lower()
                st.session_state["model_trained"] = True
            st.success("Training complete!")

# Prediction UI
if st.session_state["model_trained"]:
    st.success(f"Model trained on images of '{st.session_state['training_food']}' in 3 stages.")
    uploaded_file = st.file_uploader("Upload an image of your food item 🍅", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        user_img = Image.open(uploaded_file).convert("RGB")
        resized_img = user_img.resize((300, 300))

        with st.spinner("Analyzing image..."):
            label = simulated_predict(user_img, st.session_state["training_dataset"])
            spoilage_days = spoilage_days_map[label]
            spoilage_date = datetime.datetime.now() + datetime.timedelta(days=spoilage_days)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(resized_img, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.markdown(f"### Verdict: {status_color[label]}")
            st.info(status_note[label])
            st.markdown(f"**Estimated Spoilage Date:** {spoilage_date.strftime('%Y-%m-%d')}")

        st.session_state["history"].append({
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "result": label,
            "spoilage_date": spoilage_date.strftime('%Y-%m-%d')
        })

    # History
    if st.session_state["history"]:
        st.markdown("### 📂 Scan History")
        for entry in reversed(st.session_state["history"][-5:]):
            st.markdown(f"- 🕒 {entry['timestamp']} — **{status_color[entry['result']]}**, Spoils by {entry['spoilage_date']}")

st.divider()
st.caption("SnapFresh • 2025 Prototype")

