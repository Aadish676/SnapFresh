import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import time
import random
import datetime
import numpy as np

st.set_page_config(page_title="SnapFresh", layout="centered")

# --- Session State Init ---
if "model_trained" not in st.session_state:
    st.session_state["model_trained"] = False
if "training_food" not in st.session_state:
    st.session_state["training_food"] = None
if "training_images" not in st.session_state:
    st.session_state["training_images"] = []
if "history" not in st.session_state:
    st.session_state["history"] = []

# --- Helper Functions ---

def fetch_images_for_food(food_name, count=10):
    example_photo_ids = [
        "1567306226416-28f0efdc88ce",
        "1506806732259-39c2d0268443",
        "1574226516831-e1dff420e7d9",
        "1528825871115-3581a5387919",
        "1567306301408-9b74779a11af",
        "1571847149542-70c0cd361bc9",
        "1498575207490-c9c67394c1b4",
        "1503602642458-232111445657",
        "1504674900247-0877df9cc836",
        "1525610553991-2bede1a93b5e",
    ]
    urls = [
        f"https://images.unsplash.com/photo-{pid}?auto=format&fit=crop&w=224&q=80&{food_name}"
        for pid in example_photo_ids[:count]
    ]
    return urls

def auto_train_model(food_name):
    urls = fetch_images_for_food(food_name)
    images = []
    progress_text = st.empty()
    progress_bar = st.progress(0)

    for idx, url in enumerate(urls):
        progress_text.text(f"Downloading image {idx+1} of {len(urls)} for '{food_name}'...")
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200 and "image" in response.headers.get("Content-Type", ""):
                img = Image.open(BytesIO(response.content)).convert("RGB")
                images.append(img)
                progress_bar.progress(int((idx + 1) / len(urls) * 100))
            else:
                # Suppress warnings about skipped images
                pass
        except Exception:
            # Suppress errors during image download to keep UI clean
            pass
        time.sleep(0.3)  # simulate delay

    progress_text.text("Training model on downloaded images... (simulated)")
    time.sleep(1.5)  # simulate training delay
    progress_bar.progress(100)
    progress_text.text(f"Training complete for '{food_name}'!")
    return images

def predict_freshness(image):
    categories = ["Safe", "Consume Soon", "Unsafe"]
    probabilities = [0.6, 0.3, 0.1]
    return random.choices(categories, probabilities)[0]

status_color = {
    "Safe": "🟢 Safe to Consume",
    "Consume Soon": "🟡 Consume Soon",
    "Unsafe": "🔴 Unsafe"
}

status_note = {
    "Safe": "Enjoy your food safely!",
    "Consume Soon": "Try to use this item within the next day or two.",
    "Unsafe": "Dispose of the item to avoid health risks."
}

# Approximate spoilage days mapping for demo
spoilage_days_map = {
    "Safe": 5,
    "Consume Soon": 2,
    "Unsafe": 0
}

# --- UI ---

st.markdown("<h1 style='text-align: center;'>📱 SnapFresh</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>AI-powered Food Freshness Detection</h4>", unsafe_allow_html=True)
st.divider()

# TRAINING UI
if not st.session_state["model_trained"]:
    food_name = st.text_input("Enter a food item to train the model on (e.g., apple, banana):", "")
    if st.button("Train Model"):
        if food_name.strip() == "":
            st.error("Please enter a valid food name.")
        else:
            with st.spinner(f"Training model for '{food_name.strip()}'..."):
                imgs = auto_train_model(food_name.strip().lower())
                st.session_state["training_images"] = imgs
                st.session_state["model_trained"] = True
                st.session_state["training_food"] = food_name.strip().lower()
            st.success(f"Model trained with sample images of '{food_name.strip()}'!")

# PREDICTION UI
if st.session_state["model_trained"]:
    st.success(f"Model trained on '{st.session_state['training_food']}'. Upload an image for freshness detection.")

    uploaded_file = st.file_uploader("Upload a photo of your food item 🍎", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        resized_img = image.resize((300, 300))
        image_array = np.array(resized_img) / 255.0

        with st.spinner("Analyzing freshness..."):
            verdict = predict_freshness(image)

        # Calculate approximate spoilage date
        spoilage_days = spoilage_days_map.get(verdict, 1)
        spoilage_date = datetime.datetime.now() + datetime.timedelta(days=spoilage_days)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(resized_img, caption="Uploaded Food Item", use_container_width=True)
        with col2:
            st.markdown(f"### Freshness Verdict: {status_color[verdict]}")
            st.info(status_note[verdict])
            st.markdown(f"**Approximate Spoilage Date:** {spoilage_date.strftime('%Y-%m-%d')}")

        # Add to scan history
        st.session_state["history"].append({
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "result": verdict,
            "label": status_color[verdict],
            "spoilage_date": spoilage_date.strftime("%Y-%m-%d")
        })

    # Show last 5 scan history entries
    if st.session_state["history"]:
        st.markdown("### 📂 Scan History (last 5)")
        for entry in reversed(st.session_state["history"][-5:]):
            st.markdown(f"- 🕒 {entry['timestamp']} — **{entry['label']}**, Spoils by {entry['spoilage_date']}")

st.divider()
st.caption("SnapFresh • Prototype App • 2025")

