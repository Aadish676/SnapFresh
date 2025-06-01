import streamlit as st
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import datetime

st.set_page_config(page_title="SnapFresh", layout="centered")

# Categories and URLs (sample 3 images per category for demo)
image_urls = {
    "Good": [
        "https://images.unsplash.com/photo-1567306226416-28f0efdc88ce?auto=format&fit=crop&w=224&q=80",
        "https://images.unsplash.com/photo-1506806732259-39c2d0268443?auto=format&fit=crop&w=224&q=80",
        "https://images.unsplash.com/photo-1528825871115-3581a5387919?auto=format&fit=crop&w=224&q=80",
    ],
    "Moderate": [
        "https://images.unsplash.com/photo-1576085898320-8d952d5c9146?auto=format&fit=crop&w=224&q=80",
        "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=224&q=80",
        "https://images.unsplash.com/photo-1573531435624-6ec6e9cdd5c0?auto=format&fit=crop&w=224&q=80",
    ],
    "Rotten": [
        "https://images.unsplash.com/photo-1542831371-d531d36971e6?auto=format&fit=crop&w=224&q=80",
        "https://images.unsplash.com/photo-1518976024611-48802c3c3ff9?auto=format&fit=crop&w=224&q=80",
        "https://images.unsplash.com/photo-1508214751196-bcfd4ca60f91?auto=format&fit=crop&w=224&q=80",
    ]
}

# Manual simple KNN
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(test_feature, train_features, train_labels, k=3):
    distances = [euclidean_distance(test_feature, f) for f in train_features]
    idxs = np.argsort(distances)[:k]
    votes = {}
    for i in idxs:
        votes[train_labels[i]] = votes.get(train_labels[i], 0) + 1
    return max(votes, key=votes.get)

# Feature extractor: resize and flatten normalized pixels
def extract_feature(image: Image.Image):
    img = image.resize((64,64)).convert('RGB')
    arr = np.array(img) / 255.0
    return arr.flatten()

# Spoilage days mapping
spoilage_days = {
    "Good": 5,
    "Moderate": 2,
    "Rotten": 0
}

# Download training images and prepare dataset
@st.cache_data(show_spinner=False)
def load_training_data():
    features = []
    labels = []
    for label, urls in image_urls.items():
        for url in urls:
            try:
                res = requests.get(url, timeout=5)
                img = Image.open(BytesIO(res.content))
                feat = extract_feature(img)
                features.append(feat)
                labels.append(label)
            except Exception as e:
                # Skip bad images
                pass
    return np.array(features), labels

# Main app UI
st.title("📱 SnapFresh: Food Freshness Detector")
st.write("Upload a photo of your food item and get freshness status and estimated spoilage date.")

uploaded_file = st.file_uploader("Upload food image (jpg/jpeg/png):", type=["jpg", "jpeg", "png"])

if uploaded_file:
    user_img = Image.open(uploaded_file)
    st.image(user_img, caption="Your Food Image", width=300)

    # Extract feature of uploaded image
    user_feature = extract_feature(user_img)

    # Load training data
    train_features, train_labels = load_training_data()

    # Predict category
    prediction = knn_predict(user_feature, train_features, train_labels, k=3)

    # Calculate spoilage date
    days_left = spoilage_days.get(prediction, 0)
    spoilage_date = datetime.date.today() + datetime.timedelta(days=days_left)

    # Display results
    freshness_color = {
        "Good": "🟢 Safe to Consume",
        "Moderate": "🟡 Consume Soon",
        "Rotten": "🔴 Unsafe - Discard"
    }

    freshness_note = {
        "Good": "Your food looks fresh. Enjoy!",
        "Moderate": "Your food is moderately fresh. Consume soon.",
        "Rotten": "Your food looks spoiled. Please discard."
    }

    st.markdown(f"### Freshness Status: {freshness_color[prediction]}")
    st.info(freshness_note[prediction])
    if days_left > 0:
        st.markdown(f"**Estimated Spoilage Date:** {spoilage_date.strftime('%Y-%m-%d')}")
    else:
        st.markdown(f"**Spoilage Date:** Already spoiled!")

st.caption("SnapFresh • Prototype App • 2025")
