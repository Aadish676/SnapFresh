import streamlit as st
import os
import shutil
import requests
from PIL import Image
from io import BytesIO
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from torchvision import transforms
from duckduckgo_search import DDGS

# Setup
st.set_page_config(page_title="SnapFresh", page_icon="🍎", layout="centered")

# UI Title
st.markdown("<h1 style='text-align: center;'>🍏 SnapFresh - Food Freshness Predictor</h1>", unsafe_allow_html=True)

# Hide warnings and Streamlit elements
st.markdown("""
<style>
    .st-emotion-cache-1v0mbdj.e115fcil1 { visibility: hidden; height: 0; }
    footer { visibility: hidden; }
    img { max-height: 200px !important; }
</style>
""", unsafe_allow_html=True)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Function to download images silently
def download_images(food, label, count=50):
    query = f"{food} {label} food"
    with DDGS() as ddgs:
        results = ddgs.images(query, max_results=count)
        os.makedirs(f"dataset/{label}", exist_ok=True)
        i = 0
        for result in results:
            url = result.get("image")
            if url:
                try:
                    img_data = requests.get(url, timeout=5).content
                    img = Image.open(BytesIO(img_data)).convert("RGB")
                    img = transform(img)
                    save_img = transforms.ToPILImage()(img)
                    save_img.save(f"dataset/{label}/{food}_{label}_{i}.jpg")
                    i += 1
                    if i >= count:
                        break
                except:
                    continue

# Extract features from images
def extract_features_and_labels():
    X, y = [], []
    for label in os.listdir("dataset"):
        folder = os.path.join("dataset", label)
        for file in os.listdir(folder):
            try:
                img_path = os.path.join(folder, file)
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img)
                X.append(img_tensor.view(-1).numpy())
                y.append(label)
            except:
                continue
    return np.array(X), np.array(y)

# Ask user for food type
food_name = st.text_input("Enter the food item you want to check:", "")

if food_name:
    # Download training images if not already present
    if not os.path.exists("dataset"):
        for label in ["fresh", "moderate", "rotten"]:
            download_images(food_name, label, 50)

    # Train model
    X, y = extract_features_and_labels()
    if len(set(y)) >= 2:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train, y_train)

        # Upload image
        uploaded_image = st.file_uploader("Upload an image of the food:", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            img = Image.open(uploaded_image).convert("RGB")
            st.image(img, caption="Uploaded Image", use_container_width=True)

            img_tensor = transform(img).view(1, -1).numpy()
            prediction = model.predict(img_tensor)
            predicted_label = le.inverse_transform(prediction)[0]

            # Show result
            st.markdown(f"""
                <h3 style='text-align: center;'>🧠 Predicted Freshness: <span style='color:#00B300'>{predicted_label.title()}</span></h3>
            """, unsafe_allow_html=True)
    else:
        st.warning("Not enough images for training. Please try a different food item.")

# Clean up dataset after session ends (optional)
# shutil.rmtree("dataset", ignore_errors=True)


