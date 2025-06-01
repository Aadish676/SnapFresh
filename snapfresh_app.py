import streamlit as st
import os
import shutil
import random
from PIL import Image, ImageDraw
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# Create necessary folders
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# -----------------------------
# Generate dummy images for testing
def fetch_images(food_name, label, count=15):
    for i in range(count):
        img = Image.new("RGB", (64, 64), color=(random.randint(100, 255), 255, random.randint(0, 150)))
        draw = ImageDraw.Draw(img)
        draw.text((5, 25), label, fill=(0, 0, 0))
        img.save(f"dataset/{label}_{i}.jpg")

# -----------------------------
# Preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path).resize((64, 64))
    return np.array(img).flatten()

# -----------------------------
# Train the KNN model
def train_model():
    X, y = [], []
    for filename in os.listdir("dataset"):
        if filename.endswith(".jpg"):
            label = filename.split("_")[0]
            img_array = preprocess_image(os.path.join("dataset", filename))
            X.append(img_array)
            y.append(label)
    if len(X) == 0:
        return None, None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

# -----------------------------
# Predict freshness
def predict_freshness(model, img_file):
    img = Image.open(img_file).resize((64, 64))
    features = np.array(img).flatten().reshape(1, -1)
    prediction = model.predict(features)[0]
    spoilage = {"fresh": 5, "moderate": 2, "rotten": 0}
    return prediction.upper(), spoilage.get(prediction.lower(), "?")

# -----------------------------
# Streamlit UI
st.title("🍌 SnapFresh – Food Freshness Estimator")

# Ask for food type first
food_name = st.text_input("Enter the food name (e.g., Banana):")

# Clear old dataset if needed
if st.button("Clear Previous Data"):
    shutil.rmtree("dataset")
    os.makedirs("dataset")
    st.success("Old dataset cleared.")

# Generate dummy dataset & train
if st.button("Train Model"):
    if not food_name:
        st.warning("Please enter a food name.")
    else:
        fetch_images(food_name, "fresh")
        fetch_images(food_name, "moderate")
        fetch_images(food_name, "rotten")
        st.info("Dummy images generated.")
        model, acc = train_model()
        if model:
            st.success(f"Model trained successfully with accuracy: {acc:.2f}")
        else:
            st.error("Training failed. No images found.")

# Image prediction
st.subheader("📷 Upload an image to predict freshness")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    if os.path.exists("dataset") and len(os.listdir("dataset")) > 0:
        model, _ = train_model()
        if model:
            prediction, days = predict_freshness(model, uploaded_image)
            st.markdown(f"### 🟢 Predicted Freshness: `{prediction}`")
            st.markdown(f"### 🕒 Estimated spoilage in: `{days}` days")
        else:
            st.error("Model not trained. Please click 'Train Model' first.")
    else:
        st.error("No training data found. Please train the model first.")
