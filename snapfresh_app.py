import streamlit as st
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import datetime
import random
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
import tensorflow_hub as hub

st.set_page_config(page_title="SnapFresh - AI Food Freshness", layout="centered")

# Load MobileNetV2 feature extractor from TensorFlow Hub
@st.cache_resource(show_spinner=False)
def load_feature_extractor():
    model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    feature_extractor = hub.KerasLayer(model_url, input_shape=(224, 224, 3))
    return feature_extractor

feature_extractor = load_feature_extractor()

def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    if img_array.shape[-1] == 4:
        # Remove alpha channel if present
        img_array = img_array[..., :3]
    return img_array.astype(np.float32)

def get_embedding(image):
    preprocessed = preprocess_image(image)
    batch = np.expand_dims(preprocessed, axis=0)
    embedding = feature_extractor(batch)
    return embedding.numpy()[0]

# Dummy image URLs for demo (replace with real search or API calls)
# Each category has 50 images for better training
# Here URLs are just repeated to simulate 50 images per category
# In real app, you'd use an API or scraping to get unique URLs

def generate_dummy_urls(base_url, count=50):
    return [base_url + f"?v={i}" for i in range(count)]

def get_image_urls(food_item):
    # Normally, you would query an API like Unsplash or Bing Image Search for food + category
    # For demo, we simulate with example placeholder images repeated

    base_good = "https://images.unsplash.com/photo-1574226516831-e1dff420e37c"  # fresh fruit
    base_moderate = "https://images.unsplash.com/photo-1567306226416-28f0efdc88ce"  # slightly aged fruit
    base_rotten = "https://images.unsplash.com/photo-1504674900247-0877df9cc836"  # rotten fruit

    good_urls = generate_dummy_urls(base_good)
    moderate_urls = generate_dummy_urls(base_moderate)
    rotten_urls = generate_dummy_urls(base_rotten)

    return {
        "Good": good_urls,
        "Moderate": moderate_urls,
        "Rotten": rotten_urls,
    }

def download_images(urls):
    images = []
    for url in urls:
        try:
            response = requests.get(url, timeout=5)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            images.append(img)
        except Exception:
            # Skip invalid images
            continue
    return images

def train_knn_model(images_per_class):
    X = []
    y = []
    for label, images in images_per_class.items():
        for img in images:
            emb = get_embedding(img)
            X.append(emb)
            y.append(label)
    if not X:
        return None
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)
    return knn

def estimate_spoilage_date(freshness):
    today = datetime.date.today()
    if freshness == "Good":
        return today + datetime.timedelta(days=7)
    elif freshness == "Moderate":
        return today + datetime.timedelta(days=2)
    else:
        return today  # Rotten, already spoiled

st.title("📱 SnapFresh - AI Food Freshness Detector")

st.markdown("## Step 1: Enter your food item to train the model")
food_item = st.text_input("What food item do you want to check freshness for?", placeholder="e.g., apple, banana, tomato")

if food_item:
    st.info(f"Training model for '{food_item}'... This may take about 30-60 seconds.")

    urls_dict = get_image_urls(food_item)
    images_per_class = {}

    # Download and cache images per category
    for category, urls in urls_dict.items():
        with st.spinner(f"Downloading {len(urls)} images for '{category}'..."):
            images = download_images(urls)
        images_per_class[category] = images
        st.success(f"Downloaded {len(images)} images for category: {category}")

    # Train KNN model on embeddings
    with st.spinner("Extracting features and training classifier..."):
        knn_model = train_knn_model(images_per_class)

    if knn_model is None:
        st.error("Failed to train model due to no valid images.")
        st.stop()

    st.success("Model trained successfully! Now upload an image to predict freshness.")

    # Step 2: Upload an image to test
    uploaded_file = st.file_uploader("Upload a photo of your food item 🍎", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        user_image = Image.open(uploaded_file).convert("RGB")

        st.image(user_image, caption="Your Uploaded Food Item", width=250)

        with st.spinner("Predicting freshness..."):
            user_emb = get_embedding(user_image)
            prediction = knn_model.predict([user_emb])[0]
            probs = knn_model.predict_proba([user_emb])[0]
            confidence = max(probs) * 100

            spoil_date = estimate_spoilage_date(prediction)

        freshness_emojis = {
            "Good": "🟢 Fresh",
            "Moderate": "🟡 Moderate",
            "Rotten": "🔴 Rotten"
        }

        st.markdown(f"### Freshness Prediction: {freshness_emojis.get(prediction, '')} **{prediction}**")
        st.write(f"Confidence: {confidence:.1f}%")
        st.write(f"Estimated spoilage date: **{spoil_date.strftime('%Y-%m-%d')}**")

        if prediction == "Good":
            st.success("Your food item is fresh and safe to consume!")
        elif prediction == "Moderate":
            st.warning("Your food item is moderately fresh. Try to consume soon!")
        else:
            st.error("Your food item appears rotten. Dispose of it safely!")

else:
    st.info("Please enter a food item above to start training the model.")
