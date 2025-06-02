import streamlit as st
from PIL import Image
import os
import requests
from io import BytesIO

# Dummy model prediction function (replace with real model)
def predict_freshness(image):
    # This is a placeholder: add your real model inference here
    return "Fresh"  # or "Moderate" / "Rotten"

# Function to download sample images quietly
def download_sample_images():
    # Only run if folder doesn't exist or empty
    folder = "sample_images"
    if not os.path.exists(folder) or len(os.listdir(folder)) < 50:
        os.makedirs(folder, exist_ok=True)
        urls = [
            # Add 50 food image URLs here for training dataset (shortened example)
            "https://images.unsplash.com/photo-1504674900247-0877df9cc836",
            "https://images.unsplash.com/photo-1525755662778-989d0524087e",
            # Add more URLs to reach 50...
        ]
        for i, url in enumerate(urls):
            try:
                response = requests.get(url + "?w=500")  # smaller size
                img = Image.open(BytesIO(response.content))
                img.save(os.path.join(folder, f"food_{i+1}.jpg"))
            except Exception as e:
                print(f"Error downloading {url}: {e}")

def main():
    st.set_page_config(page_title="SnapFresh - Food Freshness Detector", layout="centered")

    st.title("SnapFresh - Food Freshness Detector")
    name = st.text_input("Enter your name")

    if not name:
        st.warning("Please enter your name to proceed.")
        return

    st.write(f"Hello, {name}! Upload an image of food to check freshness.")

    # Download dataset quietly (hidden spinner)
    with st.spinner("Preparing dataset..."):
        download_sample_images()

    uploaded_file = st.file_uploader("Upload a food image (jpg/png)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)

        if st.button("Predict Freshness"):
            result = predict_freshness(image)
            st.success(f"Prediction: {result}")

if __name__ == "__main__":
    main()
