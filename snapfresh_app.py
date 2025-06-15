import streamlit as st
from PIL import Image
import random

# App Config
st.set_page_config(page_title="SnapFresh - Food Freshness Detector", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        .uploaded-img img {
            max-width: 250px !important;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        .stImage > img {
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Info
st.sidebar.image("https://i.ibb.co/7jRW3Pz/fresh-icon.png", width=100)
st.sidebar.title("SnapFresh")
st.sidebar.markdown("🍏 Predict food freshness with one click.")

# Main Title
st.title("🥗 SnapFresh - AI Freshness Detector")
st.caption("A lightweight demo using simulated freshness logic")

# File Upload
uploaded_file = st.file_uploader("📷 Upload a food image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Image Display in 2 columns
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### Preview")
            st.markdown('<div class="uploaded-img">', unsafe_allow_html=True)
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_column_width=False)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown("### ⏳ Predicting freshness...")
            # Simulated predictions
            freshness_levels = {
                "Fresh": "🟢 Fresh - Great to consume!",
                "Stale": "🟡 Stale - Eat soon!",
                "Almost Spoiled": "🟠 Almost Spoiled - Consume today!",
                "Spoiled": "🔴 Spoiled - Unsafe to eat!"
            }

            selected = random.choice(list(freshness_levels.keys()))
            spoil_days = {
                "Fresh": random.randint(5, 7),
                "Stale": random.randint(3, 4),
                "Almost Spoiled": random.randint(1, 2),
                "Spoiled": 0
            }

            st.success(freshness_levels[selected])
            if spoil_days[selected] > 0:
                st.info(f"⏰ Estimated Spoilage in: {spoil_days[selected]} day(s)")
            else:
                st.error("⚠️ Already spoiled — do not consume.")

    except Exception:
        st.warning("⚠️ Unable to process the image. Please try a different file.")
else:
    st.info("👆 Upload a food image to get started.")

st.markdown("---")
st.caption("SnapFresh © 2025 | Smart, Safe, Sustainable Eating")

