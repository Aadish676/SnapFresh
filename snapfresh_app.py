import streamlit as st
from PIL import Image
import random

# Custom CSS to resize image preview
st.markdown("""
    <style>
    .uploaded-img img {
        max-width: 250px !important;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Info
st.sidebar.image("https://i.ibb.co/7jRW3Pz/fresh-icon.png", width=100)
st.sidebar.title("SnapFresh")
st.sidebar.markdown("🍎 A smart way to check how fresh your food is!")

# Main Title
st.title("📷 SnapFresh - Food Freshness Detector")

# Description
st.markdown("""
Welcome to **SnapFresh**!  
Just upload a picture of your food item and we'll give you an idea about its **freshness** and **spoilage timeline**.  
This is a **demo version** using simulated predictions.
""")

# Upload section
uploaded_file = st.file_uploader("🖼️ Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Preview")
        st.markdown('<div class="uploaded-img">', unsafe_allow_html=True)
        st.image(Image.open(uploaded_file), use_column_width=False)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("### ⏳ Predicting freshness...")
        # Simulate result
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
else:
    st.info("Please upload a food image to begin.")

