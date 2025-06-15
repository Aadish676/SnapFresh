# SnapFresh: ML-based Food Freshness Predictor
# Note: This is a simplified prototype code for demonstration purposes.

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template
from datetime import datetime

# Flask App Setup
app = Flask(__name__)

# Load Pre-trained ML Model (Assuming it's trained on spoilage phases)
MODEL_PATH = 'snapfresh_model.h5'
model = load_model(MODEL_PATH)

# Image preprocessing function
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # normalize
    image = np.expand_dims(image, axis=0)
    return image

# Prediction function
def predict_freshness(image):
    classes = ['Fresh', 'Mid-Fresh', 'Spoiling', 'Spoiled']
    prediction = model.predict(image)
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, float(confidence)

# Routes
@app.route('/')
def home():
    return render_template('index.html')  # Basic upload UI

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty file'})

    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    # Process and predict
    img = preprocess_image(filepath)
    label, confidence = predict_freshness(img)

    result = {
        'predicted_freshness': label,
        'confidence': f"{confidence*100:.2f}%",
        'predicted_spoilage_date': (datetime.now().date()).isoformat() if label == 'Fresh' else 'Soon'
    }
    return jsonify(result)

# Run the app
if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)

