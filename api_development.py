import os
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import numpy as np
import tensorflow as tf
import joblib
import io
import requests

# Download the model file from GitHub
model_url = "https://github.com/ottawa-ehospital/Chest-X-Ray-Image-Classification/raw/main/chest_xray_classification_model_20240407_071036"
model_path = tf.keras.utils.get_file("chest_xray_classification_model_20240407_071036", model_url)

# Download the label encoder file from GitHub
le_url = "https://github.com/ottawa-ehospital/Chest-X-Ray-Image-Classification/raw/main/labels_chest.pkl"
le_path = tf.keras.utils.get_file("labels_chest.pkl", le_url)

app = Flask(__name__)

CORS(app)

# Load trained model
loaded_model = tf.saved_model.load(model_path)

# Get the concrete function for inference
infer = loaded_model.signatures["serving_default"]

# Load label encoder
le = joblib.load(le_path)

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    # Read the file into a BytesIO object
    file = io.BytesIO(file.read())
    img = load_img(file, target_size=(250, 250))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = infer(tf.constant(img_array))['output_0']  # Use the infer function for inference
    predicted_class = np.argmax(prediction)
    # Use label encoder to get original class label
    predicted_label = le.inverse_transform([predicted_class])
    result = {'class': predicted_label[0]}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)  # Set debug=False for production