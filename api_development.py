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

app = Flask(__name__)
CORS(app)

# Define the path to the saved_model.pb file on Heroku
model_path = './chest_xray_classification_model_20240407_071036/saved_model.pb'

# Load trained model
loaded_model = tf.saved_model.load(model_path)

# Get the concrete function for inference
infer = loaded_model.signatures["serving_default"]

# Load label encoder
le_path = './labels_chest.pkl'
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
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
