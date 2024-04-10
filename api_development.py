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
# The URL of the file on OneDrive
url = 'https://uottawa-my.sharepoint.com/personal/pshey040_uottawa_ca/_layouts/15/guestaccess.aspx?share=EssqmnhTvktLjxdExPc2wGgBqVKIG_LCYbtFup3r5IwDPA&e=TUuJ03'

# Send a GET request to the URL
response = requests.get(url)

# Write the content of the response to a file
with open('chest_xray_classification', 'wb') as f:
    f.write(response.content)

# Now load the model from the local path
loaded_model = tf.saved_model.load('chest_xray_classification')

# Get the concrete function for inference
infer = loaded_model.signatures["serving_default"]

# Load label encoder
le = joblib.load('labels_chest.pkl')

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
