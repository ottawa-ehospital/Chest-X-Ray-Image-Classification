from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join('uploads', filename)
    file.save(file_path)

    image = preprocess_image(file_path)

    prediction = model.predict(np.expand_dims(image, axis=0))

    predicted_class = classes[np.argmax(prediction)]

    os.remove(file_path)

    return jsonify({'class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)