import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

base_path = r"C:\Users\Parsa\Downloads\Chest-XXX-ray\Chest-X-ray"
classes = ['Healthy', 'Covid', 'Pneumonia', 'Tuberculosis']

def preprocess_image(image_path, target_size=(224,224)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = image / 255.0
    return image

def load_data(base_path, classes):
    data = []
    labels = []
    for label, class_name in enumerate(classes):
        class_folder = os.path.join(base_path, class_name)
        for image_filename in os.listdir(class_folder):
            image_path = os.path.join(class_folder, image_filename)
            image = preprocess_image(image_path)
            data.append(image)
            labels.append(label)
        return np.array(data), np.array(labels)

data, labels = load_data(base_path, classes)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.1, random_state=42)
