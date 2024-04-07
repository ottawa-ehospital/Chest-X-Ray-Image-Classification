import os
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import joblib

def preprocess_data(data_dir, target_size):
    X = []
    Y = []
    for label in os.listdir(data_dir):
        if label == '.DS_Store':  # Skip .DS_Store file
            continue
        if label == 'Healthy':  # Special case for 'Healthy' folder
            label_dir = os.path.join(data_dir, label)
        else:
            label_dir = os.path.join(data_dir, label, 'Disease')  # Accessing 'Disease' folder within each label folder
        for image_file in os.listdir(label_dir):
            if image_file.endswith('.jpg') or image_file.endswith('.jpeg'):  # Filter JPEG images
                image_path = os.path.join(label_dir, image_file)
                img = load_img(image_path, target_size=target_size)
                img_array = img_to_array(img) / 255.0  # Normalize pixel values
                X.append(img_array)
                Y.append(label)
    X = np.array(X)
    Y= np.array(Y)
    return X, Y

data_dir = r"C:\Users\Parsa\Downloads\Chest-XXX-ray\Chest-X-ray"  # Update with your data directory
target_size = (250, 250)  # Update with your desired target size
X, Y= preprocess_data(data_dir, target_size)

# Convert string labels to integers
le = LabelEncoder()
Y = le.fit_transform(Y)

# Save the label encoder
joblib.dump(le, 'labels_chest.pkl')

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,  # milder augmentations
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)
datagen.fit(X_train)

# Model definition and training
base_model = ResNet50(include_top=False, input_shape=(250, 250, 3), pooling='avg', weights='imagenet')

# Fine-tuning
for layer in base_model.layers:
    layer.trainable = True
for layer in base_model.layers[:-10]:  # Unfreeze the last 10 layers
    layer.trainable = False

model = Sequential([
    base_model,
    Dense(512, activation='relu'),  # Increase the number of units
    Dropout(0.5),
    Dense(4, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Increase the learning rate

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20)  # Increase patience

model.fit(datagen.flow(X_train, Y_train, batch_size=32), epochs=50, validation_data=(X_test, Y_test), callbacks=[early_stopping])  # Increase the number of epochs

# Evaluate model
loss, accuracy = model.evaluate(X_test, Y_test)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')

# Save trained model
now = datetime.now()
now_str = now.strftime("%Y%m%d_%H%M%S")
filename = f"chest_xray_classification_model_{now_str}"
tf.saved_model.save(model, filename)
