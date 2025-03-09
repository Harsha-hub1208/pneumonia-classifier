import os
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ✅ Download dataset from Google Drive (if needed)
dataset_url = "https://drive.google.com/drive/folders/1lTZTGLFLG1HnE4nAlCQSw0W5ojpBSkgc?usp=drive_link"  # Replace with Google Drive link
dataset_path = "dataset.zip"

if not os.path.exists("dataset/train"):
    gdown.download(dataset_url, dataset_path, quiet=False)
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall("dataset")

# Function to load and preprocess dataset
def data_new(path):
    subfolders = os.listdir(path)
    images = []
    labels = []

    for sub in subfolders:
        if sub.endswith('.DS_Store'):
            continue
        else:
            new_path = os.path.join(path, sub)
            for image in os.listdir(new_path):
                if image.endswith('.jpeg') or image.endswith('.jpg') or image.endswith('.png'):
                    img_path = os.path.join(new_path, image)
                    images.append(img_path)
                    labels.append(sub)

    df = pd.DataFrame({'images': images, 'label': labels})
    return df

# Dataset paths
train_dir = "dataset/train"
test_dir = "dataset/test"
val_dir = "dataset/val"

# Data Augmentation
train_generator = ImageDataGenerator(rescale=1/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_generator = ImageDataGenerator(rescale=1/255)
val_generator = ImageDataGenerator(rescale=1/255)

# Create data loaders
train_data = train_generator.flow_from_directory(train_dir, target_size=(120,120), batch_size=16, class_mode='binary')
test_data = test_generator.flow_from_directory(test_dir, target_size=(120,120), batch_size=16, class_mode='binary')
val_data = val_generator.flow_from_directory(val_dir, target_size=(120,120), batch_size=16, class_mode='binary')

# Define CNN Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(120,120,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(512, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(train_data, epochs=25, validation_data=val_data)

# ✅ Save trained model
model.save("model.h5")
print("Model saved successfully!")

