from flask import Flask
from keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import marshal
import numpy as np
import shap
import tensorflow as tf
import os

app = Flask(__name__)

def process_input(images_directory):
    background_data = []
    image_paths = [os.path.join(images_directory, f) for f in os.listdir(images_directory) if os.path.isfile(os.path.join(images_directory, f))]
    
    for i, image_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
        try:
            image = load_img(image_path, target_size=(224, 224))
            preprocessed_image = img_to_array(image) / 255.0
            background_data.append(preprocessed_image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    return np.array(background_data)


# Define the base path for your images
images_base_path = "../Webapp/images"


# Create background data using the process_input function
background_train = process_input(images_base_path)

# Serialize background data using marshal (optional)
serialized_data = marshal.dumps(background_train)

# Convert background data to numpy array
background_train_np = np.array(background_train)

# Define models
model_7_14 = tf.keras.models.load_model('../Webapp/templates/36_Multi_1e-5_500_Unfreeze.h5')
model_15_23 = tf.keras.models.load_model('../Webapp/templates/25_Multi_1e-6_500_Unfreeze.h5')

# Create separate models for each output you want to explain
model7_14_age = tf.keras.Model(inputs=model_7_14.input, outputs=model_7_14.get_layer('Prediction_Age').output)
model7_14_gender = tf.keras.Model(inputs=model_7_14.input, outputs=model_7_14.get_layer('Prediction_Gender').output)
model15_23_age = tf.keras.Model(inputs=model_15_23.input, outputs=model_15_23.get_layer('Prediction_Age').output)
model15_23_gender = tf.keras.Model(inputs=model_15_23.input, outputs=model_15_23.get_layer('Prediction_Gender').output)

# Create a GradientExplainer with the background data
explainer7_14_age = shap.GradientExplainer(model7_14_age, background_train_np)
explainer7_14_gender = shap.GradientExplainer(model7_14_gender, background_train_np)
explainer15_23_age = shap.GradientExplainer(model15_23_age, background_train_np)
explainer15_23_gender = shap.GradientExplainer(model15_23_gender, background_train_np)

