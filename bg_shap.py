from flask import Flask
from keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import marshal
import numpy as np
import shap
import tensorflow as tf
import sys  # Import sys module

app = Flask(__name__)

def process_input(dt_train):
    background_data = []
    if 'Path_Name' not in dt_train.columns:
        raise ValueError("The 'Path_Name' column does not exist in the DataFrame.")
    
    for i, image_path in enumerate(dt_train['Path_Name']):
        print(f"Processing image {i+1}/{len(dt_train)}: {image_path}")
        try:
            image = load_img(image_path, target_size=(224, 224))  # Ensure path is accessible
            preprocessed_image = img_to_array(image) / 255.0
            background_data.append(preprocessed_image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    return np.array(background_data)

def read_dataframe_from_csv(csv_file_path):
    return pd.read_csv(csv_file_path)

def create_explainers(background_data):
    background_data_np = np.array(background_data)
    model_7_14 = tf.keras.models.load_model('../Webapp/templates/36_Multi_1e-5_500_Unfreeze.h5')
    model_15_23 = tf.keras.models.load_model('../Webapp/templates/25_Multi_1e-6_500_Unfreeze.h5')

    model7_14_age = tf.keras.Model(inputs=model_7_14.input, outputs=model_7_14.get_layer('prediction_layer').output)
    model7_14_gender = tf.keras.Model(inputs=model_7_14.input, outputs=model_7_14.get_layer('prediction_layer2').output)
    model15_23_age = tf.keras.Model(inputs=model_15_23.input, outputs=model_15_23.get_layer('prediction_layer').output)
    model15_23_gender = tf.keras.Model(inputs=model_15_23.input, outputs=model_15_23.get_layer('prediction_layer2').output)

    explainer7_14_age = shap.GradientExplainer(model7_14_age, background_data_np)
    explainer7_14_gender = shap.GradientExplainer(model7_14_gender, background_data_np)
    explainer15_23_age = shap.GradientExplainer(model15_23_age, background_data_np)
    explainer15_23_gender = shap.GradientExplainer(model15_23_gender, background_data_np)

    return explainer7_14_age, explainer7_14_gender, explainer15_23_age, explainer15_23_gender

# Define the path to your CSV file containing the paths to images
csv_file_path = "../Webapp/templates/Bg_train.csv"

# Read DataFrame from CSV file
sdf_train = read_dataframe_from_csv(csv_file_path)

# Create background data using the process_input function
background_train = process_input(sdf_train)

# Serialize background data using marshal (optional)
serialized_data = marshal.dumps(background_train)

# Create explainers
explainer7_14_age, explainer7_14_gender, explainer15_23_age, explainer15_23_gender = create_explainers(background_train)

if __name__ == '__main__':
    app.run(debug=True)
