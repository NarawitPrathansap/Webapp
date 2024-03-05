from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import shap
import tensorflow as tf
import sys
import os

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

def create_explainers(background_data):
    background_data_np = np.array(background_data)
    model_7_14 = tf.keras.models.load_model('../Webapp/templates/36_Multi_1e-5_500_Unfreeze.h5')
    model_15_23 = tf.keras.models.load_model('../Webapp/templates/25_Multi_1e-6_500_Unfreeze.h5')

    # Use the correct layer names based on the error output
    model7_14_age = tf.keras.Model(inputs=model_7_14.input, outputs=model_7_14.get_layer('Prediction_Age').output)
    model7_14_gender = tf.keras.Model(inputs=model_7_14.input, outputs=model_7_14.get_layer('Prediction_Gender').output)
    model15_23_age = tf.keras.Model(inputs=model_15_23.input, outputs=model_15_23.get_layer('Prediction_Age').output)
    model15_23_gender = tf.keras.Model(inputs=model_15_23.input, outputs=model_15_23.get_layer('Prediction_Gender').output)

    explainer7_14_age = shap.GradientExplainer(model7_14_age, background_data_np)
    explainer7_14_gender = shap.GradientExplainer(model7_14_gender, background_data_np)
    explainer15_23_age = shap.GradientExplainer(model15_23_age, background_data_np)
    explainer15_23_gender = shap.GradientExplainer(model15_23_gender, background_data_np)

    return explainer7_14_age, explainer7_14_gender, explainer15_23_age, explainer15_23_gender
def compute_shap_values(model, background_data, test_images):
    # Initialize the SHAP explainer with the model and background data
    explainer = shap.GradientExplainer(model, background_data)

    # Compute SHAP values for the test images
    shap_values = explainer.shap_values(test_images)
    
    return shap_values
# Assume you have a function to load your model
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python bg_shap.py <background_images_directory> <model_path>")
        sys.exit(1)

    background_images_path = sys.argv[1]
    model_path = sys.argv[2]

    # Load and preprocess background data
    background_data = process_input(background_images_path)

    # Load your model
    model = load_model(model_path)

    # Optionally, select a subset of background_data as test_images or load separate test images
    test_images = background_data  # For demonstration, using the same as background

    # Compute SHAP values
    shap_values = compute_shap_values(model, background_data, test_images)