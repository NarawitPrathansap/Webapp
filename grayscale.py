import subprocess
import pandas as pd
import numpy as np
import sys
import shap
from keras.preprocessing.image import load_img, img_to_array
import request



#please change this!
def process_input(dt_test):
    background_data = []
    for i, image_path in enumerate(dt_test['Path_Name']):
        print(f"Processing image {i+1}/{len(dt_test)}: {image_path}")
        try:
            image = load_img(image_path, target_size=(224, 224))
            preprocessed_image = img_to_array(image) / 255.0
            background_data.append(preprocessed_image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    return np.array(background_data)

def read_dataframe_from_csv(csv_file_path):
    return pd.read_csv(csv_file_path)

def create_explainers(background_data):
    # Call the bg_shap.py file to create explainers
    result = subprocess.run(["python", "bg_shap.py"], capture_output=True)
    if result.returncode == 0:
        explainer7_14_age, explainer7_14_gender, explainer15_23_age, explainer15_23_gender = result.stdout.splitlines()
        # Convert byte strings to actual objects
        explainer7_14_age = eval(explainer7_14_age.decode())
        explainer7_14_gender = eval(explainer7_14_gender.decode())
        explainer15_23_age = eval(explainer15_23_age.decode())
        explainer15_23_gender = eval(explainer15_23_gender.decode())
        return explainer7_14_age, explainer7_14_gender, explainer15_23_age, explainer15_23_gender
    else:
        print("Error:", result.stderr.decode())
        return None

# Here you would receive the outputs from the functions 1 and 2 as JSON data in the request
data = request.json

# Extract the outputs from the JSON data
output_1 = data['output_1']
output_2 = data['output_2']

def select_explainer(output_1, output_2, explainer7_14_age, explainer7_14_gender, explainer15_23_age, explainer15_23_gender):
    if 7 <= output_1 <= 14 and output_2 in [1, 3]:
        return explainer7_14_age
    elif 7 <= output_1 <= 14 and output_2 in [2, 4]:
        return explainer7_14_gender
    elif 15 <= output_1 <= 23 and output_2 in [1, 3]:
        return explainer15_23_age
    elif 15 <= output_1 <= 23 and output_2 in [2, 4]:
        return explainer15_23_gender
    else:
        return None

if __name__ == '__main__':
        # Determine which explainer to use based on the output conditions
    explainer_to_use = select_explainer(output_1, output_2)

    if explainer_to_use:
            # Calculate SHAP values using the selected explainer
        shap_values = explainer_to_use.shap_values(background_data)

data= [np.array(shap_values)]
image_array = data[0]
positive =np.where(image_array >= 0, image_array, 0)
negative = np.where(image_array < 0, image_array, 0)
negative_aps =  np.abs(negative)

flattened_array_pos = positive.flatten()
flattened_array_neg = negative_aps.flatten()

normalized_array_pos = (flattened_array_pos - np.min(flattened_array_pos)) / (np.max(flattened_array_pos) - np.min(flattened_array_pos))
normalized_array_neg = (flattened_array_neg - np.min(flattened_array_neg)) / (np.max(flattened_array_neg) - np.min(flattened_array_neg))

normalized_positive = normalized_array_pos.reshape(positive.shape)
normalized_neg = normalized_array_neg.reshape(negative_aps.shape)

grayscale_image_pos = normalized_positive/ 3.0
grayscale_image_neg = normalized_neg/ 3.0

grayscale_image_positive = np.mean(grayscale_image_pos, axis=4)
grayscale_image_negative = np.mean(grayscale_image_neg, axis=4)

grayscale_image_positive = grayscale_image_positive.squeeze()
grayscale_image_negative = grayscale_image_negative.squeeze()

percentile_95_pos = np.percentile(grayscale_image_positive, 95)
percentile_95_neg = np.percentile(grayscale_image_negative, 95)

grayscale_pos_thresholded = grayscale_image_positive
grayscale_neg_thresholded = grayscale_image_negative

grayscale_pos_thresholded[grayscale_pos_thresholded < percentile_95_pos] = 0
grayscale_neg_thresholded[grayscale_neg_thresholded < percentile_95_neg] = 0




