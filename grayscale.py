import sys
import json
import pandas as pd
import numpy as np
import shap
from keras.preprocessing.image import load_img, img_to_array

def process_input(data):
    # Assuming data is received as JSON
    shap_values = np.array(data['shap_values'])

    # Perform the processing as before
    image_array = shap_values[0]
    positive = np.where(image_array >= 0, image_array, 0)
    negative = np.where(image_array < 0, image_array, 0)
    negative_aps = np.abs(negative)

    flattened_array_pos = positive.flatten()
    flattened_array_neg = negative_aps.flatten()

    normalized_array_pos = (flattened_array_pos - np.min(flattened_array_pos)) / (np.max(flattened_array_pos) - np.min(flattened_array_pos))
    normalized_array_neg = (flattened_array_neg - np.min(flattened_array_neg)) / (np.max(flattened_array_neg) - np.min(flattened_array_neg))

    normalized_positive = normalized_array_pos.reshape(positive.shape)
    normalized_neg = normalized_array_neg.reshape(negative_aps.shape)

    grayscale_image_pos = normalized_positive / 3.0
    grayscale_image_neg = normalized_neg / 3.0

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

    # Return the processed data as a dictionary
    return {
        'grayscale_pos_thresholded': grayscale_pos_thresholded.tolist(),
        'grayscale_neg_thresholded': grayscale_neg_thresholded.tolist()
    }

if __name__ == '__main__':
    # Read input JSON from stdin
    input_data = json.loads(sys.stdin.read())

    # Process the input data
    output_data = process_input(input_data)

    # Write the output JSON to stdout
    sys.stdout.write(json.dumps(output_data))
