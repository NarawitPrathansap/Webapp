from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
import os
import requests
import io
import shap
from langdetect import detect
from transformers import BertTokenizer, BertModel
import pickle
import torch
from joblib import load
import subprocess


app = Flask(__name__)

from efficientnet.layers import Swish, DropConnect
from efficientnet.model import ConvKernalInitializer
from tensorflow.keras.utils import get_custom_objects

get_custom_objects().update({
    'ConvKernalInitializer': ConvKernalInitializer,
    'Swish': Swish,
    'DropConnect':DropConnect
})

model_7_23 = load_model('../Webapp/templates/26_Multi_1e-6_250_Unfreeze.h5')
model_7_14 = load_model('../Webapp/templates/36_Multi_1e-5_500_Unfreeze.h5')
model_15_23 = load_model('../Webapp/templates/25_Multi_1e-6_500_Unfreeze.h5')
weights_path = '../Webapp/templates/best.pt'
#yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)


random_forest_model = load('../Webapp/templates/random_forest.joblib')

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')

app.config['UPLOAD_FOLDER'] = 'uploads/'


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# Preparing and pre-processing the image
def preprocess_img(img_path):
    img = Image.open(img_path)
    img_resize = img.resize((224, 224))
    img2arr = image.img_to_array(img_resize)
    img_reshape = img2arr.reshape((1,) + img2arr.shape)
    return img_reshape


def predict_result(img_array):
    predictions = model_7_23.predict(img_array)
    prediction_age = predictions[0]
    prediction_gender = predictions[1]

    # Assuming your model returns age as a continuous value and gender as a probability that needs argmax
    # Adjust these lines according to your model's actual output format
    age = prediction_age[0][0]
    age  = np.around(age)
    gender = prediction_gender[0][0]

    return age, gender


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            print('No file part')
            return redirect(request.url)
        image = request.files['image']
        question = request.form.get('question', '')
        if image.filename == '':
            print('No selected file')
            return redirect(request.url)
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)
            
            # Load the image
            img = Image.open(image_path)
            width, height = img.size
            frac = 0.6

            # Correct the crop method call, should use a tuple for the box
            # Crop 60% from the left of the image
            crop_left_width = int(width * frac)
            cropped_left = img.crop((0, 0, crop_left_width, height))
            left_filename = 'left_' + filename
            left_image_path = os.path.join(app.config['UPLOAD_FOLDER'], left_filename)
            cropped_left.save(left_image_path)

            # Crop 60% from the right of the image and flip it
            crop_right_width = width - crop_left_width
            cropped_right = img.crop((crop_right_width, 0, width, height))
            flipped_right_side = cropped_right.transpose(Image.FLIP_LEFT_RIGHT)
            right_filename = 'right_' + filename
            right_image_path = os.path.join(app.config['UPLOAD_FOLDER'], right_filename)
            flipped_right_side.save(right_image_path)

            # Generate URLs for the images
            image_url = url_for('uploaded_file', filename=filename)
            left_image_url = url_for('uploaded_file', filename=left_filename)
            right_image_url = url_for('uploaded_file', filename=right_filename)

            # Print the received question for debugging
            print("Received question:", question)
            # Preprocess both images
            left_image_array = preprocess_img(left_image_path)
            right_image_array = preprocess_img(right_image_path)
            prediction_age_left, gender_prob_left  = predict_result(left_image_array)
            prediction_age_right, gender_prob_right  = predict_result(right_image_array)
            
            # Adjust probabilities: If the model predicts female (prob < 0.5), adjust by doing 1 - prob
            # This way, for female predictions, a higher adjusted value (closer to 1) indicates higher confidence in the female prediction
            adjusted_prob_left = gender_prob_left if gender_prob_left >= 0.5 else 1 - gender_prob_left
            adjusted_prob_right = gender_prob_right if gender_prob_right >= 0.5 else 1 - gender_prob_right

            # Select the image with the highest adjusted probability
            if adjusted_prob_left >= adjusted_prob_right:
                selected_image_path = left_image_path
                selected_age_prediction = prediction_age_left
                selected_gender_prob = adjusted_prob_left
                selected_image = "select left" + filename
            else:
                selected_image_path = right_image_path
                selected_age_prediction = prediction_age_right
                selected_gender_prob = adjusted_prob_right
                selected_image = "select right" + filename
            print(f"Selected {selected_image} image with adjusted gender probability: {selected_gender_prob} and age prediction: {selected_age_prediction}")    
            # Determine which model to use based on age prediction
            if selected_age_prediction <= 14:
                final_model = model_7_14
            else:
                final_model = model_15_23

            selected_image_array = preprocess_img(selected_image_path)

            final_predictions = final_model.predict(selected_image_array)

            # If your final model outputs classification probabilities, you might do something like this:
            if final_model == model_7_14 or final_model == model_15_23:
                # Assuming a binary classification outcome as an example
                predicted_class_index = np.argmax(final_predictions[0], axis=-1)
                # Convert predicted_class_index to a meaningful label if applicable
                predicted_label = "Label1" if predicted_class_index == 0 else "Label2"
                print(f"Final prediction (classification): {predicted_label}")
                predicted_age = final_predictions[0][0]  # Assuming the prediction is the first element
                print(f"Final prediction (regression): {predicted_age}")
            selected_image_url = url_for('uploaded_file', filename=selected_image)

            # Render your template with the selected image URL
            return render_template('predict.html', 
                                image_url=image_url,
                                selected_image_url=selected_image_url,  # Add this
                                question=question,
                                prediction=predicted_age)


if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True,port=5001)#host='0.0.0.0',port=5001