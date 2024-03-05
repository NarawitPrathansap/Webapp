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
import pickle
import torch
import subprocess
import json
import shap
from keras.preprocessing.image import load_img, img_to_array

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

# 7-23
height = width = model_7_23.input_shape[1]

def predict_image(img_path,model, height, width):
    # Read the image and resize it
    img = image.load_img(img_path, target_size=(height, width))
    # Convert it to a Numpy array with target shape.
    x = image.img_to_array(img)
    # Reshape
    x = x.reshape((1,) + x.shape)
    x /= 255.
    result = model.predict([x])

    return result

#หาค่า confident
def calculate_confident(value):
    if value >= 0.5: #male
        confident = value
    else:
        confident = 1 - value #female
    return confident



app.config['UPLOAD_FOLDER'] = 'uploads/'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def index():
    return render_template('index.html')

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

# Convert background data to numpy array
background_train_np = np.array(background_train)


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
            # Define the output filenames
            left_image_filename = 'left_' + filename
            right_image_filename = 'right_' + filename
            left_image_path = os.path.join(app.config['UPLOAD_FOLDER'], left_image_filename)
            right_image_path = os.path.join(app.config['UPLOAD_FOLDER'], right_image_filename)

            # Call the cut_image.py script as a subprocess
            subprocess.run(['python', 'index.py', image_path, left_image_path, right_image_path])



            # img_path รับจากข้างนอก  2 ภาพ ??????????????????????????????????????????????????????????????????/1
            # Assume img_path_1 and img_path_2 are the paths to your images
            img_paths = [left_image_path, right_image_path]  # Using the results from cut_image.py


            pred_list_regression = []  # Store regression results
            pred_list_classification = []  # Store binary classification results

        ##img_path = test['Path_Name'].tolist()

        for i in range(len(img_paths)):
            predictions = predict_image(img_paths[i], model_7_23, height, width)

            # Access the regression result (output 0)
            regression_result = predictions[0]

            # Access the classification result (output 1)
            classification_result = predictions[1] # Use a threshold to determine the class

            pred_list_regression.append(regression_result)
            pred_list_classification.append(classification_result)


        # Gender prediction
        list_Classification_predict = []
        for i in pred_list_classification:
            i = i[0][0]
            list_Classification_predict.append(i)


        con_0 = calculate_confident(list_Classification_predict[0])
        con_1 = calculate_confident(list_Classification_predict[1])

        # ดูว่าค่าไหนมากที่สุด
        max_confident = max(con_0, con_1)
        print("Maximum confident value:", max_confident)

        # Choose an image based on confidence values
        if con_0 > con_1:
            img = left_image_path  # Path to the first (left) image
        elif con_1 > con_0:
            img = right_image_path  # Path to the second (right) image
        else:  # If confidence values are equal
            img = left_image_path  # Default to the first (left) image path


        # ทำนายค่าอายุของภาพที่ลือก
        # Age prediction
        if con_0 > con_1:
            predict_age = pred_list_regression[0][0][0]
        elif con_1 > con_0:
            predict_age = pred_list_regression[1][0][0]
        else: # ถ้าเท่ากัน 
            predict_age = pred_list_regression[0][0][0]

        age_predict = np.around(predict_age) # array
        age_predict # ได้ค่าอายุมาแล้ว


        # เลือกตัวแบบ
        if age_predict <= 14:
            model = model_7_14
            print('Age between 7-14 years')
        else: 
            model = model_15_23
            print('Age between 15-23 years')

        image_url = url_for('uploaded_file', filename=filename) # For the original uploaded image
        selected_image_url = url_for('uploaded_file', filename=os.path.basename(img)) # For the selected image after processing

        predictions_highCon = predict_image(img, model, height, width) # ภาพที่เลือก
        
        # Access the regression result (output 0)
        predictions_highCon_Age = predictions_highCon[0][0][0] # [0] บอกว่าดึงจาก layer ไหน [0][0] ถอด[[]]ออก

        age_ans = int(np.around(predictions_highCon_Age)) # array
        

        # Access the classification result (output 1)
        predictions_highCon_Gender = predictions_highCon[1][0][0] # Use a threshold to determine the class
        
        if predictions_highCon_Gender  >= 0.5: #male
           gender_ans = "Male"
        else:
           gender_ans = "Female"

        # Run the classification model using subprocess
        classification_result = subprocess.run(['python', 'clf.py', question], capture_output=True, text=True)
        classification_output = classification_result.stdout
        classification_response = json.loads(classification_output)
        prediction_class = classification_response.get('prediction')
        
        if prediction_class == 0:
            answer = gender_ans
        if prediction_class == 1:
            answer = age_ans
        if prediction_class == 2:
            answer = gender_ans +"Where"
        if prediction_class == 3:
            answer = age_ans + "Where"
        if prediction_class == 4:
            answer = "Sorry, no answer available for this question."
        
    # Assuming `background_user_upload_image` is the path to the user uploaded image
    background_user_upload_image = img

    # Load the user uploaded image
    user_uploaded_image = load_img(background_user_upload_image, target_size=(224, 224))

    # Preprocess the image
    preprocessed_user_uploaded_image = img_to_array(user_uploaded_image) / 255.0

    # Reshape the image to match the model input shape
    reshaped_user_uploaded_image = np.expand_dims(preprocessed_user_uploaded_image, axis=0)

    # Calculate SHAP values
    shap_values = explainer15_23_gender.shap_values(reshaped_user_uploaded_image)
    
    grey = subprocess.run(['python', 'grayscale.py'], input=json.dumps(shap_values), capture_output=True, text=True)

    # Check if we got output and parse it
    if grey.stdout:
        output_data = json.loads(grey.stdout)
        grayscale_image_path = output_data.get('grayscale_image_path', "")
        # Process grayscale_neg_thresholded and grayscale_pos_thresholded if needed
    else:
        grayscale_image_path = ""
        print("Error running grayscale.py or no grayscale image generated")
 

    # Call the YOLOv5 detection script as a subprocess
    detect = subprocess.run(['python', 'yolo.py', img], capture_output=True, text=True)
        # Assuming yolo.py writes a CSV file and outputs its path
    if detect.stdout:
        csv_path = detect.stdout.strip()  # Extract the CSV path from stdout
    else:
        csv_path = ""
        print("Error running yolo.py or no CSV generated")

    plot_yolo_greyscale_image = subprocess.run(['python', 'shap_yolo.py', img,csv_path,grayscale_image_path], capture_output=True, text=True)
    


    return render_template('predict.html', image_url=image_url, selected_image_url=selected_image_url, question=question, predicted_age=age_ans,predictions_gender=gender_ans,shap_values=shap_values,answer_true=answer
                           img_url=url_for('uploaded_file', filename='output_' + filename))
                           




if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True,port=5001)#host='0.0.0.0',port=5001