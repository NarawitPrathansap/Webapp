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

        age_ans = np.around(predictions_highCon_Age) # array
        age_ans = age_ans.to_numpy()# อายุที่จะเอาไปตอบบทแชท
        

        # Access the classification result (output 1)
        predictions_highCon_Gender = predictions_highCon[1][0][0] # Use a threshold to determine the class
        
        if predictions_highCon_Gender  >= 0.5: #male
           gender_ans = "Male"
        else:
           gender_ans = "Female"





    return render_template('predict.html', image_url=image_url, selected_image_url=selected_image_url, question=question, predicted_age=age_ans,predictions_gender=gender_ans)



if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True,port=5001)#host='0.0.0.0',port=5001