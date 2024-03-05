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


from tensorflow.keras.preprocessing import image
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

# img_path รับจากข้างนอก  2 ภาพ ??????????????????????????????????????????????????????????????????/

pred_list_regression = []  # Store regression results
pred_list_classification = []  # Store binary classification results

##img_path = test['Path_Name'].tolist()

for i in range(len(img_path)):
    predictions = predict_image(img_path[i], model_7_23, height, width)

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

#หาค่า confident
def calculate_confident(value):
    if value >= 0.5: #male
        confident = value
    else:
        confident = 1 - value #female
    return confident

con_0 = calculate_confident(list_Classification_predict[0])
con_1 = calculate_confident(list_Classification_predict[1])

# ดูว่าค่าไหนมากที่สุด
max_confident = max(con_0, con_1)
print("Maximum confident value:", max_confident)

# เลือกภาพ
if con_0 > con_1:
    img = # path ภาพแรก
elif con_1 > con_0:
    img = # path ภาพสอง
else: # ถ้าเท่ากัน 
    img = # path ภาพแรก


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



predictions_highCon = predict_image(img, model, height, width) # ภาพที่เลือก

# Access the regression result (output 0)
predictions_highCon_Age = predictions_highCon[0]

# Access the classification result (output 1)
predictions_highCon_Gender = predictions_highCon[1] # Use a threshold to determine the class
