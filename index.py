from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, render_template, request, redirect, url_for, send_from_directory


# Config
model_file = "templates/26_Multi_1e-6_250_Unfreeze.h5"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'


# Load model
model = load_model(model_file)
model.make_predict_function()


