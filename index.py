from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
import os


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

from efficientnet.layers import Swish, DropConnect
from efficientnet.model import ConvKernalInitializer
from tensorflow.keras.utils import get_custom_objects

get_custom_objects().update({
    'ConvKernalInitializer': ConvKernalInitializer,
    'Swish': Swish,
    'DropConnect':DropConnect
})

# Load the model
model1 = tf.keras.models.load_model('templates/26_Multi_1e-6_250_Unfreeze.h5')


# Preparing and pre-processing the image
def preprocess_img(img_path):
    img = Image.open(img_path)
    img_resize = img.resize((224, 224))
    img2arr = image.img_to_array(img_resize)
    img_reshape = img2arr.reshape((1,) + img2arr.shape)
    return img_reshape


def predict_result(img_array):
    predictions = model1.predict(img_array)
    prediction_age = predictions[0]
    prediction_gender = predictions[1]

    # Assuming your model returns age as a continuous value and gender as a probability that needs argmax
    # Adjust these lines according to your model's actual output format
    age = prediction_age[0] # Assuming the first prediction is age as a continuous value
    gender = np.argmax(prediction_gender[0], axis=-1)  # Assuming the second prediction is gender as a binary classification

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
            prediction = "Dummy prediction result"  # Replace with your model's prediction logic
            # Preprocess both images
            left_image_array = preprocess_img(left_image_path)
            right_image_array = preprocess_img(right_image_path)
            prediction_age1, prediction_gender1 = predict_result(left_image_array)
            prediction_age2, prediction_gender2 = predict_result(right_image_array)
            # Render the result template with the image URLs
            return render_template('result.html', 
                                   image_url=url_for('uploaded_file', filename=filename),
                                   #right_image_url=url_for('uploaded_file', filename=right_filename),
                                   question=question,
                                   prediction=prediction 
                                   #prediction_age1=prediction_age1, 
                                   #prediction_gender1=prediction_gender1,
                                   #prediction_age2=prediction_age2, 
                                   #prediction_gender2=prediction_gender2)
            )



if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True,port=5001)#host='0.0.0.0',port=5001