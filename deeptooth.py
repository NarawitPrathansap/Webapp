from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
import os
import torch
import subprocess
import json
import shap
from keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from langdetect import detect

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

def get_tooth_parts(dataframe):
    # Join all 'name' values from the dataframe
    return ', '.join(dataframe['name'])


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

def compute_iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def nms_per_class(df, iou_threshold=0.5):
    # Initialize an empty DataFrame to store NMS results
    df_nms = pd.DataFrame()

    # Get unique class IDs
    class_ids = df['class'].unique()

    for class_id in class_ids:
        # Filter detections by class
        df_class = df[df['class'] == class_id].copy()

        # Apply NMS
        df_class_sorted = df_class.sort_values(by='confidence', ascending=False).reset_index(drop=True)
        suppressed_indices = set()

        for i in range(len(df_class_sorted)):
            if i in suppressed_indices:
                continue

            for j in range(i+1, len(df_class_sorted)):
                if j in suppressed_indices:
                    continue

                boxA = df_class_sorted.iloc[i][['xmin', 'ymin', 'xmax', 'ymax']]
                boxB = df_class_sorted.iloc[j][['xmin', 'ymin', 'xmax', 'ymax']]

                if compute_iou(boxA, boxB) > iou_threshold:
                    suppressed_indices.add(j)

        # Filter out suppressed detections
        df_nms_class = df_class_sorted.drop(index=list(suppressed_indices)).reset_index(drop=True)

        # Append results for this class to the main DataFrame
        df_nms = pd.concat([df_nms, df_nms_class], ignore_index=True)

    return df_nms



def detect(image_path):
    # Load the model
    weights_path = '../Webapp/templates/best.pt'
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    results = yolo_model(image_path)
    # Convert results to a pandas DataFrame
    df_results = results.pandas().xyxyn[0]
    # Apply NMS
    df_nms_filtered = nms_per_class(df_results) 
    # Convert NMS-filtered results to JSON and print
    print(df_nms_filtered.to_json(orient="records"))

    return df_nms_filtered
def plot_bboxes_on_image_pos(image_path, df, grayscale_image, output_path):
    selected_bboxes = []

    # Load the original image
    img = Image.open(image_path)
    img_array = np.array(img)

    # Get the image dimensions
    image_width, image_height = img.size

    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the original image
    ax.imshow(img)

    # Overlay the grayscale image with transparency
    ax.imshow(grayscale_image, cmap='Reds', alpha=0.5, extent=[0, image_width, image_height, 0])

    # Iterate over rows in the DataFrame
    for index, row in df.iterrows():
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        confidence, class_label,class_name = row['confidence'], row['class'],row['name']

        # Convert coordinates to absolute values
        abs_xmin = xmin * image_width
        abs_ymin = ymin * image_height
        abs_width = (xmax - xmin) * image_width
        abs_height = (ymax - ymin) * image_height

        # Map bounding box to the grayscale image
        # Calculate the region of interest in the grayscale image
        roi_xmin = int(xmin * grayscale_image.shape[1])
        roi_ymin = int(ymin * grayscale_image.shape[0])
        roi_xmax = int(xmax * grayscale_image.shape[1])
        roi_ymax = int(ymax * grayscale_image.shape[0])

        # Extract the region of interest from the grayscale image
        roi = grayscale_image[roi_ymin:roi_ymax, roi_xmin:roi_xmax]

        # Calculate the percentage of nonzero pixels
        nonzero_percentage = np.count_nonzero(roi) / (roi.shape[0] * roi.shape[1])

        # If the percentage of nonzero pixels exceeds 10%, collect the bounding box
        if nonzero_percentage > 0.1:
            # Add bounding box information to the selected_bboxes list
            selected_bboxes.append({'xmin': abs_xmin, 'ymin': abs_ymin,'xmax': abs_xmin + abs_width, 'ymax': abs_ymin + abs_height,
                                    'confidence': confidence, 'class': class_label,'name':class_name})

            # Create a rectangle patch
            rect = patches.Rectangle(
                (abs_xmin, abs_ymin),
                abs_width,
                abs_height,
                linewidth=2,
                edgecolor='r',
                facecolor='none'  # Set facecolor to 'none' for an unfilled rectangle
            )

            # Add the rectangle to the axes
            ax.add_patch(rect)
            # Add confidence and class label as text
            text = f'Class: {class_label}'
            #\nConfidence: {confidence:.2f}'
            plt.text(abs_xmin, abs_ymin - 10, text, color='black', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))


    plt.savefig(output_path)
    plt.close()

    return selected_bboxes




def plot_bboxes_on_image_neg(image_path, df, grayscale_image, output_path):
    selected_bboxes = []

    # Load the original image
    img = Image.open(image_path)
    img_array = np.array(img)

    # Get the image dimensions
    image_width, image_height = img.size

    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the original image
    ax.imshow(img)

    # Overlay the grayscale image with transparency
    ax.imshow(grayscale_image, cmap='Blues', alpha=0.5, extent=[0, image_width, image_height, 0])

    # Iterate over rows in the DataFrame
    for index, row in df.iterrows():
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        confidence, class_label,class_name = row['confidence'], row['class'],row['name']

        # Convert coordinates to absolute values
        abs_xmin = xmin * image_width
        abs_ymin = ymin * image_height
        abs_width = (xmax - xmin) * image_width
        abs_height = (ymax - ymin) * image_height

        # Map bounding box to the grayscale image
        # Calculate the region of interest in the grayscale image
        roi_xmin = int(xmin * grayscale_image.shape[1])
        roi_ymin = int(ymin * grayscale_image.shape[0])
        roi_xmax = int(xmax * grayscale_image.shape[1])
        roi_ymax = int(ymax * grayscale_image.shape[0])

        # Extract the region of interest from the grayscale image
        roi = grayscale_image[roi_ymin:roi_ymax, roi_xmin:roi_xmax]

        # Calculate the percentage of nonzero pixels
        nonzero_percentage = np.count_nonzero(roi) / (roi.shape[0] * roi.shape[1])

        # If the percentage of nonzero pixels exceeds 10%, collect the bounding box
        if nonzero_percentage > 0.1:
            # Add bounding box information to the selected_bboxes list
            selected_bboxes.append({'xmin': abs_xmin, 'ymin': abs_ymin,'xmax': abs_xmin + abs_width, 'ymax': abs_ymin + abs_height,
                                    'confidence': confidence, 'class': class_label,'name':class_name})

            # Create a rectangle patch
            rect = patches.Rectangle(
                (abs_xmin, abs_ymin),
                abs_width,
                abs_height,
                linewidth=2,
                edgecolor='b',
                facecolor='none'  # Set facecolor to 'none' for an unfilled rectangle
            )

            # Add the rectangle to the axes
            ax.add_patch(rect)
            # Add confidence and class label as text
            text = f'Class: {class_label}'
            #\nConfidence: {confidence:.2f}'
            plt.text(abs_xmin, abs_ymin - 10, text, color='black', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))
    plt.savefig(output_path)
    plt.close()

    return selected_bboxes
def get_auto_lang_answer(prediction_class, gender=None, age=None, selected_bboxes_pos=None, selected_bboxes_neg=None, question=''):
    try:
        detected_lang = detect(question)
        language = 'th' if detected_lang == 'th' else 'en'
    except Exception as e:
        print(f"Language detection failed: {e}")  # Logging the exception
        language = 'en'  # Default to English if detection fails

    # Combining tooth parts from both positive and negative bboxes if they exist
    tooth_parts = []
    if selected_bboxes_pos is not None:
        tooth_parts.extend(selected_bboxes_pos['name'].tolist())
    if selected_bboxes_neg is not None:
        tooth_parts.extend(selected_bboxes_neg['name'].tolist())
    tooth_parts_str = ', '.join(tooth_parts)    

    # Define answers database
    answers_db = {
        "en": {
            0: "The predicted gender from this panoramic image is {gender}.",
            1: "The estimated age of the individual in this panoramic image is {age} years.",
            2: "Based on the panoramic image, the predicted gender is {gender}, with attention to the {tooth_part}.",
            3: "From the panoramic analysis, the estimated age is {age} years, considering the {tooth_part}.",
            4: "Sorry, no answer available for this question."
        },
        "th": {
            0: "เพศที่คาดการณ์ไว้จากภาพพาโนรามานี้คือ {gender}.",
            1: "อายุที่ประเมินของบุคคลในภาพพาโนรามานี้คือ {age} ปี.",
            2: "ตามภาพพาโนรามา, เพศที่คาดการณ์ได้คือ {gender}, โดยเน้นที่ {tooth_part}.",
            3: "จากการวิเคราะห์ภาพพาโนรามา, อายุที่ประเมินได้คือ {age} ปี, โดยพิจารณาที่ {tooth_part}.",
            4: "ขออภัย, ไม่มีคำตอบสำหรับคำถามนี้."
        }
    }

    # Continue with your existing logic
    answer_template = answers_db[language].get(prediction_class, "")
    answer = answer_template.format(gender=gender, age=age, tooth_part=tooth_parts_str)
    
    return answer

@app.route('/predict', methods=['POST'])
def predict():
    img_paths = []  # Initialize img_paths at the beginning to ensure it's always defined
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
        prediction_class = subprocess.run(['python', 'clf.py', question], capture_output=True, text=True)
        
        
    
        
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
    
        shap_values_1 = np.array(shap_values)

        # Perform the processing as before
        image_array = shap_values_1[0]
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
        print(grayscale_image_neg.shape)
        grayscale_image_positive = np.mean(grayscale_image_pos, axis=-1)
        grayscale_image_negative = np.mean(grayscale_image_neg, axis=-1)


        grayscale_image_positive = grayscale_image_positive.squeeze()
        grayscale_image_negative = grayscale_image_negative.squeeze()

        percentile_95_pos = np.percentile(grayscale_image_positive, 95)
        percentile_95_neg = np.percentile(grayscale_image_negative, 95)

        grayscale_pos_thresholded = grayscale_image_positive
        grayscale_neg_thresholded = grayscale_image_negative

        grayscale_pos_thresholded[grayscale_pos_thresholded < percentile_95_pos] = 0
        grayscale_neg_thresholded[grayscale_neg_thresholded < percentile_95_neg] = 0




        output_path_pos = os.path.join(app.config['UPLOAD_FOLDER'], 'output_pos.png')
        output_path_neg = os.path.join(app.config['UPLOAD_FOLDER'], 'output_neg.png')

        # Proceed with detection and plotting
        df_yolo_results = detect(img)  # Make sure 'detect' returns a DataFrame with YOLO detection results

        # Assuming grayscale_pos_thresholded and grayscale_neg_thresholded are defined and ready to use
        selected_bboxes_pos = plot_bboxes_on_image_pos(img, df_yolo_results, grayscale_pos_thresholded, output_path_pos)
        selected_bboxes_neg = plot_bboxes_on_image_neg(img, df_yolo_results, grayscale_neg_thresholded, output_path_neg)
    # Convert server paths to web-accessible URLs
        output_url_pos = url_for('uploaded_file', filename='output_pos.png')
        output_url_neg = url_for('uploaded_file', filename='output_neg.png')

        # Depending on the prediction, generate an answer and choose the correct template and parameters
        if prediction_class == 0 or prediction_class == 1:
            gender_or_age = gender_ans if prediction_class == 0 else age_ans
            answer = get_auto_lang_answer(prediction_class, gender=gender_or_age, selected_bboxes_pos=selected_bboxes_pos, selected_bboxes_neg=selected_bboxes_neg, question=question)
            return render_template('predict2.html', image_url=image_url, question=question, answer=answer)

        elif prediction_class == 2:
            # Assuming 'predictions_highCon_Gender' is defined
            selected_bboxes = selected_bboxes_pos if predictions_highCon_Gender >= 0.5 else selected_bboxes_neg
            output_url = output_url_pos if predictions_highCon_Gender >= 0.5 else output_url_neg
            answer = get_auto_lang_answer(prediction_class, gender=gender_ans, selected_bboxes_pos=selected_bboxes, question=question)
            return render_template('predict.html', image_url=image_url, question=question, answer=answer, output_url=output_url)

        elif prediction_class == 3:
            # Assuming 'age_ans' is defined
            output_url = output_url_neg if age_ans <= 14 else output_url_pos
            answer = get_auto_lang_answer(prediction_class, age=age_ans, selected_bboxes_pos=selected_bboxes_pos, question=question)
            return render_template('predict.html', image_url=image_url, question=question, answer=answer, output_url=output_url)



        elif prediction_class == 4:
        # Correctly call `get_auto_lang_answer` and assign its return value to `answer`
            answer = get_auto_lang_answer(prediction_class, question=question)
        # Use the `answer` variable in the call to `render_template`
            return render_template('predict3.html', image_url=image_url, question=question, answer=answer)


                           

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True,port=5001)#host='0.0.0.0',port=5001