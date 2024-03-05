import pickle
from flask import Flask, render_template, request, redirect, url_for, send_from_directory



app = Flask(__name__)

try:
    with open('../Webapp/templates/random_forest_model_real1.pkl', 'rb') as file:
        random_forest_model = pickle.load(file)
except Exception as e:
    print(f"Error loading the pickle file: {e}")

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True,port=5001)#host='0.0.0.0',port=5001