import sys
import json
from joblib import load
from transformers import BertTokenizer, BertModel
import torch

random_forest_model = load('../Webapp/templates/random_forest.joblib')  # Correct the path as needed
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')


def encode_text(text):
    # Tokenize and encode the text
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    outputs = bert_model(**inputs)
    # Use the pooled output for simplicity
    return outputs.pooler_output.detach().numpy()

def predict(text):
    # Encode the input text
    encoded_text = encode_text(text)
    # Predict with the Random Forest model
    prediction = random_forest_model.predict(encoded_text)
    return int(prediction[0])

if __name__ == '__main__':
    input_text = sys.argv[1]  # Text input from command line
    prediction = predict(input_text)
    # Output the prediction as a JSON string
    print(json.dumps({'prediction': prediction}))