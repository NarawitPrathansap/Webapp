# text_classifier.py
import sys
import json
from joblib import load
from transformers import BertTokenizer, BertModel
import torch

def classify_text(text, tokenizer, model, random_forest_model):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)

    # Obtain embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Use the [CLS] token's embedding for classification purposes
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()

    # Predict with Random Forest
    prediction = random_forest_model.predict(embeddings)
    prediction_proba = random_forest_model.predict_proba(embeddings)

    # Assuming binary classification for simplicity: [0, 1]
    # Modify as needed for your specific use case
    response = {
        "prediction": int(prediction[0]),
        "confidence": max(prediction_proba[0])
    }
    return response

if __name__ == '__main__':
    text = sys.argv[1]

    # Load models
    random_forest_model = load('../Webapp/templates/random_forest.joblib')
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')

    result = classify_text(text, tokenizer, bert_model, random_forest_model)
    print(json.dumps(result))
