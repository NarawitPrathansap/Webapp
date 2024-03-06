import sys
import json
from joblib import load
from transformers import BertTokenizer, BertModel
import torch

random_forest_model = load('../Webapp/templates/random_forest.joblib')  # Correct the path as needed
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')


def classify_question(text, tokenizer, bert_model, random_forest_model):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding='max_length')
    output = bert_model(**inputs)
    last_hidden_states = output.last_hidden_state
    cls_embeddings = last_hidden_states[:, 0, :].detach().numpy()
    predictions = random_forest_model.predict(cls_embeddings)
    return predictions[0]

if __name__ == '__main__':
    text = " ".join(sys.argv[1:])  # Improved handling for text input from command line arguments
    # Now correctly passing bert_model and random_forest_model in the correct order
    prediction = classify_question(text, tokenizer, bert_model, random_forest_model)

    # Output the prediction so it can be captured by subprocess.run
    print(json.dumps({"prediction": prediction.tolist()}))  # Ensure JSON serializability