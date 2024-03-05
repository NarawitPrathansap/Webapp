import sys
import json
from joblib import load
from transformers import BertTokenizer, BertModel
import torch
from langdetect import detect

# Define the answers database globally
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

def classify_question(text, tokenizer, model, random_forest_model):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding='max_length')
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.pooler_output.numpy()
    prediction = random_forest_model.predict(embeddings)
    prediction_proba = random_forest_model.predict_proba(embeddings)
    response = {
        "prediction": int(prediction[0]),
        "confidence": float(max(prediction_proba[0]))
    }
    return response

def fetch_answer(category, lang):
    # Placeholder for dynamic content based on prediction, such as 'gender' or 'age'
    dynamic_content = {"gender": "male", "age": "30", "tooth_part": "molar"}
    template = answers_db[lang][category]
    answer = template.format(**dynamic_content)
    return answer

def handle_question(question_text, tokenizer, bert_model, random_forest_model):
    question_lang = detect(question_text)
    result = classify_question(question_text, tokenizer, bert_model, random_forest_model)
    category = result['prediction']
    answer = fetch_answer(category, question_lang)
    response = {
        "language": question_lang,
        "category": category,
        "answer": answer
    }
    return response

if __name__ == '__main__':
    text = sys.argv[1] if len(sys.argv) > 1 else "This is a sample question."
    random_forest_model = load('../Webapp/templates/random_forest.joblib')  # Correct the path as needed
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')

    # Use handle_question function if you want to process the question and get an answer.
    # result = handle_question(text, tokenizer, bert_model, random_forest_model)
    # For direct classification and debugging:
    result = classify_question(text, tokenizer, bert_model, random_forest_model)
    print(json.dumps(result))
