from flask import Flask, render_template, request, jsonify  
import requests
from bs4 import BeautifulSoup
import re
import torch
from transformers import BertTokenizer, BertForQuestionAnswering

def get_soup(keyword):
    keyword = keyword.replace(" ", "_")
    url = "https://simple.wikipedia.org/wiki/" + keyword
    response = requests.get(url)
    html_content = response.content
    soup = BeautifulSoup(html_content, "html.parser")
    raw_text = ""
    paragraphs = soup.find_all('p')
    for p in paragraphs:
        raw_text += p.get_text()
    text = re.sub(r'\[.*?\]|\(.*?\)', '', raw_text)
    text = re.sub(r'\b\n(?=[^\W\d_])', '', text)
    return text

def answer_question(context, question):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer.encode_plus(question, context, return_tensors='pt', max_length=512, truncation=True)

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        
    start_scores = torch.softmax(outputs.start_logits, dim=1).cpu().numpy()[0]
    end_scores = torch.softmax(outputs.end_logits, dim=1).cpu().numpy()[0]

    start_index = torch.argmax(outputs.start_logits)
    end_index = torch.argmax(outputs.end_logits)

    all_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].tolist()[0])

    answer_tokens = all_tokens[start_index:end_index+1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    confidence = (start_scores[start_index] + end_scores[end_index]) / 2

    return answer, confidence

def logic(question, text):
    answers_dict = {}
    num_parts = 10
    full_stops_indices = [i for i, char in enumerate(text) if char == '.']
    full_stops_per_part = len(full_stops_indices) // num_parts
    split_indices = [full_stops_indices[(idx+1)*full_stops_per_part] for idx in range(num_parts-1)]
    parts = [text[i:j] for i, j in zip([0] + split_indices, split_indices + [None])]

    for i, part in enumerate(parts):
        answer, confidence = answer_question(part, question)
        answers_dict[f"Chunk {i+1}"] = {answer :confidence}

        max_score = max(answers_dict.values(), key=lambda x: list(x.values())[0])
        max_value = list(max_score.keys())[0]
        
    print(answers_dict)
    print("max confidence:", list(max_score.values())[0])
    print("____________________________________________________________________________________")
    print(max_value)
    return max_value

app = Flask(__name__) 

@app.route("/", methods=['POST', 'GET']) 
def query_view(): 
    
    if request.method == 'POST': 
        keyword = request.form['keywordInput'] 
        question = request.form['prompt']
        text = get_soup(keyword)
        out = logic(question, text)
  
        return jsonify({'response': out })  
    return render_template('index.html') 
  
if __name__ == "__main__": 
    app.run(debug=True)
