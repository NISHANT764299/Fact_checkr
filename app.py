from flask import Flask, request, render_template, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-news-finetuned")
model = BertForSequenceClassification.from_pretrained("bert-news-finetuned")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        text = request.form["news"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            result = "🟢 Real News" if prediction == 1 else "🔴 Fake News" 

    return render_template("index.html", result=result)

@app.route("/sources")
def sources():
    return render_template("sources.html")

if __name__ == "__main__":
    app.run(debug=True)
