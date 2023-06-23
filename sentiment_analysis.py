import os
import logging

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from cloudevents.http import CloudEvent
from cloudevents.conversion import to_binary

import requests

from flask import Flask, request, jsonify

model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'

TRANSFORMERS_CACHE = os.environ['TRANSFORMERS_CACHE']
reviews_sentiment_sink = os.environ['reviews_sentiment_sink']
attributes = {
    "type": os.environ['ce_type'],
    "source": os.environ['ce_source']
}

def create_app():
    global tokenizer
    global model
    global device

    app = Flask(__name__)
    app.logger.setLevel(logging.INFO)

    app.logger.info("starting app")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    app.logger.info("Model loaded")

    return app

app = create_app()

@app.route("/status")
def status():
    return jsonify({"status": "ok"}), 200

@app.route('/analyze', methods = ['POST'])
def process():
    json_payload = request.json

    app.logger.info("Input: " + str(json_payload))

    try:
        review_text = json_payload['review_text']
    except KeyError:
        app.logger.error("Not valid data input syntax")
        return 'bad request', 400
    inputs = tokenizer(review_text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    inputs = inputs.to(device)

    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=1).detach().cpu().numpy()
    sentiment = int(predictions.argmax(axis=1)[0]) - 1  # Convert 0-4 to -1-3
    response = f"{'positive' if sentiment > 0 else 'negative' if sentiment < 0 else 'neutral'}"

    sentiment_data = json_payload
    sentiment_data['score'] = sentiment
    sentiment_data['response'] = response

    event = CloudEvent(attributes, sentiment_data)
    headers, body = to_binary(event)

    requests.post(reviews_sentiment_sink, data=body, headers=headers)

    return '', 204
