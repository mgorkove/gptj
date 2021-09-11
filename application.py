from flask import Flask
from flask import request
from transformers import pipeline
import json

application = Flask(__name__)
classifier = pipeline("zero-shot-classification",model="facebook/bart-large-mnli")

def get_score(text):
    candidate_labels = ['very positive','positive','neutral','negative','very negative']
    labels_scores_mapping = {'very positive':2,'positive':1,'neutral':0,'negative':-1,'very negative':-2}
    label = classify(text, candidate_labels)
    return labels_scores_mapping[label]


def get_category(text):
    candidate_labels = ['compensation', 'manager', 'career advancement', 'culture', 'work-life balance', 'recognition', 'other']
    label = classify(text, candidate_labels)
    return label

def classify(text, labels):
    scores = classifier(text, labels)
    ordered_labels = scores['labels']
    return ordered_labels[0]


@application.route('/classify', methods = ['POST'])
def classify_message():
    # returns score and category of message
    text = request.args.get('text')
    score = get_score(text)
    category = get_category(text)
    return json.dumps({"score": score, "category": category})

if __name__ == "__main__":
    application.run()

    

