from transformers import LongformerForSequenceClassification, LongformerTokenizerFast
import pandas as pd
import datasets
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, Trainer, TrainingArguments, LongformerConfig
import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import os
import json
import random
from datasets import ClassLabel, Dataset, Features, Value
import evaluate

with open("dataset.json", "r", encoding='utf-8') as json_file:
    data = json.load(json_file)

data_test = data['test']

# Load the trained model
model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', device_map="auto", num_labels=2)
tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')


# Define a function to tokenize a single text
def tokenize_text(text):
    return tokenizer(text, padding='max_length', truncation=True, max_length=4096, return_tensors="pt")


texts = [item['text'] for item in data_test]
labels = [item['label'] for item in data_test]

# Visualizing or using the extracted texts :
predictions = []
i=0
for text, true_label in zip(texts, labels):
    # Tokenizing the text
    i+=1
    inputs = tokenize_text(text)
    inputs = {k: v.to(model.device) for k, v in inputs.items()} 

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Gain of the probabilities
    probs = torch.softmax(outputs.logits, dim=-1)
    predicted_labels = torch.argmax(probs, dim=-1).cpu().numpy()
    
    # Adding th prediction to the list
    predicted_label = predicted_labels[0]
    predictions.append(predicted_label)

with open('pred_labels.csv', 'w', newline='') as pred_file:
    fieldnames = ['predictions', 'vrai_labels']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for pred, vrai_label in zip(predictions, labels):
        writer.writerow({'predictions': pred, 'vrai_labels': vrai_label})