import pandas as pd
import datasets
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, Trainer, TrainingArguments, LongformerConfig
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import os
import json
import random
import evaluate

if torch.cuda.is_available():
    print(f"CUDA is available! Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        gpu_info = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {gpu_info.name} with {gpu_info.total_memory // (1024 ** 2)} MB memory")
else:
    print("CUDA is not available. Training will be done on the CPU.")

config = LongformerConfig()

config

with open("dataset_llama.json", "r", encoding='utf-8') as json_file:
    data = json.load(json_file)

# Extract training data
train_data = data['train']

# Convert data lists to DataFrame
df_train = pd.DataFrame(train_data)

# Split the data into training and validation sets (80% train, 20% validation)
train_df, val_df = train_test_split(df_train, test_size=0.2, random_state=42)

# Convert DataFrames to Hugging Face Dataset objects
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Define labels and convert them
unique_labels = ['ai', 'human']
num_classes = len(unique_labels)
class_label = ClassLabel(num_classes=num_classes, names=unique_labels)

# Convert 'label' to ClassLabel format
def map_labels(example):
    # Ensure labels are already integers
    if isinstance(example['label'], str):
        example['label'] = class_label.str2int(example['label'])
    return example

train_dataset = train_dataset.map(map_labels)
val_dataset = val_dataset.map(map_labels)

train_dataset = train_dataset.remove_columns('__index_level_0__')
val_dataset = val_dataset.remove_columns('__index_level_0__')

# Update features to use ClassLabel
features = Features({
    'text': Value(dtype='string'),
    'label': class_label
})

train_dataset = train_dataset.cast(features)
val_dataset = val_dataset.cast(features)

# Load model and tokenizer
model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', num_labels=num_classes, device_map="auto")
tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', max_length=4096)

# Tokenization function
def tokenization(batched_text):
    return tokenizer(batched_text['text'], padding='max_length', truncation=True, max_length=4096)

# Apply tokenization
train_data = train_dataset.map(tokenization, batched=True)
val_data = val_dataset.map(tokenization, batched=True)

# Prepare metrics
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# Training arguments
training_args = TrainingArguments(
    output_dir="./longformer_results",
    num_train_epochs=5,  # Adjust based on needs
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=4,
    save_strategy="epoch",
    disable_tqdm=False,
    load_best_model_at_end=True,
    warmup_steps=500,
    weight_decay=0.015,
    logging_steps=2,
    fp16=True,
    logging_dir='./longformer_logs',
    dataloader_num_workers=0,
    evaluation_strategy="epoch",
    learning_rate=2e-5
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()

