import csv
import Levenshtein
from rouge_score import rouge_scorer
import re
import json
from sklearn.model_selection import train_test_split

llama_file = "your_dataset_here.csv"

true_texts = []
generations = []
with open(llama_file, 'r', newline='', encoding='utf-8-sig') as file:
    reader = csv.DictReader(file)
    for row in reader:
        true_texts.append(row["true_text"])
        generations.append(row["generation"])

# Initialize lists for scores and distances
distances = []
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_scores = []
word_overlaps = []

# Calculate Levenshtein distance, ROUGE scores, and word overlap
for gen, text in zip(generations, true_texts):
    # Levenshtein distance
    distance = Levenshtein.distance(text, gen)
    distances.append(distance)

    # ROUGE scores
    scores = scorer.score(text, gen)
    rouge_scores.append(scores)

    # Word overlap
    extracted_words = set(re.findall(r'\w+', text.lower()))
    generated_words = set(re.findall(r'\w+', gen.lower()))
    common_words = extracted_words.intersection(generated_words)
    word_overlap_percentage = (len(common_words) / len(extracted_words)) * 100 if len(extracted_words) > 0 else 0.0
    word_overlaps.append(word_overlap_percentage)

# Calculate ROUGE-1 F-measures
rouge_1_f_list = []
for idx, scores in enumerate(rouge_scores):
    rouge_1 = scores['rouge1']
    rouge_1_f = rouge_1.fmeasure
    rouge_1_f_list.append(rouge_1_f)

# Combine data for sorting
data = list(zip(true_texts, generations, distances, rouge_1_f_list, word_overlaps))

# Filter data where the Levenshtein distance is greater than 300
filtered_data = [item for item in data if item[2] > 300]

# Sort filtered data by ROUGE-1 F-measure in descending order
data_sorted = sorted(filtered_data, key=lambda x: x[3], reverse=True)

# Keep the top 100 generated examples
top_100 = data_sorted[:100]

# Remove the top 100 from the complete dataset
data_remaining = [item for item in data if item not in top_100]

# Calculate the total number of items to get the correct distribution
total_items = len(true_texts) + len(generations)
train_size = int(0.4 * total_items)  # 40% for training
test_size = int(0.1 * total_items)   # 10% for testing
print(train_size)
print(test_size)

# Create lists with the remaining true and generated texts, removing duplicates
remaining_true_texts = [{"text": item[0], "label": 1} for item in data_remaining if item[0] not in generations]
remaining_generated_texts = [{"text": item[1], "label": 0} for item in data_remaining if item[1] not in true_texts]

# Ensure the top 100 generated examples are in the test set
test_data = [{"text": gen_text, "label": 0} for _, gen_text, _, _, _ in top_100]

# Determine how many examples of true and generated texts should be added to the training and test sets
remaining_train_true, remaining_test_true = train_test_split(
    remaining_true_texts, test_size=test_size, random_state=42)

# Splitting remaining generated texts into training and testing sets
remaining_train_generated, remaining_test_generated = train_test_split(
    remaining_generated_texts, test_size=test_size, random_state=42)

# Combine true and generated texts for training
train_data = remaining_train_true + remaining_train_generated

# Add the remaining true texts to the test set
test_data.extend(remaining_test_true)

# Add the remaining generated texts to the test set
test_data.extend(remaining_test_generated)

# Ensure the training data doesn't contain duplicates with the test set
train_data = [item for item in train_data if item not in test_data]

# Save train and test data as JSON
with open('train_data.json', 'w', encoding='utf-8') as train_file:
    json.dump(train_data, train_file, ensure_ascii=False, indent=4)

with open('test_data.json', 'w', encoding='utf-8') as test_file:
    json.dump(test_data, test_file, ensure_ascii=False, indent=4)

print("Data has been split and saved as 'train_data.json' and 'test_data.json'.")

