import csv
import textstat
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
from transformers import BertTokenizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import pyphen
import pandas as pd

# Downloading stopwords
nltk.download('punkt')
nltk.download('stopwords')

# Load French stopwords
stop_words = set(stopwords.words('french'))

# Add specific stopwords related to your corpus
additional_stop_words = {"de", "la", "et", "à", "le", "les", "des", "en", "un", "une", "du", "pour", "avec", "dans", "sur", "est", "sont", "ce", "se", "ne", "pas", "qui", "que", "où", "au", "par", "(", ")", "'", "a"}
stop_words.update(additional_stop_words)

# Example CSV files
filenames = ['command_r_35B_one_shot_gen.csv', 'llama_3_8B_one_shot_gen.csv', 'llama_3_70B_one_shot_gen.csv', 'command_r_plus_4bit_one_shot_gen.csv']
models = ['Command-R-35B', 'Llama-3-8B', 'Llama-3-70B', 'Command-R-plus-4bit']

# Prepare the data
all_subtokens_per_sentence = {model: [] for model in models}
all_sentence_lengths_words_all = {model: [] for model in models}
all_num_sentences_all = {model: [] for model in models}

for file, model in zip(filenames, models):
    with open(file, newline='', encoding='utf-8') as f:
        sentences = []
        cases = list(csv.DictReader(f))
        for case in cases:
            case_to_generate = case.get('generation', 'N/A')
            sentences.append(case_to_generate)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def getAverageSubtokensPerSentence(sentences, tokenizer):
        subtokens_per_sentence = []
        for sentence in sentences:
            subtokens = tokenizer.tokenize(sentence)
            subtokens_per_sentence.append(len(subtokens))
        return subtokens_per_sentence

    subtokens_per_sentence = getAverageSubtokensPerSentence(sentences, tokenizer)
    all_subtokens_per_sentence[model] = subtokens_per_sentence

    sentence_stats = []
    for sentence in sentences:
        # Split each "sentence" into phrases
        phrases = nltk.sent_tokenize(sentence)
        
        # Number of phrases
        num_phrases = len(phrases)
        
        # Length of each phrase in terms of words
        phrase_lengths_words = [len(nltk.word_tokenize(phrase)) for phrase in phrases]
        
        # Length of each phrase in terms of characters
        phrase_lengths_chars = [len(phrase) for phrase in phrases]
        
        # Length of the longest phrase (in words)
        max_length_words = max(phrase_lengths_words) if phrase_lengths_words else 0
        
        # Length of the shortest phrase (in words)
        min_length_words = min(phrase_lengths_words) if phrase_lengths_words else 0
        
        # Average phrase length (in words)
        avg_length_words = sum(phrase_lengths_words) / num_phrases if num_phrases > 0 else 0
        
        # Add sentence statistics to the list
        sentence_stats.append({
            'num_phrases': num_phrases,
            'max_length_words': max_length_words,
            'min_length_words': min_length_words,
            'avg_length_words': avg_length_words,
            'phrase_lengths_words': phrase_lengths_words,
            'phrase_lengths_chars': phrase_lengths_chars
        })

    sentence_lengths_words_all = [length for stat in sentence_stats for length in stat['phrase_lengths_words']]
    all_sentence_lengths_words_all[model] = sentence_lengths_words_all

    num_sentences_all = [stat['num_phrases'] for stat in sentence_stats]
    all_num_sentences_all[model] = num_sentences_all

# Create DataFrame for visualizations
df_sentence_lengths = pd.DataFrame({
    'Sentence Length (in words)': [],
    'Model': []
})

df_num_sentences = pd.DataFrame({
    'Number of Sentences': [],
    'Model': []
})

df_list = pd.DataFrame({
            'Subtokens': all_subtokens_per_sentence[model],
            'Model': model
        })

for model in models:
    df_sentence_lengths = pd.concat([df_sentence_lengths, 
                                     pd.DataFrame({
                                         'Sentence Length (in words)': all_sentence_lengths_words_all[model],
                                         'Model': model
                                     })])
    
    df_num_sentences = pd.concat([df_num_sentences, 
                                  pd.DataFrame({
                                      'Number of Sentences': all_num_sentences_all[model],
                                      'Model': model
                                  })])
    
    df_list = pd.concat([df_list,
                         pd.DataFrame({
                             'Subtokens': all_subtokens_per_sentence[model],
                             'Model': model
                         })])

# Check unique values in 'Model'
print(df_sentence_lengths['Model'].unique())
print(df_num_sentences['Model'].unique())

plt.figure(figsize=(10, 6))
sns.boxplot(x='Model', y='Subtokens', data=df_list[df_list['Subtokens'].notnull()], palette="tab10")
plt.title('Number of Tokens per Medical Case Report')
plt.xlabel('Model')
plt.ylabel('Number of Tokens')
plt.xticks(rotation=30, fontsize=12)
plt.tight_layout()
plt.show()

# 1. Histogram of sentence lengths (in words) for two groups of models
plt.figure(figsize=(12, 6))
sns.histplot(data=df_sentence_lengths[df_sentence_lengths['Model'].isin(['Command-R-35B', 'Command-R-plus-4bit'])], 
             x='Sentence Length (in words)', 
             hue='Model', 
             hue_order=['Command-R-35B', 'Command-R-plus-4bit'],
             bins=30, 
             kde=True)
plt.title('Distribution of Sentence Lengths (in words) for Command-R')
plt.xlabel('Number of words per sentence')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(data=df_sentence_lengths[df_sentence_lengths['Model'].isin(['Llama-3-8B', 'Llama-3-70B'])], 
             x='Sentence Length (in words)', 
             hue='Model', 
             hue_order=['Llama-3-8B', 'Llama-3-70B'],
             bins=30, 
             kde=True)
plt.title('Distribution of Sentence Lengths (in words) for Llama-3')
plt.xlabel('Number of words per sentence')
plt.ylabel('Frequency')
plt.show()

# 2. Distribution of the number of sentences per report for two groups of models
max_num_sentences = int(df_num_sentences['Number of Sentences'].max()) + 1

plt.figure(figsize=(12, 6))
sns.histplot(data=df_num_sentences[df_num_sentences['Model'].isin(['Command-R-35B', 'Command-R-plus-4bit'])], 
             x='Number of Sentences', 
             hue='Model', 
             hue_order=['Command-R-35B', 'Command-R-plus-4bit'],
             bins=range(1, max_num_sentences), 
             kde=True)
plt.title('Distribution of the Number of Sentences per Report for Command-R')
plt.xlabel('Number of Sentences')
plt.ylabel('Number of Reports')
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(data=df_num_sentences[df_num_sentences['Model'].isin(['Llama-3-8B', 'Llama-3-70B'])], 
             x='Number of Sentences', 
             hue='Model', 
             hue_order=['Llama-3-8B', 'Llama-3-70B'],
             bins=range(1, max_num_sentences), 
             kde=True)
plt.title('Distribution of the Number of Sentences per Report for Llama-3')
plt.xlabel('Number of Sentences')
plt.ylabel('Number of Reports')
plt.show()

