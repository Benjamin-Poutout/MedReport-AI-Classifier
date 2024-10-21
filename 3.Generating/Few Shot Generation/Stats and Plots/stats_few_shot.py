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

nltk.download('punkt')
nltk.download('stopwords')

# Load French stop words
stop_words = set(stopwords.words('french'))

# Additional stop words (beyond those in NLTK)
additional_stop_words = {"de", "la", "et", "à", "le", "les", "des", "en", "un", "une", "du", "pour", "avec", "dans", "sur", "est", "sont", "ce", "se", "ne", "pas", "qui", "que", "où", "au", "par", "(", ")", "'", "a"}
stop_words.update(additional_stop_words)

filenames = ['command_r_35B_few_shot_gen.csv', 'llama_3_8B_five_shot_fin.csv', 'llama_3_70B_five_shot_gen_clean.csv', 'command_r_plus_4bit_few_shot.csv']

for file in filenames:
    try:
        with open(file, newline='', encoding='utf-8') as f:
            print(f"{file} opened successfully.")
    except FileNotFoundError as e:
        print(f"{file} not found: {e}")

for file in filenames:
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

    # Create a boxplot of subtokens per sentence
    plt.figure(figsize=(10, 6))
    sns.boxplot(subtokens_per_sentence)
    plt.title('Number of Tokens per Medical Case Report')
    plt.xlabel('Tokens')
    plt.show()

    print(min(subtokens_per_sentence))
    print(max(subtokens_per_sentence))

    # Find the index of the minimum
    min_index = subtokens_per_sentence.index(min(subtokens_per_sentence))

    # Find the index of the maximum
    max_index = subtokens_per_sentence.index(max(subtokens_per_sentence))

    print(f"The index of the minimum is {min_index}")
    print(f"The index of the maximum is {max_index}")

    sentences.pop(max_index)

    print(f"The dataset contains {len(sentences)} case reports.")

    subtokens_per_sentence = getAverageSubtokensPerSentence(sentences, tokenizer)
    print(min(subtokens_per_sentence))
    print(max(subtokens_per_sentence))

    # Find the index of the minimum
    min_index = subtokens_per_sentence.index(min(subtokens_per_sentence))

    # Find the index of the maximum
    max_index = subtokens_per_sentence.index(max(subtokens_per_sentence))

    print(f"The index of the minimum is {min_index}")
    print(f"The index of the maximum is {max_index}")

    sentences.pop(max_index)

    subtokens_per_sentence = getAverageSubtokensPerSentence(sentences, tokenizer)
    print(min(subtokens_per_sentence))
    print(max(subtokens_per_sentence))

    # Find the index of the minimum
    min_index = subtokens_per_sentence.index(min(subtokens_per_sentence))

    # Find the index of the maximum
    max_index = subtokens_per_sentence.index(max(subtokens_per_sentence))

    print(f"The index of the minimum is {min_index}")
    print(f"The index of the maximum is {max_index}")

    plt.figure(figsize=(10, 6))
    sns.boxplot(subtokens_per_sentence)
    plt.title('Number of Tokens per Medical Case Report')
    plt.xlabel('Tokens')
    plt.show()

    # Store the statistics for each sentence
    sentence_stats = []

    for sentence in sentences:
        # Split each sentence into phrases
        phrases = nltk.sent_tokenize(sentence)
        
        # Number of phrases
        num_phrases = len(phrases)
        
        # Length of each phrase in words
        phrase_lengths_words = [len(nltk.word_tokenize(phrase)) for phrase in phrases]
        
        # Length of each phrase in characters
        phrase_lengths_chars = [len(phrase) for phrase in phrases]
        
        # Length of the longest phrase (in words)
        max_length_words = max(phrase_lengths_words) if phrase_lengths_words else 0
        
        # Length of the shortest phrase (in words)
        min_length_words = min(phrase_lengths_words) if phrase_lengths_words else 0
        
        # Average phrase length (in words)
        avg_length_words = sum(phrase_lengths_words) / num_phrases if num_phrases > 0 else 0
        
        # Add the sentence statistics to the list
        sentence_stats.append({
            'num_phrases': num_phrases,
            'max_length_words': max_length_words,
            'min_length_words': min_length_words,
            'avg_length_words': avg_length_words,
            'phrase_lengths_words': phrase_lengths_words,
            'phrase_lengths_chars': phrase_lengths_chars
        })

    # Global statistics summary
    total_num_phrases = sum(stat['num_phrases'] for stat in sentence_stats)
    longest_phrase = max((max(stat['phrase_lengths_words']) for stat in sentence_stats if stat['phrase_lengths_words']), default=0)
    shortest_phrase = min((min(stat['phrase_lengths_words']) for stat in sentence_stats if stat['phrase_lengths_words']), default=0)
    avg_length_all_phrases = sum(stat['avg_length_words'] for stat in sentence_stats) / len(sentence_stats) if sentence_stats else 0

    # Display the global results
    print(f"Total number of phrases: {total_num_phrases}")
    print(f"Longest phrase length (in words): {longest_phrase}")
    print(f"Shortest phrase length (in words): {shortest_phrase}")
    print(f"Average phrase length (in words): {avg_length_all_phrases:.2f}")

    # 1. Histogram of phrase lengths (in words)
    phrase_lengths_words_all = [length for stat in sentence_stats for length in stat['phrase_lengths_words']]
    plt.figure(figsize=(10, 6))
    sns.histplot(phrase_lengths_words_all, bins=30, kde=True)
    plt.title('Distribution of Phrase Lengths (in words)')
    plt.xlabel('Number of words per phrase')
    plt.ylabel('Frequency')
    plt.show()

    # 2. Boxplot of phrase lengths (in words)
    plt.figure(figsize=(10, 6))
    sns.boxplot(phrase_lengths_words_all)
    plt.title('Boxplot of Phrase Lengths (in words)')
    plt.xlabel('Phrase length (in words)')
    plt.show()

    # 3. Distribution of the number of phrases per sentence
    num_phrases_all = [stat['num_phrases'] for stat in sentence_stats]
    plt.figure(figsize=(10, 6))
    sns.histplot(num_phrases_all, bins=range(1, max(num_phrases_all)+1), kde=True)
    plt.title('Distribution of Number of Phrases per Report')
    plt.xlabel('Number of phrases')
    plt.ylabel('Number of Reports')
    plt.show()

    dic = pyphen.Pyphen(lang='fr')

    def count_syllables(word):
        syllables = dic.inserted(word).split('-')
        return len(syllables)

    words = [word for sentence in sentences for word in nltk.word_tokenize(sentence)]

    # Calculate the total number of syllables
    total_syllables = sum(count_syllables(word) for word in words)
    print(f"Total number of syllables: {total_syllables}")
    print(f"Total number of words: {len(words)}")

    all_words = ' '.join([word for sentence in sentences for word in nltk.word_tokenize(sentence) if word.lower() not in stop_words and word.isalpha()])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)

    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud')
    plt.show()
    
    # 1. Sentence Lengths
    num_words_per_sentence = [len(nltk.word_tokenize(sentence)) for sentence in sentences]
    num_chars_per_sentence = [len(sentence) for sentence in sentences]

    # 2. Subtoken Analysis
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    subtokens_per_sentence = [len(tokenizer.tokenize(sentence)) for sentence in sentences]

    # 3. Vocabulary and Lexical Richness
    all_words = [word.lower() for sentence in sentences for word in nltk.word_tokenize(sentence)]
    unique_words = set(all_words)
    total_vocab = len(unique_words)
    ttr = len(unique_words) / len(all_words) if all_words else 0

    # 4. Word Frequency and Word Cloud
    all_words = [word.lower() for sentence in sentences for word in nltk.word_tokenize(sentence) if word.lower() not in stop_words and word.isalpha()]
    word_freq = Counter(all_words)
    common_words = word_freq.most_common(20)  # 20 most frequent words

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud")
    plt.show()

    # 5. Linguistic Specificities (medical terms)
    medical_terms = ["diabetes", "hypertension", "cancer", "asthma", "heart", "kidney"]  # Example of medical terms
    medical_term_freq = {term: word_freq[term] for term in medical_terms}

    # 6. Comparison Between Sections or Subsections
    # Implementation depends on specific data, not included here

    # 7. N-Gram Analysis
    vectorizer = CountVectorizer(ngram_range=(2, 3))
    X = vectorizer.fit_transform(sentences)
    ngrams = vectorizer.get_feature_names_out()
    ngram_freq = X.sum(axis=0).A1
    ngram_freq_dict = dict(zip(ngrams, ngram_freq))
    top_ngrams = Counter(ngram_freq_dict).most_common(20)

    # 8. Linguistic Complexity (Readability Index)
    def calculate_flesch_kincaid(text):
        return textstat.flesch_kincaid_grade(text)

    # Calculate the Flesch-Kincaid score for each sentence
    flesch_kincaid_scores = [calculate_flesch_kincaid(sentence) for sentence in sentences]

    # Display the results
    scores = []
    print("Flesch-Kincaid Scores:")
    for i, score in enumerate(flesch_kincaid_scores):
        scores.append(score)
    print(scores)

    # 9. Author Diversity (if applicable)
    # Not applicable here, requires additional data

    # Display main results
    print(f"Number of sentences: {len(sentences)}")
    print(f"Average number of words per sentence: {sum(num_words_per_sentence) / len(num_words_per_sentence) if num_words_per_sentence else 0}")
    print(f"Average number of characters per sentence: {sum(num_chars_per_sentence) / len(num_chars_per_sentence) if num_chars_per_sentence else 0}")
    print(f"Average number of subtokens per sentence: {sum(subtokens_per_sentence) / len(subtokens_per_sentence) if subtokens_per_sentence else 0}")
    print(f"Total unique word count: {total_vocab}")
    print(f"Lexical richness (TTR): {ttr}")
    print(f"20 most frequent words: {common_words}")
    print(f"Frequency of specific medical terms: {medical_term_freq}")
    print(f"20 most frequent bigrams and trigrams: {top_ngrams}")
    print(f"Average Flesch-Kincaid readability score: {sum(flesch_kincaid_scores) / len(flesch_kincaid_scores) if flesch_kincaid_scores else 0}")

