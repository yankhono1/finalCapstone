import pandas as pd
import spacy
from textblob import TextBlob
from collections import defaultdict

# Load the spaCy model
nlp = spacy.load('en_core_web_md')

# Load the dataset
amazon = pd.read_csv('C:\Users\NewStart\Hyperion_code_data_science\T21\datasets\amazon_product_reviews.csv')

# Drop missing values
amazon.dropna(inplace=True)

# Define a function for preprocessing using spaCy
def preprocess(text):
    doc = nlp(text.lower().strip())
    processed = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(processed)

# Apply preprocessing to the text column
amazon['processed_text'] = amazon['reviews.text'].apply(preprocess)

# Dictionary to feed the data into
positive_words = defaultdict(int)
negative_words = defaultdict(int)

# Sentiment analysis
for sentence in amazon['processed_text']:
    doc = nlp(sentence)
    polarity = TextBlob(sentence).sentiment.polarity
    
    for token in doc:
        if polarity > 0:
            positive_words[token.lemma_.lower()] += 1
        elif polarity < 0:
            negative_words[token.lemma_.lower()] += 1

# Output the results
print("Positive words:", positive_words)
print("Negative words:", negative_words)


