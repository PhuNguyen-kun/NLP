# Build a simple n-gram language model

text = "I love deep learning. I love natural language processing. I enjoy learning about language models."

import re

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    tokens = text.split()  # split into tokens
    return tokens

tokens = preprocess(text)
# print(tokens)

# Make a trigram language model
from collections import defaultdict

trigram_model = defaultdict(list)

for i in range(len(tokens) - 2):
    key = (tokens[i], tokens[i+1])
    next_word = tokens[i+2]
    trigram_model[key].append(next_word)

# for key, value in trigram_model.items():
#     print(f"{key} → {value}")

# Predict the next word in a sentence using the trigram model 
import random

def predict_next_word(model, w1, w2):
    key = (w1, w2)
    if key in model:
        return random.choice(model[key])  # randomly choose one possible word
    else:
        return "?"

# Example predictions
print(predict_next_word(trigram_model, "i", "love"))
print(predict_next_word(trigram_model, "natural", "language"))

# Tính xác suất của từng từ tiếp theo
def get_probabilities(model, w1, w2):
    key = (w1, w2)
    if key in model:
        next_words = model[key]
        total = len(next_words)
        probs = {}
        for word in set(next_words):
            probs[word] = next_words.count(word) / total
        return probs
    else:
        return {}

# Xem xác suất
print(get_probabilities(trigram_model, "i", "love"))
