import spacy # type: ignore

nlp = spacy.load("en_core_web_sm")

# Xử lý văn bản
doc = nlp("Apple is looking at buying U.K. startup for $1 billion! It's a big move in the tech industry.")

# Lowercasing, Tokenization, Stop-word Removal, Lemmatization
processed_tokens = []
for token in doc:
    if not token.is_stop and not token.is_punct:  # Loại bỏ stop words & dấu câu
        processed_tokens.append(token.lemma_.lower())

print(processed_tokens)