# Train a word embedding model on a small corpus using Gensim's Word2Vec

from gensim.utils import simple_preprocess # type: ignore
from gensim.models import Word2Vec # type: ignore
from sklearn.manifold import TSNE # type: ignore
import matplotlib.pyplot as plt # type: ignore

corpus = [
    "I love natural language processing",
    "Deep learning is a part of machine learning",
    "I enjoy learning about word embeddings",
    "Word2Vec creates vector representations of words",
    "Machine learning and NLP are exciting fields"
]

processed_corpus = [simple_preprocess(sentence) for sentence in corpus]
# print(processed_corpus)

model = Word2Vec(sentences=processed_corpus, vector_size=50, window=2, min_count=1, sg=1)

# print(model.wv["learning"]) # Get the vector for the word "learning" 
# print(model.wv.most_similar("and", topn=3))  # Find 3 words similar to "learning"

words = list(model.wv.index_to_key)
word_vectors = model.wv[words]

# Giảm chiều
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
word_vec_2d = tsne.fit_transform(word_vectors)

# Vẽ biểu đồ
plt.figure(figsize=(8, 6))
for i, word in enumerate(words):
    plt.scatter(word_vec_2d[i, 0], word_vec_2d[i, 1])
    plt.annotate(word, xy=(word_vec_2d[i, 0], word_vec_2d[i, 1]))
plt.title("Word Embedding Visualization (2D)")
plt.show()


