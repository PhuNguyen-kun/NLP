#Build a simple text classification or sentiment analysis model

import tensorflow as tf # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.datasets import imdb # type: ignore # cSpell:ignore imdb
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout # type: ignore

# Tải dữ liệu IMDB (dữ liệu gồm các bài đánh giá phim)
num_words = 10000  # Số từ tối đa
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

# Đảm bảo tất cả các chuỗi có độ dài bằng nhau
maxlen = 500
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# print(f"Training samples: {len(x_train)}")
# print(f"Testing samples: {len(x_test)}")

# Xây dựng mô hình LSTM
model = Sequential()
model.add(Embedding(num_words, 128, input_length=maxlen))  # Lớp nhúng (embedding)
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))  # LSTM layer
model.add(Dense(1, activation='sigmoid'))  # Output layer (0 hoặc 1)

# # Biên dịch mô hình
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Tóm tắt mô hình
# model.summary()

# # Huấn luyện mô hình
# history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# # Đánh giá mô hình trên dữ liệu test
# score, accuracy = model.evaluate(x_test, y_test, batch_size=64)
# print(f"Test score: {score}")
# print(f"Test accuracy: {accuracy}")

# Dự đoán trên dữ liệu mới
new_review = ["This movie is fantastic! I loved it."]

# Chuyển đổi các từ trong bài đánh giá mới thành chỉ số (từ điển của IMDB)
word_index = imdb.get_word_index()
new_review_tokenized = [[word_index.get(word, 0) for word in review.lower().split()] for review in new_review]

# Pad các chuỗi để đảm bảo độ dài giống như dữ liệu huấn luyện
new_review_padded = pad_sequences(new_review_tokenized, maxlen=maxlen)

# Dự đoán cảm xúc (positive/negative) cho bài đánh giá mới
prediction = model.predict(new_review_padded)

print(f"Predicted sentiment: {'Positive' if prediction[0] > 0.5 else 'Negative'}")
