## Week 1: Introduction to NLP & Text Preprocessing

**Mục tiêu:**
- Giới thiệu các khái niệm cơ bản trong NLP.
- Làm rõ các thách thức trong xử lý ngôn ngữ tự nhiên.
- Làm quen với các kỹ thuật tiền xử lý văn bản.

**Kiến thức chính:**
- NLP (Natural Language Processing) là gì và tại sao quan trọng?
- Các bước tiền xử lý văn bản phổ biến:
  - Lowercasing: chuyển chữ hoa thành chữ thường.
  - Tokenization: tách văn bản thành từ hoặc câu.
  - Stop-word removal: loại bỏ các từ không mang nhiều ý nghĩa.
  - Stemming/Lemmatization: đưa từ về gốc.

**Thực hành:**
- Cài đặt spaCy hoặc NLTK.
- Load một tập dữ liệu mẫu (ví dụ: văn bản tin tức, tweet).
- Thực hiện các bước: tokenization, loại bỏ stop word, stemming hoặc lemmatization.
- Xây dựng pipeline tiền xử lý văn bản đơn giản.

**Kết quả đạt được:**
- Hiểu cách chuyển đổi văn bản thô thành dữ liệu có cấu trúc.
- Chuẩn bị cho các bước mô hình hóa tiếp theo (n-gram, embedding,...).

---------------------------------------------------------------------------------------------------------------------------------
## Week 2: Statistical Language Models (N-gram, Bag-of-Words)

**Mục tiêu:**
- Tìm hiểu các phương pháp thống kê trong xử lý ngôn ngữ.
- Làm quen với mô hình n-gram và Bag-of-Words.
- So sánh các phương pháp truyền thống với mô hình học sâu hiện đại.

**Kiến thức chính:**
- N-gram model:
  - Dự đoán từ tiếp theo dựa trên (n-1) từ trước đó.
  - Ví dụ: trigram model dự đoán từ thứ 3 dựa trên 2 từ trước.
  - Ưu điểm: đơn giản, dễ hiểu.
  - Hạn chế: không nhớ ngữ cảnh dài, dễ gặp từ chưa từng thấy → xác suất bằng 0.
  - Cần áp dụng kỹ thuật smoothing để xử lý.

- Bag-of-Words (BoW):
  - Biểu diễn văn bản thành vector dựa trên tần suất từ.
  - Không quan tâm đến thứ tự từ.
  - Ưu điểm: dễ cài đặt, hiệu quả với mô hình đơn giản.
  - Hạn chế: mất ngữ cảnh, không nắm bắt được nghĩa của từ.

- Giới hạn của phương pháp thống kê:
  - Không hiểu ngữ nghĩa.
  - Không xử lý tốt dữ liệu hiếm hoặc từ mới.
  - Là nền tảng cho sự phát triển của mô hình học sâu.

**Thực hành:**
- Dùng Python với thư viện cơ bản (collections, numpy).
- Xây dựng mô hình trigram để dự đoán từ tiếp theo.
- Tính xác suất xuất hiện và trực quan hóa phân phối xác suất.

**Kết quả đạt được:**
- Hiểu và triển khai được mô hình ngôn ngữ thống kê đơn giản.
- Làm quen với khái niệm xác suất trong xử lý ngôn ngữ.
- Nhận ra giới hạn và sự cần thiết của các phương pháp hiện đại hơn.
