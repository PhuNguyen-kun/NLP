# Explore a basic implementation of a Transformer model

# 1. Sentiment analysis
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.eval()

text = "I love this movie! It's amazing."

inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

prediction = torch.argmax(logits, dim=-1)
print(f"Predicted sentiment: {'Positive' if prediction.item() == 1 else 'Negative'}")

# 2. Text generation
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

output = model.generate(input_ids, max_length=50, num_return_sequences=1)

output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)