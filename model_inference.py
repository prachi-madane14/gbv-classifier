import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Load the model architecture
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3)

# Load the model weights on CPU
model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Define class labels corresponding to the model's output
labels = ["Negative", "Neutral", "Positive"]  # Adjust this according to your labels

# Example of how you can test the model
def predict(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction_idx = torch.argmax(logits, dim=-1).item()
    return labels[prediction_idx]

# Test the model with an example text
text = "Women should stay in their place and stop trying to act smart. No one wants to hear their opinions!"
prediction = predict(text)
print(f"Sentiment: {prediction}")
