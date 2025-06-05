import re
import os
import pandas as pd
from datetime import datetime
import torch

def clean_tweet(text):
    """Cleans tweet text by removing RTs, mentions, URLs, hashtags, and special characters."""
    text = re.sub(r'RT\s@\w+: ', '', text)  # Remove Retweet (RT) text
    text = re.sub(r'@\w+', '', text)  # Remove @mentions
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    return text.strip()

def clean_tweets(tweets):
    """Cleans a list of tweet objects into dicts with cleaned text and timestamp."""
    cleaned = []
    for tweet in tweets:
        cleaned.append({
            "text": clean_tweet(tweet.text),
            "timestamp": getattr(tweet, "created_at", "Unknown"),
            "username": getattr(tweet, "author_id", "Unknown")  # If you have author_id
        })
    return cleaned

def save_tweets_to_csv(tweets, query):
    """Saves cleaned tweets to a CSV file."""
    df = pd.DataFrame(tweets)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    filename = "tweets.csv"
    
    df.to_csv(filename, mode="a", index=False, header=not os.path.exists(filename))
    print(f"✅ Saved {len(tweets)} tweets to {filename}")

def load_past_tweets(query):
    """Loads past analyzed tweets from a CSV file."""
    filename = f"analyzed_tweets_{query.replace(' ', '_')}.csv"
    if os.path.exists(filename):
        return pd.read_csv(filename).to_dict(orient="records")
    return []

def predict_sentiment(text, sentiment_model, sentiment_tokenizer, labels_sentiment):
    """Predicts sentiment label for a given text using a transformer model."""
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentiment_model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    label_idx = torch.argmax(outputs.logits, dim=-1).item()
    return labels_sentiment[label_idx]

def predict_gbv(text, gbv_model, gbv_tokenizer, labels_gbv):
    """Predicts GBV label for a given text using a transformer model."""
    inputs = gbv_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gbv_model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = gbv_model(**inputs)
    label_idx = torch.argmax(outputs.logits, dim=-1).item()
    return labels_gbv[label_idx]
