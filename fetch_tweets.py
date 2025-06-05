import time
import threading
from cachetools import TTLCache
import tweepy
from utils import clean_tweets, save_tweets_to_csv  # Import from utils.py

# Twitter API Setup
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAOJvzQEAAAAAJIMILtcmCBOcgUosWanTH%2F17eOc%3DQGLx9diQAqop0yCYeuTSxVYYfdFTyJ6B7re7LJClBSbUGsdZb3"
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Cache to reduce API calls
cache = TTLCache(maxsize=100, ttl=900)  # Store results for 15 minutes

def fetch_tweets(query, count):
    """Fetches tweets from Twitter API and cleans them."""
    if query in cache:
        return cache[query]

    try:
        tweets = client.search_recent_tweets(
        query=query,
        tweet_fields=["text", "created_at"],
        max_results=max(min(count, 100), 10)
    )


        if tweets.data:
            cleaned_tweets = clean_tweets(tweets.data[:count])  # Clean tweets
            save_tweets_to_csv(cleaned_tweets, query)  # Save to CSV
            cache[query] = cleaned_tweets
            return cleaned_tweets
        return []

    except tweepy.errors.TooManyRequests:
        print("Rate limit exceeded. Waiting for 15 minutes before retrying...")
        time.sleep(900)
        return fetch_tweets(query, count)
    except Exception as e:
        print(f"Error fetching tweets: {e}")
        return []

def fetch_tweets_with_timeout(query, count=10, timeout=20):
    """Fetch tweets but cancel if it takes too long."""
    result = []

    def worker():
        nonlocal result
        result = fetch_tweets(query, count)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        print("⚠️ Timeout exceeded while fetching tweets.")
        return []

    return result

# Test fetching
if __name__ == "__main__":
    tweets = fetch_tweets_with_timeout("gender-based violence", 10)
    print(tweets)
