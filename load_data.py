"""
Load or generate labeled tweet dataset for sentiment analysis.
Uses Sentiment140-style format: target 0=negative, 4=positive.
"""
import os
import pandas as pd


def get_sample_data():
    """Return a built-in sample of labeled tweets (no external file needed)."""
    positive = [
        "I love this product! Best purchase ever.",
        "Amazing experience, highly recommend to everyone.",
        "So happy with the results, thank you!",
        "This is fantastic and exactly what I needed.",
        "Great service and fast delivery. Will buy again!",
        "Absolutely wonderful! Made my day.",
        "Could not be happier. Five stars!",
        "Best decision I ever made. Love it!",
        "Incredible quality. Exceeded my expectations.",
        "So grateful for this. Thank you so much!",
        "Happy birthday! Hope you have an amazing day!",
        "Just got promoted! So excited for this new chapter.",
        "The concert was incredible! Best night ever.",
        "Finally finished my project. So relieved and proud!",
        "Coffee and sunshine - perfect morning.",
        "New job starts Monday. Can't wait!",
        "Team won the game! What a victory!",
        "Got accepted to my dream school!",
        "This movie was so good. Must watch!",
        "Best vacation ever. Memories for a lifetime.",
    ]
    negative = [
        "This is terrible. Worst experience ever.",
        "Very disappointed with the quality. Do not buy.",
        "Complete waste of money. Regret purchasing.",
        "Horrible service. Will never use again.",
        "Broken on arrival. So frustrated.",
        "Absolutely awful. Avoid at all costs.",
        "Waste of time. Nothing worked as promised.",
        "Extremely disappointed. Expected much better.",
        "Poor customer service. Very unhelpful.",
        "Product failed after one day. Unacceptable.",
        "So angry right now. This is ridiculous.",
        "Missed my flight. Worst day ever.",
        "The food was cold and the service was slow.",
        "Lost my wallet. Feeling so stressed.",
        "Another delay. When will this end?",
        "Customer support was useless. So frustrated.",
        "Overpriced and underdelivered. Not worth it.",
        "Cancelled my order. Too many issues.",
        "This app keeps crashing. So annoying.",
        "Never ordering from here again. Terrible.",
    ]
    # Expand with variations to get ~2000+ samples
    tweets = []
    labels = []
    for i in range(80):
        for t in positive:
            tweets.append(t)
            labels.append(4)  # positive
        for t in negative:
            tweets.append(t)
            labels.append(0)  # negative
    return pd.DataFrame({"text": tweets, "target": labels})


def load_dataset(custom_path=None):
    """
    Load labeled tweet dataset.
    If custom_path is provided and exists, load from CSV (Sentiment140 format).
    Otherwise return built-in sample data.
    """
    if custom_path and os.path.isfile(custom_path):
        df = pd.read_csv(custom_path, encoding="latin-1")
        # Sentiment140: columns may be target, id, date, flag, user, text
        if "text" not in df.columns and df.shape[1] >= 6:
            df.columns = ["target", "id", "date", "flag", "user", "text"]
        if "target" not in df.columns and "sentiment" in df.columns:
            df["target"] = df["sentiment"]
        return df[["text", "target"]].dropna()
    return get_sample_data()
