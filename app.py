import os
import pandas as pd
import re
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
sheet_id = "13g5gckP9kDbFef1nuIQoIghs7BpbyqOz05oWaRLa8tM"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
df = pd.read_csv(url)
print("Data loaded:", df.shape)

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["message_text"].apply(clean_text)

# Model 1: tactic detection
tactic_columns = [
    "authority",
    "urgency",
    "fear",
    "reciprocity",
    "social_proof",
    "greed"
]

X = df["clean_text"]
y_tactics = df[tactic_columns]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_tactics,
    test_size=0.2,
    random_state=42
)

vectorizer = TfidfVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)

tactic_model = MultiOutputClassifier(
    RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
)

tactic_model.fit(X_train_vec, y_train)
print("Tactic model trained")

# Model 2: scam type detection
y_type = df["scam_type"]

type_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

type_model.fit(vectorizer.transform(X), y_type)

# Explanations
tactic_explanations = {
    "authority": "pretends to be police, government, or a trusted organization",
    "urgency": "pressures you to act
