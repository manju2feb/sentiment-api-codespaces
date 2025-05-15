import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Sample dataset
data = {
    "text": [
        "I love this product!",
        "This is the worst experience.",
        "Absolutely fantastic!",
        "I hate it.",
        "Very satisfied and happy.",
        "Not happy with the service.",
        "Excellent support!",
        "Terrible, won’t buy again.",
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

pipeline.fit(X_train, y_train)

# Save the model
os.makedirs('model', exist_ok=True)
joblib.dump(pipeline, 'model/sentiment_model.pkl')
print("✅ Model saved at model/sentiment_model.pkl")
