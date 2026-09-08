# Model training script

from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.utils import preprocess

import pandas as pd
import kagglehub
import pickle
import os

model_name = "sms_spam_classifier.pkl"

print("Downloading data")

dataset_path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
print(f"Dataset path: {dataset_path}")
data = pd.read_csv(os.path.join(dataset_path, "spam.csv"), encoding="ISO-8859-1")

print("Transforming data")

data.drop(data[["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"]], axis=1, inplace=True)
data.rename(columns={"v1": "Spam", "v2": "Text"}, inplace=True)

data["Spam"] = data["Spam"].apply(lambda x: 1 if x == "spam" else 0)
data["Text"] = data["Text"].apply(lambda x: preprocess(x))

print("Training model")

X_train, X_test, y_train, y_test = train_test_split(
    data["Text"], data["Spam"], test_size=0.2, random_state=8, stratify=data["Spam"]
)

pipeline = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                max_features=None, ngram_range=(1, 2), min_df=2, max_df=0.9
            ),
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=1000, solver="liblinear", C=100, random_state=8
            ),
        ),
    ]
)

pipeline.fit(X_train, y_train)

print("Evaluating model")
predictions = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

print("Saving model")

with open(model_name, "wb") as file:
    pickle.dump(pipeline, file)
