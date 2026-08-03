from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import pandas as pd
import kagglehub
import pickle
import spacy
import re
import os

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def preprocess(text: str) -> str:
    """
    Preprocess a text message for classification.

    The preprocessing pipeline performs the following steps:

    - converts text to lowercase,
    - removes URLs,
    - removes numbers,
    - removes punctuation,
    - lemmatizes words,
    - removes stop words and short tokens.

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    str
        Preprocessed text suitable for feature extraction.
    """
    # lowercase
    text = text.lower()

    # remove urls
    text = re.sub(r"http\S+|www\S+", "", text)

    # remove numbers
    text = re.sub(r"\d+", "", text)

    # remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # lemmatization
    doc = nlp(text)

    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_space and len(token) > 2
    ]

    return " ".join(tokens)


def train(path: str = "classifier_model.pkl"):
    """
    Train a spam classification model and save it to disk.

    The function downloads the SMS Spam Collection dataset, preprocesses the
    text, trains a TF-IDF + Logistic Regression pipeline, evaluates its
    performance, and serializes the trained model.

    Parameters
    ----------
    path : str, default="classifier_model.pkl"
        Output path for the serialized model.
    """
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

    with open(path, "wb") as file:
        pickle.dump(pipeline, file)
