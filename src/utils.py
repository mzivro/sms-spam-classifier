import pickle
import spacy
import re

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
