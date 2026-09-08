# SMS Spam Classifier

A machine learning web application for binary SMS classification built with **FastAPI**, **Streamlit**, and **scikit-learn**. The application predicts whether a text message is **spam** or **ham** using a TF-IDF feature extractor and a Logistic Regression classifier.

[Try it out]("https://mzivro-sms-spam-clf.streamlit.app/")

## Features

* Binary SMS spam classification working with around **97% of accuracy**
* Automatic text preprocessing using spaCy
* REST API built with FastAPI
* Interactive web interface built with Streamlit
* Automatic model download from Hugging Face
* Demo app and training script included

## Tech Stack

* scikit-learn
* spaCy
* FastAPI
* Pydantic
* Streamlit
* Pandas
* KaggleHub
* HuggingFaceHub
* Pickle

## Model training Pipeline

1. Download the SMS Spam Collection dataset from Kaggle.
2. Preprocess text:

   * convert to lowercase,
   * remove URLs,
   * remove numbers,
   * remove punctuation,
   * remove stop words,
   * lemmatize words.
3. Convert text into TF-IDF features.
4. Train a Logistic Regression classifier.
5. Save the trained model for future inference.

## Installation

Clone the repository:

```bash
git clone https://github.com/mzivro/sms-spam-classifier.git
cd sms-spam-classifier
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the spaCy language model:

```bash
python -m spacy download en_core_web_sm
```

## Running the API

The model will be automatically trained with first start-up of API.

```bash
fastapi run src/server.py
```

The API will be available at:

```
http://localhost:8000
```

Interactive API documentation:

```
http://localhost:8000/docs
```

## Running the Streamlit Client

```bash
streamlit run src/client.py
```

The application will be available at:

```
http://localhost:8501
```

## API

### GET `/health`

Returns the current health status of the API.

Response

```json
{
  "status": "ok"
}
```

### GET `/ready`

Checks whether the model has been successfully loaded.

Successful response:

```json
{
  "status": "ready"
}
```

If the model is not available, the API returns:

```
503 Service Unavailable
```

### POST `/predict`

Request

```json
{
  "text": "Congratulations! You have won a free prize."
}
```

Response

```json
{
  "prediction": 1
}
```

Where:

* `0` = Ham
* `1` = Spam

## Docker

Build the image:

```bash
docker build -t sms-spam-classifier .
```

Run the container:

```bash
docker run -p 8000:8000 sms-spam-classifier
```

## Dataset

The project uses the **SMS Spam Collection** dataset from the UCI Machine Learning Repository, downloaded via KaggleHub.

This dataset is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

Dataset source:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

Kaggle link: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

License:
https://creativecommons.org/licenses/by/4.0/

Reference: \
Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84.

## License

MIT License. Feel free to use, modify, and build upon this project.
