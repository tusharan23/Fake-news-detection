# Fake News Detection using ML & Streamlit

This project uses Natural Language Processing and machine learning to detect fake news articles. Models are trained on a labeled dataset using TF-IDF features and deployed through a Streamlit web app.

## Features

- Logistic Regression and Random Forest classifiers
- TF-IDF vectorization
- Accuracy, precision, recall, and F1-score evaluation
- Interactive web interface using Streamlit

##  Folder Contents

- `train_and_evaluate.ipynb`: Training notebook
- `app.py`: Streamlit app
- `Fake.csv` / `True.csv`: Dataset
- `*.jb`: Saved models & vectorizer

##  How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
