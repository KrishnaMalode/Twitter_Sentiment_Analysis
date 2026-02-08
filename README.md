# Twitter Sentiment Analysis (Streamlit)

Sentiment classification using Python, NLTK, and Scikit-learn with a labeled tweet dataset.

## Features

- **NLP pipeline**: tokenization → stemming → lemmatization → TF-IDF vectorization  
- **4 models**: Naive Bayes, Logistic Regression, Decision Tree, Random Forest  
- **Model comparison**: accuracy, precision, recall, F1  
- **Confusion matrix** for the best model (by F1)  
- **Live sentiment detection** for your own text  
- **Demo** on sample tweets  
- **Export** predictions as CSV  

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Custom dataset

Place a CSV with columns `text` and `target` (0 = negative, 4 = positive). In the app sidebar choose "Custom CSV file" and enter the path (e.g. `data/training.csv`). Sentiment140-style CSVs (target, id, date, flag, user, text) are auto-detected.

## Tech stack

- **Streamlit** – UI  
- **NLTK** – tokenization, Porter stemmer, WordNet lemmatizer, stopwords  
- **Scikit-learn** – TF-IDF, train/test split, 4 classifiers, metrics  
