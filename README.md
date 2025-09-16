# Decoding Decades: Classifying Rap Lyrics by Era

This project was developed for a university exam in Natural Language Processing.  
It predicts the **decade of release** (1980s–2020s) from rap lyrics and explores linguistic changes over time.

## Overview
- **Dataset:** ~3,900 rap songs collected from Genius. After cleaning and filtering, ~3,879 songs remained, spanning 1980–2024.
- **Task:** Multiclass classification of lyrics into decades, combined with exploratory analysis of lexical richness, sentiment, and trends.
- **Evaluation:** 5-fold stratified cross-validation; macro F1-score as primary metric.

## Methods
- **Preprocessing:** Tokenization (NLTK), lowercasing, stopword handling adapted for rap slang, vocabulary restricted to training set, lemmatization for analysis.
- **Features:**  
  - TF-IDF vectors for classical models  
  - Tokenized & padded sequences for neural models
- **Models:** Logistic Regression, Random Forest, CNN, LSTM

## Results
- **Logistic Regression (TF-IDF):** F1 ≈ 0.63 (best overall, balanced across decades)  
- **Random Forest:** F1 ≈ 0.62 (stronger at extremes, less balanced)  
- **CNN:** F1 ≈ 0.58 (captures local n-grams, struggles with adjacent decades)  
- **LSTM:** F1 ≈ 0.39 (limited by dataset size and repetition in modern lyrics)  

Linguistic trends: lexical richness declined from ~13% (1980s) to ~9% (2020s).

## Usage
Requirements: Python 3.10+

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pandas numpy scikit-learn nltk matplotlib tensorflow gensim lyricsgenius rapidfuzz
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon')"

## Notes
To rebuild the dataset, a Genius API token is required (lyricsgenius).
Raw lyrics are not included for copyright reasons.
The accompanying paper details methodology, preprocessing, and results.


## Limitations & Future Work
Dataset size and class imbalance constrain model performance.
Possible artist overlap across splits may inflate scores.
Future improvements: larger dataset, artist-aware splitting, integrating linguistic features (e.g., lexical richness, sentiment) into classifiers.
