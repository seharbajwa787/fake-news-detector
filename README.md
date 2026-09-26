# 📰 Fake News Detector

A machine learning model that reads a news article and predicts whether it is **REAL** or **FAKE**, trained on a dataset of 4,594 real-world news articles.

**[🔗 Live Demo](https://claude.ai/artifact/BAjMRG2qUJnXD2GtqLPzUD)** — try it instantly in your browser, no install needed.

## Overview

Misinformation spreads fast, especially on social media. This project builds a text classifier that flags whether a news article is likely genuine or fabricated, using classic NLP + machine learning techniques.

## Results

| Metric | Score |
|---|---|
| Accuracy (full model) | **94.12%** |
| Precision (FAKE) | 0.94 |
| Precision (REAL) | 0.94 |
| Dataset size | 4,594 articles (balanced) |

## How it works

1. **Dataset** — 4,594 news articles labeled REAL or FAKE, sourced from Reuters (real) and outlets flagged by fact-checkers (fake).
2. **Feature extraction** — Article text is converted into numeric features using **TF-IDF** (Term Frequency–Inverse Document Frequency).
3. **Model** — A **Passive Aggressive Classifier** is trained on the TF-IDF vectors — a fast, accurate algorithm well suited to text classification.
4. **Evaluation** — The model is tested on a held-out 20% split it never saw during training.

## Project structure

```
fake-news-detector/
├── README.md
├── requirements.txt
├── src/
│   ├── download_dataset.py     # downloads the dataset
│   └── fake_news_detector.py   # trains & evaluates the model
├── data/                        # dataset goes here (downloaded, not committed)
├── models/                      # trained model files saved here after running
└── demo/
    └── index.html               # lightweight browser-only live demo
```

## Setup & usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the dataset
python src/download_dataset.py

# 3. Train and evaluate the model
python src/fake_news_detector.py
```

This will print accuracy, a classification report, and save the trained model to `models/`.

## Live browser demo

`demo/index.html` is a **standalone, dependency-free** version of the model — the top 350 most predictive words and their learned weights are embedded directly in the page, so it runs a simplified version of the classifier entirely client-side (~85% accuracy vs. 94% for the full Python model). Just open the file in any browser, or visit the [live demo link](https://claude.ai/artifact/BAjMRG2qUJnXD2GtqLPzUD).

## Limitations

- The dataset is mostly 2016 US political/election news, so the model performs best on political content and less reliably on unrelated topics (sports, tech, health, etc.).
- This is a proof-of-concept / educational project, not a production fact-checking tool.

## Tech stack

- Python
- pandas
- scikit-learn (TF-IDF, Passive Aggressive Classifier)
- joblib (model persistence)
- HTML/CSS/JavaScript (browser demo)

## Dataset credit

Dataset originally compiled by [G. McIntire](https://github.com/GeorgeMcIntire/fake_real_news_dataset).

## License

MIT
