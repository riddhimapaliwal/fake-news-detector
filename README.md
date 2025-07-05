# Fake News Detector – Enhanced

This repository contains a quick-start **Fake News Detection** project, adapted and extended from the open‑source work by [satishgunjal/fake-news-detection](https://github.com/satishgunjal/fake-news-detection).

## 🔥 What’s new in **this fork**

* 📝 **Updated dataset** – now uses the 2024 Kaggle Fake/Real News dataset (balanced).
* 🏷️ **Better text cleaning** – custom regex pipeline + spaCy lemmatization.
* 🤖 **Model upgrade** – swapped PassiveAggressiveClassifier for **TF‑IDF + Linear SVM**, giving +3 F1 score.
* 📊 **Streamlit dashboard** – run `streamlit run app.py` for an interactive demo (type or paste an article, get prediction & probability bar).
* 📈 **Evaluation notebook** – detailed Precision‑Recall curves & confusion matrix.

## 🚀 Quick start
```bash
git clone https://github.com/your‑username/fake_news_enhanced.git
cd fake_news_enhanced
pip install -r requirements.txt
streamlit run app.py
```

## 🗂️ Repo structure
```
fake_news_enhanced/
├── data/                 <- (empty) put `train.csv` & `test.csv` here
├── notebooks/
│   └── 01_train.ipynb    <- end‑to‑end training notebook
├── model_train.py        <- CLI script to (re)train & save model.pkl
├── app.py                <- Streamlit UI
├── requirements.txt
└── README.md
```

## 📜 License & Attribution
Original base code MIT‑licensed by Satish Gunjal.  
All modifications in this repo © 2025 Riddhima Paliwal – MIT License.
