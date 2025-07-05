"""Quick CLI script to train Fake News detector."""
# Project: Fake News Detection
# Author: Riddhima Paliwal
# Year: 2025
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib
import pathlib, argparse, re, spacy

nlp = spacy.load('en_core_web_sm', disable=['ner','parser'])

def clean(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
    doc = nlp(text.lower())
    return ' '.join([tok.lemma_ for tok in doc if not tok.is_stop and tok.is_alpha])

def train(csv_path):
    df = pd.read_csv(csv_path)
    df['text_clean'] = df['text'].apply(clean)
    X_train, X_test, y_train, y_test = train_test_split(df['text_clean'], df['label'], test_size=0.2, random_state=42)

    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    clf = LinearSVC(C=1.0)
    clf.fit(X_train_tfidf, y_train)

    model_dir = pathlib.Path('models')
    model_dir.mkdir(exist_ok=True)
    joblib.dump({'vectorizer': tfidf, 'clf': clf}, model_dir/'model.pkl')
    print("Model saved to models/model.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='data/train.csv')
    args = parser.parse_args()
    train(args.csv)
