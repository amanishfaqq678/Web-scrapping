from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save charts without requiring an interactive plot window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import spacy
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
RAW_CSV = DATA_DIR / "articles_raw.csv"

URLS = [
    "https://techncruncher.blogspot.com/2025/01/top-10-ai-tools-that-will-transform.html",
    "https://techncruncher.blogspot.com/2023/12/limewire-ai-studio-review-2023-details.html",
    "https://techncruncher.blogspot.com/2023/01/top-10-ai-tools-in-2023-that-will-make.html",
    "https://techncruncher.blogspot.com/2022/11/top-10-ai-content-generator-writer.html",
    "https://techncruncher.blogspot.com/2022/09/cj-affiliate-ultimate-guide-to.html",
]

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def ensure_folders() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def run_numpy_demo() -> None:
    """Task 2.1: compare NumPy's vectorised sum with a Python loop."""
    print("\n" + "=" * 65)
    print("TASK 2.1 - NUMPY VS PYTHON")
    print("=" * 65)
    size = 1_000_000  # Kept practical for a laptop; the lab demonstrates the same idea.
    data = np.random.rand(size)

    start_np = time.perf_counter()
    numpy_sum = np.sum(data)
    numpy_time = time.perf_counter() - start_np

    start_py = time.perf_counter()
    manual_sum = 0.0
    for value in data:
        manual_sum += float(value)
    python_time = time.perf_counter() - start_py

    print(f"NumPy sum:   {numpy_sum:.6f} | Time: {numpy_time:.6f} seconds")
    print(f"Python sum:  {manual_sum:.6f} | Time: {python_time:.6f} seconds")
    print("The sums should be almost equal. NumPy is usually faster because its")
    print("vectorised operation runs in optimized compiled code.")


def scrape_articles() -> pd.DataFrame:
    """Task 2.2: collect URL, title and paragraph text from the lab URLs."""
    print("\nStarting web scraping. Internet access is required for this option.")
    articles = []

    for url in URLS:
        try:
            response = requests.get(url, headers=HEADERS, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "lxml")

            title_tag = soup.find("h3", class_="post-title")
            if title_tag:
                title = title_tag.get_text(" ", strip=True)
            elif soup.title:
                title = soup.title.get_text(" ", strip=True)
            else:
                title = "Untitled article"

            content_div = soup.find("div", class_="post-body entry-content")
            if content_div is None:
                print(f"[SKIP] Article body not found: {url}")
                continue

            paragraphs = content_div.find_all("p")
            body_text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)

            if len(body_text) > 100:
                articles.append({"url": url, "title": title, "text": body_text})
                print(f"[OK] Scraped: {title[:70]}")
            else:
                print(f"[SKIP] Article text is too short: {url}")

            time.sleep(1)
        except requests.RequestException as exc:
            print(f"[ERROR] Request failed for {url}: {exc}")
        except Exception as exc:
            print(f"[ERROR] Could not parse {url}: {exc}")

    return pd.DataFrame(articles, columns=["url", "title", "text"])


def load_raw_data(use_scraper: bool = False) -> pd.DataFrame:
    """Load the supplied CSV by default; scrape only when --scrape is requested."""
    if use_scraper:
        df = scrape_articles()
        if not df.empty:
            df.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")
            print(f"\nRaw scraped data saved to: {RAW_CSV}")
            return df
        print("\nScraping returned no articles. Trying the supplied CSV instead.")

    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found: {RAW_CSV}\n"
            "Place articles_raw.csv inside the data folder or run with --scrape."
        )

    df = pd.read_csv(RAW_CSV)
    required = {"url", "title", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "The raw CSV is missing required columns: "
            + ", ".join(sorted(missing))
            + ". Required columns are url, title, text."
        )

    df = df.dropna(subset=["title", "text"]).copy()
    df["title"] = df["title"].astype(str)
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip().ne("")]
    print(f"\nLoaded {len(df)} articles from {RAW_CSV.name}.")
    return df.reset_index(drop=True)


def load_spacy_model():
    """Load the English model and show a clear setup message if it is missing."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:
        raise RuntimeError(
            "spaCy English model is missing. In the VS Code terminal run:\n"
            "python -m spacy download en_core_web_sm\n"
            "Then run the project again."
        ) from exc


def extract_linguistic_features(text: str, nlp) -> dict:
    doc = nlp(text)
    return {
        "num_tokens": len(doc),
        "num_sentences": len(list(doc.sents)),
        "num_entities": len(doc.ents),
        "num_nouns": sum(1 for token in doc if token.pos_ == "NOUN"),
        "title_length": None,
    }


def add_features(df: pd.DataFrame, nlp) -> pd.DataFrame:
    """Task 2.3: extract spaCy linguistic features and title length."""
    print("\n" + "=" * 65)
    print("TASK 2.3 - SPACY FEATURE ENGINEERING")
    print("=" * 65)
    rows = [extract_linguistic_features(text, nlp) for text in df["text"]]
    feature_df = pd.DataFrame(rows)
    result = pd.concat([df.reset_index(drop=True), feature_df], axis=1)
    result["title_length"] = result["title"].str.len()
    print(result[["title", "num_tokens", "num_sentences", "num_entities",
                  "num_nouns", "title_length"]].head().to_string(index=False))
    return result


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Task 2.3: TextBlob polarity and subjectivity."""
    print("\n" + "=" * 65)
    print("TASK 2.3 - TEXTBLOB SENTIMENT")
    print("=" * 65)
    sentiments = df["text"].apply(
        lambda value: pd.Series({
            "sentiment_polarity": TextBlob(value).sentiment.polarity,
            "sentiment_subjectivity": TextBlob(value).sentiment.subjectivity,
        })
    )
    result = pd.concat([df.reset_index(drop=True), sentiments], axis=1)
    print(result[["title", "sentiment_polarity",
                  "sentiment_subjectivity"]].to_string(index=False))
    return result


def save_eda(df: pd.DataFrame) -> None:
    """Task 2.4: save histograms, pair plot, correlation matrix and TF-IDF chart."""
    print("\n" + "=" * 65)
    print("TASK 2.4 - EXPLORATORY DATA ANALYSIS")
    print("=" * 65)

    plt.style.use("ggplot")

    # A1. Title length distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(df["title_length"], kde=len(df) > 1, bins=min(5, max(1, len(df))))
    plt.title("Distribution of Article Title Lengths")
    plt.xlabel("Title Length (Characters)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_title_length_distribution.png", dpi=150)
    plt.close()

    # A2. Sentiment polarity distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(df["sentiment_polarity"], kde=len(df) > 1,
                 bins=min(5, max(1, len(df))))
    plt.title("Distribution of Article Sentiment Polarity")
    plt.xlabel("Polarity Score (-1.0 to 1.0)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_sentiment_distribution.png", dpi=150)
    plt.close()

    numeric_cols = [
        "title_length", "num_tokens", "num_sentences",
        "num_entities", "sentiment_polarity"
    ]
    pair = sns.pairplot(df[numeric_cols].dropna())
    pair.fig.suptitle("Pair Plot of Derived Features", y=1.02)
    pair.savefig(PLOTS_DIR / "03_pair_plot.png", dpi=150)
    plt.close("all")

    correlation = df[numeric_cols].corr(numeric_only=True)
    correlation.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
    print("\nCorrelation matrix:")
    print(correlation.round(2).to_string())

    # C. TF-IDF
    texts = df["text"].fillna("").astype(str)
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=20)
        matrix = vectorizer.fit_transform(texts)
        tfidf_df = pd.DataFrame(
            matrix.toarray(),
            columns=vectorizer.get_feature_names_out()
        )
        tfidf_sums = tfidf_df.sum().sort_values(ascending=False)
        tfidf_sums.rename("tfidf_score_sum").to_csv(
            OUTPUT_DIR / "tfidf_top_terms.csv"
        )

        plt.figure(figsize=(10, 5))
        sns.barplot(x=tfidf_sums.index, y=tfidf_sums.values)
        plt.title("Top 20 TF-IDF Terms Across Articles")
        plt.ylabel("TF-IDF Score Sum")
        plt.xlabel("Term")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "04_tfidf_top_terms.png", dpi=150)
        plt.close()
        print("\nTop TF-IDF terms:")
        print(tfidf_sums.head(20).to_string())
    except ValueError as exc:
        print(f"[WARNING] TF-IDF could not run: {exc}")
        print("This usually means the text is empty or contains no usable terms.")

    print(f"\nCharts saved in: {PLOTS_DIR}")
    print(f"Correlation matrix saved in: {OUTPUT_DIR / 'correlation_matrix.csv'}")


def run_extended_exercise(df: pd.DataFrame, nlp) -> None:
    """In-lab extension: verb count and mean sentence length."""
    print("\n" + "=" * 65)
    print("IN-LAB EXERCISE - EXTENDED FEATURES")
    print("=" * 65)
    extended = []
    for text in df["text"]:
        doc = nlp(text)
        sentences = list(doc.sents)
        extended.append({
            "num_verbs": sum(1 for token in doc if token.pos_ == "VERB"),
            "mean_sentence_length": (
                sum(len(sentence) for sentence in sentences) / len(sentences)
                if sentences else 0
            ),
        })
    extended_df = pd.concat(
        [df[["title"]].reset_index(drop=True), pd.DataFrame(extended)], axis=1
    )
    extended_df.to_csv(OUTPUT_DIR / "exercise_extended_features.csv", index=False)
    print(extended_df.to_string(index=False))
    print(f"Saved: {OUTPUT_DIR / 'exercise_extended_features.csv'}")


def run_tfidf_exercise(df: pd.DataFrame) -> None:
    """In-lab extension: compare TF-IDF with and without stop-word removal."""
    print("\n" + "=" * 65)
    print("IN-LAB EXERCISE - TF-IDF WITH/WITHOUT STOP WORDS")
    print("=" * 65)
    for label, stop_words in [
        ("with_stopword_removal", "english"),
        ("without_stopword_removal", None),
    ]:
        try:
            vectorizer = TfidfVectorizer(
                stop_words=stop_words, max_features=20
            )
            matrix = vectorizer.fit_transform(df["text"].fillna("").astype(str))
            scores = pd.Series(
                matrix.sum(axis=0).A1,
                index=vectorizer.get_feature_names_out()
            ).sort_values(ascending=False)
            scores.rename("tfidf_score_sum").to_csv(
                OUTPUT_DIR / f"exercise_tfidf_{label}.csv"
            )
            print(f"\n{label}:")
            print(scores.head(10).to_string())
        except ValueError as exc:
            print(f"[WARNING] {label}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lab 02: Web Scraping, NLP Feature Engineering and EDA"
    )
    parser.add_argument(
        "--scrape", action="store_true",
        help="Scrape the five URLs in the lab instead of using the supplied CSV."
    )
    parser.add_argument(
        "--exercises", action="store_true",
        help="Also run the two executable in-lab feature/TF-IDF exercises."
    )
    args = parser.parse_args()

    ensure_folders()
    run_numpy_demo()

    df = load_raw_data(use_scraper=args.scrape)
    if df.empty:
        raise RuntimeError("No article records are available. Check the input CSV.")

    nlp = load_spacy_model()
    df = add_features(df, nlp)
    df = add_sentiment(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = OUTPUT_DIR / "articles.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\nFinal engineered dataset saved: {output_csv}")
    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")

    save_eda(df)

    if args.exercises:
        run_extended_exercise(df, nlp)
        run_tfidf_exercise(df)

    print("\n" + "=" * 65)
    print("LAB 02 COMPLETED")
    print("=" * 65)
    print("Open the outputs folder to see the CSV files and graphs.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nPROJECT STOPPED: {error}")
        raise
