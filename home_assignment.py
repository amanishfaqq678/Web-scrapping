"""
HOME ASSIGNMENT SCAFFOLD
Scrape 20 articles from two blogs (10 per blog), then compare mean sentiment
polarity and mean noun density.

IMPORTANT:
The lab handout does not provide the 20 article URLs or confirmed selectors.
Fill in the URL lists and selectors below before running this file.
Run: python home_assignment.py
"""
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
import spacy
from bs4 import BeautifulSoup
from textblob import TextBlob

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
PLOTS = OUT / "plots"
OUT.mkdir(exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# Add TEN real article URLs for each source.
SOURCE_A = {
    "name": "blog-a",
    "urls": [
        # "https://example.com/article-1",
        # Add 9 more URLs
    ],
    "title_selector": ("h3", "post-title"),
    "content_selector": ("div", "post-body entry-content"),
}

SOURCE_B = {
    "name": "blog-b",
    "urls": [
        # "https://example.org/article-1",
        # Add 9 more URLs
    ],
    "title_selector": ("h1", None),
    "content_selector": ("div", None),
}


def extract_element(soup, selector):
    tag, css_class = selector
    return soup.find(tag, class_=css_class) if css_class else soup.find(tag)


def scrape_source(source):
    rows = []
    for url in source["urls"]:
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "lxml")

            title_tag = extract_element(soup, source["title_selector"])
            title = (
                title_tag.get_text(" ", strip=True)
                if title_tag else
                soup.title.get_text(" ", strip=True) if soup.title else "Untitled"
            )

            body = extract_element(soup, source["content_selector"])
            if body is None:
                print(f"[SKIP] Body selector did not match: {url}")
                continue

            text = " ".join(p.get_text(" ", strip=True) for p in body.find_all("p"))
            if len(text) > 100:
                rows.append({
                    "source": source["name"],
                    "url": url,
                    "title": title,
                    "text": text
                })
                print(f"[OK] {source['name']}: {title[:60]}")
            else:
                print(f"[SKIP] Text too short: {url}")
            time.sleep(1)
        except Exception as exc:
            print(f"[ERROR] {url}: {exc}")
    return pd.DataFrame(rows)


def main():
    if len(SOURCE_A["urls"]) != 10 or len(SOURCE_B["urls"]) != 10:
        raise ValueError(
            "Add exactly 10 real article URLs to SOURCE_A['urls'] and "
            "10 to SOURCE_B['urls'] before running."
        )

    corpus = pd.concat(
        [scrape_source(SOURCE_A), scrape_source(SOURCE_B)],
        ignore_index=True
    )
    if corpus.empty:
        raise RuntimeError("No articles were scraped. Check URLs and HTML selectors.")

    nlp = spacy.load("en_core_web_sm")
    corpus["sentiment_polarity"] = corpus["text"].apply(
        lambda text: TextBlob(text).sentiment.polarity
    )

    def noun_density(text):
        doc = nlp(text)
        words = [token for token in doc if token.is_alpha]
        nouns = [token for token in words if token.pos_ == "NOUN"]
        return len(nouns) / len(words) if words else 0.0

    corpus["noun_density"] = corpus["text"].apply(noun_density)
    corpus.to_csv(OUT / "two_source_corpus.csv", index=False, encoding="utf-8-sig")

    summary = corpus.groupby("source").agg(
        article_count=("url", "count"),
        mean_sentiment_polarity=("sentiment_polarity", "mean"),
        mean_noun_density=("noun_density", "mean"),
    )
    summary.to_csv(OUT / "two_source_summary.csv")
    print("\nGrouped results:")
    print(summary.round(4).to_string())

    summary[["mean_sentiment_polarity", "mean_noun_density"]].plot(
        kind="bar", figsize=(9, 5)
    )
    plt.title("Mean Sentiment Polarity and Noun Density by Blog")
    plt.ylabel("Mean score")
    plt.xlabel("Blog source")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PLOTS / "05_two_source_comparison.png", dpi=150)
    plt.close()

    if len(summary) == 2:
        sentiment_difference = abs(
            summary.iloc[0]["mean_sentiment_polarity"]
            - summary.iloc[1]["mean_sentiment_polarity"]
        )
        noun_difference = abs(
            summary.iloc[0]["mean_noun_density"]
            - summary.iloc[1]["mean_noun_density"]
        )
        print(f"\nAbsolute difference in mean polarity: {sentiment_difference:.4f}")
        print(f"Absolute difference in mean noun density: {noun_difference:.4f}")

    print("\nSaved two_source_corpus.csv, two_source_summary.csv and comparison chart.")


if __name__ == "__main__":
    main()
