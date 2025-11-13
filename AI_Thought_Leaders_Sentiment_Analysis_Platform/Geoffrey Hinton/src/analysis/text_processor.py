import re
from pathlib import Path

import pandas as pd
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

def clean_text(text: str) -> str:
    """Remove URLs, mentions, hashtags, emojis, and extra spaces."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_english(text: str) -> bool:
    """Detect if a tweet is English."""
    try:
        return detect(text) == "en"
    except:
        return False


def find_time_column(columns):
    """Try to find which column represents time/date."""
    candidates = ["timestamp", "date", "time", "created_at", "posted"]
    for col in columns:
        if col.lower() in candidates:
            return col
    return None


def _find_project_root(start: Path) -> Path:
    # Walk up to find a folder that contains "data". If not found, fallback two levels up.
    for p in [start] + list(start.parents):
        if (p / "data").is_dir():
            return p
    return start.parents[2]  # adjust if your layout is different


def process_tweets(raw_dir=None, output_path=None):
    here = Path(__file__).resolve()
    project_root = _find_project_root(here)

    # Defaults relative to repo root
    raw_dir = Path(raw_dir) if raw_dir else project_root / "data" / "raw"
    output_path = Path(output_path) if output_path else project_root / "data" / "processed" / "cleaned_tweets.csv"

    print(f" Loading raw CSV files from: {raw_dir}")
    if not raw_dir.exists():
        raise FileNotFoundError(f"[WinError 3] The system cannot find the path specified: '{raw_dir}'")

    files = [p for p in raw_dir.iterdir() if p.suffix.lower() == ".csv"]

    if not files:
        print(f"No CSV files found in {raw_dir}")
        return

    all_tweets = []
    for file in files:
        print(f" Reading: {file}")
        df = pd.read_csv(file)

        username = file.stem.split("_tweets")[0]
        df["username"] = username

        all_tweets.append(df)

    tweets = pd.concat(all_tweets, ignore_index=True)
    print(f"Loaded {len(tweets)} total tweets")

    print(" Cleaning text...")
    text_col = "text" if "text" in tweets.columns else ("content" if "content" in tweets.columns else None)
    if text_col is None:
        raise ValueError(f"Could not find a text/content column in columns: {list(tweets.columns)}")
    tweets["clean_text"] = tweets[text_col].astype(str).apply(clean_text)

    print(" Filtering English tweets...")
    tweets = tweets[tweets["clean_text"].apply(is_english)]

    print(" Normalizing timestamps...")
    time_col = find_time_column(tweets.columns)
    if time_col:
        print(f" Found time column: {time_col}")
        tweets[time_col] = pd.to_datetime(tweets[time_col], errors="coerce", utc=True)
        tweets.dropna(subset=[time_col], inplace=True)
        tweets["timestamp"] = tweets[time_col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        print(" No timestamp/date column found — skipping timestamp normalization.")
        tweets["timestamp"] = pd.NaT

    print(" Removing duplicates...")
    tweets.drop_duplicates(subset=["clean_text"], inplace=True)
    tweets.dropna(subset=["clean_text"], inplace=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tweets.to_csv(output_path, index=False, encoding="utf-8")
    print(f" Saved cleaned data to {output_path}")
    print(f" Final cleaned tweets: {len(tweets)}")


if __name__ == "__main__":
    process_tweets()