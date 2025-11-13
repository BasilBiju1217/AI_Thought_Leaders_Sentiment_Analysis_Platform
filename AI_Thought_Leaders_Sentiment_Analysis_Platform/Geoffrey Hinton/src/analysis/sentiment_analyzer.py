import time
import pandas as pd
from datetime import datetime
from pathlib import Path
from transformers import pipeline
import traceback

# Resolve paths relative to the repo root (…/illyasur)
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]  # src/analysis/... -> go up two levels
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_INPUT = PROCESSED_DIR / "cleaned_tweets.csv"
FIXED_OUTPUT = PROCESSED_DIR / "sentiment_result.csv"

def load_clean_data(filepath=DEFAULT_INPUT):
    """Load cleaned tweet dataset."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f" Cleaned file not found: {filepath}")
    print(f" Loading cleaned data from {filepath}")
    df = pd.read_csv(filepath)
    print(f" Loaded {len(df)} cleaned tweets.\n")
    return df

def analyze_sentiment(
    df,
    text_col="clean_text",
    batch_size=8,
    device=-1,  # CPU by default; use 0 for first CUDA GPU
    model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
    save_partial=False,   # disabled to ensure only one final file
    partial_every=10,     # ignored when save_partial=False
    sleep=0.0,            # set to 0.5 if you want to throttle
):
    """Run sentiment analysis on cleaned tweets."""
    if text_col not in df.columns:
        raise KeyError(f"Column '{text_col}' not found in dataset.")

    texts = df[text_col].fillna("").astype(str).tolist()
    n = len(texts)
    print(f" Running sentiment analysis on {n} tweets (batch_size={batch_size})")

    print(" Initializing sentiment model (this may take a moment)...")
    clf = pipeline(
        "sentiment-analysis",
        model=model_name,
        device=device,
    )
    print(" Sentiment model initialized successfully!\n")

    results = []
    failed_batches = []
    total_batches = (n + batch_size - 1) // batch_size

    for i in range(0, n, batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_index = i // batch_size + 1
        print(f" Processing batch {batch_index}/{total_batches} ({i}-{i+len(batch_texts)-1})...")

        try:
            output = clf(batch_texts, truncation=True, max_length=128, padding=True)
            if not output or not isinstance(output, list):
                raise ValueError(f"No valid output from model in batch {batch_index}")
            results.extend(output)
        except Exception as e:
            tb = traceback.format_exc()
            print(f" Error in batch {batch_index}: {e}\n{tb}")
            failed_batches.append((batch_index, str(e)))
            results.extend([{"label": "ERROR", "score": 0.0}] * len(batch_texts))

        if save_partial and batch_index % partial_every == 0:
            # Not used (save_partial=False) to ensure only one final output file
            pass

        if sleep:
            time.sleep(sleep)

    if len(results) < len(df):
        print(f" Only {len(results)} results for {len(df)} tweets — trimming dataframe to match.")
        df = df.iloc[:len(results)].reset_index(drop=True)

    df["sentiment_label"] = [r.get("label", "ERROR") for r in results]
    df["sentiment_score"] = [r.get("score", 0.0) for r in results]

    print(f"\n Sentiment analysis complete for {len(df)} tweets.")
    if failed_batches:
        print(f" Failed batches: {failed_batches}")

    return df

def cleanup_old_outputs(output_dir=PROCESSED_DIR):
    """Remove prior partial/timestamped outputs so only the fixed file remains."""
    for pattern in ("sentiment_partial_*.csv", "sentiment_results_*.csv"):
        for p in Path(output_dir).glob(pattern):
            try:
                p.unlink()
                print(f" Removed old file: {p}")
            except Exception:
                pass

def save_results(df, output_path=FIXED_OUTPUT):
    """Save final results to a single, fixed file name (overwrites existing)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp")  # write-then-rename for safety
    df.to_csv(tmp_path, index=False, encoding="utf-8")
    if output_path.exists():
        output_path.unlink()
    tmp_path.rename(output_path)
    print(f" Final sentiment results saved to {output_path}")
    return output_path

def main():
    try:
        # Optional: clean up old outputs so you end up with only one file afterward
        cleanup_old_outputs(PROCESSED_DIR)

        df = load_clean_data(DEFAULT_INPUT)
        df = analyze_sentiment(df, batch_size=8, device=-1, save_partial=False)
        output_file = save_results(df, FIXED_OUTPUT)
        print("\n Sentiment pipeline completed successfully!")
        print(f"Output file: {output_file}")
    except Exception:
        print("\n Fatal error during execution:")
        traceback.print_exc()

if __name__ == "__main__":
    main()