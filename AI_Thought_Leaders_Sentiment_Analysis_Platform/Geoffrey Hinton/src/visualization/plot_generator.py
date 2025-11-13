from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Resolve paths relative to the repo root (…/illyasur)
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]  # src/visualization/... -> go up two levels
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

def load_data(filepath=None):
    """Load the sentiment-analyzed tweet data."""
    if filepath is None:
        candidates = [
            PROCESSED_DIR / "sentiment_result.csv",   # preferred fixed name
          
        ]
        for p in candidates:
            if p.exists():
                print(f" Loading: {p}")
                return pd.read_csv(p)
        # Last resort: newest sentiment_*.csv
        matches = sorted(PROCESSED_DIR.glob("sentiment_*.csv"))
        if matches:
            print(f" Loading latest: {matches[-1]}")
            return pd.read_csv(matches[-1])
        raise FileNotFoundError(f"Could not find sentiment_result(s).csv in {PROCESSED_DIR}")

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f" File not found: {filepath}")
    print(f" Loading: {filepath}")
    return pd.read_csv(filepath)

def sentiment_distribution(df, output_dir=RESULTS_DIR):
    """Plot overall sentiment distribution."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    order = [c for c in ["negative", "neutral", "positive"] if c in df["sentiment_label"].unique()]
    counts = df["sentiment_label"].value_counts()
    if order:
        counts = counts.reindex(order, fill_value=0)
    counts.plot(kind="bar", color=["#d62728", "#ff7f0e", "#2ca02c"][:len(counts)])
    plt.title("Overall Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Tweet Count")
    plt.tight_layout()
    out = output_dir / "sentiment_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f" Saved: {out}")

def sentiment_by_user(df, output_dir=RESULTS_DIR):
    """Plot sentiment comparison by researcher (auto-detect username column)."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    possible_user_cols = ["username", "author", "user", "name", "profile"]
    user_col = next((col for col in df.columns if col.lower() in possible_user_cols), None)

    if not user_col:
        print(" No username/author column found. Skipping user-level plot.")
        return

    print(f" Using '{user_col}' as researcher identifier...")
    pivot = df.groupby([user_col, "sentiment_label"]).size().unstack(fill_value=0)

    plt.figure(figsize=(10, 5))
    pivot.plot(kind="bar", stacked=True, ax=plt.gca(), colormap="tab20c")
    plt.title("Sentiment Comparison by Researcher")
    plt.xlabel("Researcher")
    plt.ylabel("Tweet Count")
    plt.tight_layout()
    out = output_dir / "sentiment_by_user.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f" Saved: {out}")

def sentiment_over_time(df, output_dir=RESULTS_DIR):
    """Plot sentiment trend over time (monthly)."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    if "timestamp" not in df.columns:
        print(" No 'timestamp' column. Skipping time trend plot.")
        return

    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    mask = ts.notna()
    if mask.sum() == 0:
        print(" No valid timestamps after parsing. Skipping time trend plot.")
        return

    tmp = df.loc[mask].copy()
    tmp["month"] = ts[mask].dt.to_period("M")
    trend = tmp.groupby(["month", "sentiment_label"]).size().unstack(fill_value=0)
    trend = trend.sort_index()
    trend.index = trend.index.astype(str)

    plt.figure(figsize=(10, 5))
    trend.plot(ax=plt.gca(), marker="o")
    plt.title("Sentiment Trend Over Time (Monthly)")
    plt.xlabel("Month")
    plt.ylabel("Number of Tweets")
    plt.tight_layout()
    out = output_dir / "sentiment_trend.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f" Saved: {out}")

def main():
    df = load_data()
    sentiment_distribution(df)
    sentiment_by_user(df)
    sentiment_over_time(df)
    print(" Visualization pipeline completed successfully!")

if __name__ == "__main__":
    main()