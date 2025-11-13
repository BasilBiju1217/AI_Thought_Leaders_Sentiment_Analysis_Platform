from pathlib import Path
from datetime import datetime
import os
import json
import csv
import re

import pandas as pd
import pytest


import analysis as ana

# ------------------------ Unit tests: text processing ----------------------- #

def test_clean_text_removes_noise():
    raw = "Check this out: https://ex.com @user #AI 🔥🔥 Hello!!!"
    cleaned = ana.clean_text(raw)
    # URLs, mentions, emojis, punctuation removed; '#' removed but word kept
    assert cleaned == "Check this out AI Hello"


def test_is_english_true_false():
    assert ana.is_english("This is an English sentence with simple words.") is True
    assert ana.is_english("Bonjour tout le monde") is False


def test_find_time_column_various_cases():
    cols = ["ID", "Created_At", "user", "text"]
    found = ana.find_time_column(cols)
    assert found == "Created_At"

    cols2 = ["posted", "content"]
    found2 = ana.find_time_column(cols2)
    assert found2 == "posted"

    cols3 = ["published_on", "when"]
    assert ana.find_time_column(cols3) is None


# --------------------- End-to-end test: process_tweets ---------------------- #

def test_process_tweets_end_to_end(tmp_path: Path):
    # Prepare raw dir with one CSV
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_csv = tmp_path / "data" / "processed" / "cleaned_tweets.csv"

    df_raw = pd.DataFrame(
        {
            "text": [
                "Hello world @user http://ex.com #ML!!!",  # English; will clean to "Hello world ML"
                "Bonjour le monde",                          # Non-English; should be filtered out
                "Hello world!!!",                            # Duplicate after cleaning
            ],
            "created_at": [
                "2024-01-01T10:00:00Z",
                "2024-01-01T11:00:00Z",
                "2024-01-01T12:00:00Z",
            ],
        }
    )
    src_file = raw_dir / "alice_tweets.csv"
    df_raw.to_csv(src_file, index=False, encoding="utf-8")

    # Run processing (override dirs to avoid touching real repo paths)
    ana.process_tweets(raw_dir=raw_dir, output_path=out_csv)

    # Validate output
    assert out_csv.exists(), "cleaned_tweets.csv was not created"
    df = pd.read_csv(out_csv)

    # Should only keep English; duplicates removed => 1 row expected
    assert len(df) == 1
    assert "username" in df.columns
    assert df.loc[0, "username"] == "alice"
    assert "clean_text" in df.columns
    assert df.loc[0, "clean_text"] == "Hello world ML"

    # Timestamp normalized to ISO Z
    assert "timestamp" in df.columns
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", str(df.loc[0, "timestamp"])) is not None


def test_process_tweets_no_csvs(tmp_path: Path, capsys):
    raw_dir = tmp_path / "empty_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_csv = tmp_path / "processed" / "cleaned_tweets.csv"

    # No CSVs present -> should not raise; just return without file
    ana.process_tweets(raw_dir=raw_dir, output_path=out_csv)
    captured = capsys.readouterr().out
    assert "No CSV files found" in captured
    assert not out_csv.exists()


# ------------------------ Unit tests: sentiment I/O ------------------------- #

def test_load_clean_data_and_missing(tmp_path: Path):
    good = tmp_path / "cleaned_tweets.csv"
    pd.DataFrame({"clean_text": ["hello"], "timestamp": ["2024-01-01T00:00:00Z"]}).to_csv(good, index=False)

    df_loaded = ana.load_clean_data(good)
    assert len(df_loaded) == 1

    missing = tmp_path / "nope.csv"
    with pytest.raises(FileNotFoundError):
        ana.load_clean_data(missing)


def test_save_results_and_overwrite(tmp_path: Path):
    out = tmp_path / "sentiment_result.csv"
    df1 = pd.DataFrame({"clean_text": ["a"], "sentiment_label": ["POSITIVE"], "sentiment_score": [0.9]})
    df2 = pd.DataFrame({"clean_text": ["b"], "sentiment_label": ["NEGATIVE"], "sentiment_score": [0.8]})

    ana.save_results(df1, out)
    assert out.exists()
    first = pd.read_csv(out)
    assert list(first["clean_text"]) == ["a"]

    # Overwrite with second
    ana.save_results(df2, out)
    second = pd.read_csv(out)
    assert list(second["clean_text"]) == ["b"]


def test_cleanup_old_outputs(tmp_path: Path, monkeypatch):
    # Create files matching the cleanup patterns
    p1 = tmp_path / "sentiment_partial_001.csv"
    p2 = tmp_path / "sentiment_results_20240101.csv"
    p1.write_text("x")
    p2.write_text("y")
    assert p1.exists() and p2.exists()

    ana.cleanup_old_outputs(tmp_path)
    assert not p1.exists() and not p2.exists()


# --------------------- Sentiment analysis: mocked pipeline ------------------ #

def test_analyze_sentiment_with_mocked_pipeline_success_and_error(monkeypatch):
    # Build sample dataframe with 3 texts. We'll set batch_size=2 so the second batch errors.
    df = pd.DataFrame({"clean_text": ["It is a good day", "This is bad", "Meh"]})

    # Fake HF pipeline: first batch returns scores; second batch raises.
    class FakeCLF:
        def __init__(self):
            self.calls = 0
        def __call__(self, texts, truncation=None, max_length=None, padding=None):
            self.calls += 1
            if self.calls == 1:
                out = []
                for t in texts:
                    if "bad" in t.lower():
                        out.append({"label": "NEGATIVE", "score": 0.8})
                    else:
                        out.append({"label": "POSITIVE", "score": 0.9})
                return out
            raise RuntimeError("boom on second batch")

    def fake_pipeline(task, model=None, device=None, **kwargs):
        assert task == "sentiment-analysis"
        return FakeCLF()

    # Patch the imported pipeline symbol in the module
    monkeypatch.setattr(ana, "pipeline", fake_pipeline, raising=True)

    out = ana.analyze_sentiment(
        df.copy(),
        text_col="clean_text",
        batch_size=2,
        device=-1,
        model_name="fake-model",
        save_partial=False,
    )

    assert len(out) == 3
    assert {"sentiment_label", "sentiment_score"}.issubset(out.columns)

    # First two resolved by fake model; third filled with ERROR due to second-batch failure
    assert out.loc[0, "sentiment_label"] == "POSITIVE"
    assert out.loc[1, "sentiment_label"] == "NEGATIVE"
    assert out.loc[2, "sentiment_label"] == "ERROR"
    assert out.loc[2, "sentiment_score"] == 0.0


def test_analyze_sentiment_missing_text_col_raises():
    df = pd.DataFrame({"text": ["hello"]})
    with pytest.raises(KeyError):
        ana.analyze_sentiment(df, text_col="clean_text", batch_size=2, device=-1)