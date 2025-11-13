import os
import json
import csv
from datetime import datetime, timedelta

import pytest

# Adjust this line if your module is not named 'scrapper.py'
import scrapper
from selenium.webdriver.common.by import By


def test_hash_username_deterministic_and_format():
    h1 = scrapper.hash_username("alice")
    h2 = scrapper.hash_username("alice")
    h3 = scrapper.hash_username("bob")
    assert h1 == h2, "Hash should be deterministic for same input"
    assert h1 != h3, "Different usernames should produce different hashes"
    assert isinstance(h1, str) and len(h1) == 16, "Expect 16-char hex string"


def test_detect_language_basic(monkeypatch):
    assert scrapper.detect_language("This is a test of English.") == "en"

    # Force the internal detect() to raise to test fallback
    def boom(_):
        raise RuntimeError("langdetect failure")

    monkeypatch.setattr(scrapper, "detect", boom, raising=True)
    assert scrapper.detect_language("whatever") == "en", "Should default to 'en' on errors"


def test_parse_date_title_format():
    s = "Mar 23, 2025 · 5:15 PM UTC"
    dt = scrapper.parse_date(s)
    assert isinstance(dt, datetime)
    assert dt.year == 2025 and dt.month == 3 and dt.day == 23 and dt.hour == 17 and dt.minute == 15


def test_parse_date_relative_times():
    now = datetime.now()
    two_h_ago = scrapper.parse_date("2h ago")
    assert now - timedelta(hours=2, minutes=5) <= two_h_ago <= now - timedelta(hours=2) + timedelta(minutes=5)

    fifteen_m_ago = scrapper.parse_date("15m ago")
    assert now - timedelta(minutes=20) <= fifteen_m_ago <= now - timedelta(minutes=10)

    three_d_ago = scrapper.parse_date("3d ago")
    assert now - timedelta(days=3, hours=1) <= three_d_ago <= now - timedelta(days=2, hours=23)


def test_parse_date_absolute_formats():
    exact = scrapper.parse_date("Dec 25, 2023")
    assert exact.year == 2023 and exact.month == 12 and exact.day == 25

    # "Dec 25" should resolve to this year or last year depending on whether that date has passed
    dt = scrapper.parse_date("Dec 25")
    assert dt.month == 12 and dt.day == 25
    now = datetime.now()
    # Allowed years: current or previous (per function logic)
    assert dt.year in {now.year, now.year - 1}


def test_parse_date_invalid_returns_nowish():
    before = datetime.now()
    dt = scrapper.parse_date("not a date")
    after = datetime.now()
    assert before <= dt <= after, "Invalid date should return current time"


def test_extract_tweet_id_basic():
    url = "https://nitter.net/user/status/1234567890123456789"
    assert scrapper.extract_tweet_id(url) == "1234567890123456789"
    assert scrapper.extract_tweet_id("https://nitter.net/user/status/") is None
    assert scrapper.extract_tweet_id(None) is None


# --- Fakes for Selenium elements ---

class FakeElement:
    def __init__(self, text=None, href=None, children=None):
        self._text = text
        self._href = href
        self.children = children or {}

    @property
    def text(self):
        return self._text or ""

    def get_attribute(self, name):
        if name == "href":
            return self._href
        return None

    def find_element(self, by, selector):
        key = (by, selector)
        if key in self.children:
            return self.children[key]
        raise Exception(f"child not found for {key}")

    def find_elements(self, by, selector):
        key = (by, selector)
        return self.children.get(key, [])


class FakeContainer:
    """Container that supports both find_elements and find_element mappings."""
    def __init__(self, selectors_to_lists=None, selectors_to_elements=None):
        self.selectors_to_lists = selectors_to_lists or {}
        self.selectors_to_elements = selectors_to_elements or {}

    def find_elements(self, by, selector):
        return self.selectors_to_lists.get((by, selector), [])

    def find_element(self, by, selector):
        key = (by, selector)
        if key in self.selectors_to_elements:
            return self.selectors_to_elements[key]
        raise Exception(f"element not found for {key}")


def test_extract_urls_filters_internal_links():
    # Two links: one external, one internal (should be filtered out)
    external = FakeElement(href="https://example.com/page")
    internal = FakeElement(href="/user/status/123")
    tweet_el = FakeElement(children={
        (By.CSS_SELECTOR, ".tweet-content a"): [external, internal]
    })

    urls = scrapper.extract_urls(tweet_el)
    assert urls == ["https://example.com/page"], "Should only keep absolute external URLs"


def test_is_retweet_and_is_quote_tweet():
    cont_rt = FakeContainer(selectors_to_lists={
        (By.CSS_SELECTOR, ".retweet-header"): [object()]
    })
    assert scrapper.is_retweet(cont_rt) is True
    assert scrapper.is_quote_tweet(cont_rt) is False

    cont_quote = FakeContainer(selectors_to_lists={
        (By.CSS_SELECTOR, ".quote"): [object()]
    })
    assert scrapper.is_quote_tweet(cont_quote) is True
    assert scrapper.is_retweet(cont_quote) is False

    cont_none = FakeContainer()
    assert scrapper.is_retweet(cont_none) is False
    assert scrapper.is_quote_tweet(cont_none) is False


def _build_engagement_container(comment_text="1.2K", retweet_text="456", like_text="2m"):
    # icon element -> parent via .. with actual text
    comment_icon = FakeElement(children={(By.XPATH, ".."): FakeElement(text=comment_text)})
    retweet_icon = FakeElement(children={(By.XPATH, ".."): FakeElement(text=retweet_text)})
    like_icon = FakeElement(children={(By.XPATH, ".."): FakeElement(text=like_text)})

    return FakeContainer(selectors_to_elements={
        (By.CSS_SELECTOR, ".icon-comment"): comment_icon,
        (By.CSS_SELECTOR, ".icon-retweet"): retweet_icon,
        (By.CSS_SELECTOR, ".icon-heart"): like_icon,
    })


def test_extract_engagement_stats_parsing():
    container = _build_engagement_container(comment_text="1.2K", retweet_text="456", like_text="2m")
    stats = scrapper.extract_engagement_stats(container)
    assert stats["comment_count"] == 1200
    assert stats["retweet_count"] == 456
    assert stats["like_count"] == 2_000_000


def test_extract_engagement_stats_missing_icon_graceful():
    # Only provide like icon; others missing should default to 0
    like_icon = FakeElement(children={(By.XPATH, ".."): FakeElement(text="99")})
    container = FakeContainer(selectors_to_elements={
        (By.CSS_SELECTOR, ".icon-heart"): like_icon,
    })
    stats = scrapper.extract_engagement_stats(container)
    assert stats == {"retweet_count": 0, "like_count": 99, "comment_count": 0}


def test_export_to_csv_roundtrip(tmp_path, monkeypatch):
    # Work in a temp dir so we don't write to project root
    monkeypatch.chdir(tmp_path)

    tweets = [
        {
            "tweet_id": "1",
            "text": "Hello world",
            "created_at": "2024-01-01 00:00:00",
            "lang": "en",
            "user_id_hashed": "abcd1234",
            "retweet_count": 1,
            "like_count": 2,
            "comment_count": 3,
            "is_reply": False,
            "is_retweet": False,
            "is_quote": False,
            "urls": ["https://example.com", "https://another.example"],
        }
    ]

    fn = scrapper.export_to_csv(tweets, username="tester")
    assert os.path.exists(fn), "CSV file was not created"

    # Read back and validate content
    with open(fn, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    row = rows[0]
    # All schema fields present
    for col in ["tweet_id", "text", "created_at", "lang", "user_id_hashed",
                "retweet_count", "like_count", "comment_count",
                "is_reply", "is_retweet", "is_quote", "urls"]:
        assert col in row, f"Missing column {col}"

    # URLs should be JSON-serialized array
    parsed_urls = json.loads(row["urls"])
    assert parsed_urls == tweets[0]["urls"]


# --------------------- Optional live integration test --------------------- #
# Requires: working Chrome + chromedriver + network access + Nitter reachable.
# By default this test is skipped. Run with:
#   RUN_SELENIUM_TESTS=1 pytest -q tests/test_scrapper.py -m integration

@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_SELENIUM_TESTS") != "1",
                    reason="Set RUN_SELENIUM_TESTS=1 to enable Selenium integration test.")
def test_scrape_x_posts_integration_smoke():
    posts = scrapper.scrape_x_posts(username="nasa", num_scrolls=1, tweet_type="original")
    assert isinstance(posts, list)
    # It's possible to get 0 if rate-limited; allow empty but check schema if non-empty
    if posts:
        p = posts[0]
        for k in ["tweet_id", "text", "created_at", "lang", "user_id_hashed",
                  "retweet_count", "like_count", "comment_count",
                  "is_reply", "is_retweet", "is_quote", "urls"]:
            assert k in p, f"Missing key in scraped post: {k}"