"""News source integration for fetching financial news."""

from datetime import datetime

import feedparser


def fetch_rss_news(
    feeds: list[str] | None = None,
    max_articles: int = 20,
) -> list[dict]:
    """Fetch news articles from RSS feeds.

    Args:
        feeds: List of RSS feed URLs. Defaults to common financial news feeds.
        max_articles: Maximum number of articles to return.

    Returns:
        List of dicts with 'text', 'title', 'source', 'timestamp', 'link'.
    """
    default_feeds = feeds or [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    ]

    articles = []
    for feed_url in default_feeds:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:max_articles]:
                # Combine title and summary for analysis
                text = entry.get("title", "")
                if "summary" in entry:
                    text += "\n" + entry["summary"]

                timestamp = entry.get("published", datetime.now().isoformat())

                articles.append({
                    "text": text,
                    "title": entry.get("title", ""),
                    "source": feed.feed.get("title", feed_url),
                    "timestamp": timestamp,
                    "link": entry.get("link", ""),
                })
        except Exception:
            continue

    return articles[:max_articles]
