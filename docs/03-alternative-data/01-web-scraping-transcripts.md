---
layout: default
title: "Web Scraping & Transcript Parsing"
parent: "Part 3: Alternative Data"
nav_order: 1
---

# Web Scraping & Transcript Parsing

The most accessible alternative data comes from public websites and corporate communications. This chapter covers two core tools: `WebScraper` for fetching structured data from the web, and `TranscriptParser` for extracting actionable information from earnings call transcripts.

## Sourcing Strategies

Before writing code, consider the landscape of data sources available to you:

| Source Type | Examples | Access Method | Latency |
|-------------|----------|---------------|---------|
| SEC filings | 10-K, 10-Q, 8-K | EDGAR API / scraping | Minutes to hours |
| Earnings calls | Quarterly transcripts | Vendor APIs / scraping | Hours to days |
| Financial tables | Revenue, margins, guidance | Web scraping | Varies |
| News articles | Press releases, analysis | RSS / API | Seconds to minutes |

{: .note }
Public SEC filings via EDGAR are freely available and carry no licensing restrictions. They are an excellent starting point for alternative data research.

## Web Scraping with WebScraper

The `WebScraper` class handles the technical details of fetching and parsing web content. It manages rate limiting, retries, user-agent rotation, and HTML-to-DataFrame conversion so you can focus on the data.

### Basic Setup

```python
from puffin.data.scraper import WebScraper

scraper = WebScraper(
    rate_limit=1.0,      # Wait 1 second between requests
    max_retries=3,       # Retry failed requests 3 times
    timeout=10,          # 10 second timeout
)
```

The constructor accepts three parameters that control politeness and reliability:

- **rate_limit**: Minimum seconds between requests to the same domain. Setting this too low risks getting blocked.
- **max_retries**: Number of retry attempts with exponential backoff on transient failures (HTTP 429, 500, 503).
- **timeout**: Seconds to wait for a response before giving up.

### Scraping SEC Filings

SEC EDGAR is the primary source for corporate filings. The `scrape_sec_filings` method fetches filing metadata for a given company:

```python
from puffin.data.scraper import WebScraper

scraper = WebScraper(
    rate_limit=1.0,
    max_retries=3,
    timeout=10,
)

# Scrape SEC filings
filings = scraper.scrape_sec_filings(
    cik="0000320193",    # Apple's CIK
    form_type="10-K"     # Annual reports
)

print(filings.head())
```

Output:
```
  filing_date form_type                                        url accession_number
0  2024-01-15      10-K  https://www.sec.gov/Archives/edgar/...  0000320193240001
1  2023-01-16      10-K  https://www.sec.gov/Archives/edgar/...  0000320193230001
```

Each row contains the filing date, form type, URL to the full filing, and the unique accession number. You can pass `form_type="10-Q"` for quarterly reports or `form_type="8-K"` for material event disclosures.

### Scraping Financial Tables

Many websites display financial data in HTML tables. The scraper converts these directly into pandas DataFrames:

```python
# Scrape financial tables from a webpage
url = "https://example.com/company/financials"
data = scraper.scrape_financial_tables(url)

print(data)
```

The scraper automatically:
- Rotates user agents to avoid detection
- Implements exponential backoff for retries
- Respects rate limits to avoid blocking
- Parses HTML tables into pandas DataFrames

{: .tip }
For production use, consider commercial data APIs (Quandl, Alpha Vantage, IEX Cloud) which provide cleaner data and better reliability than scraping.

### Handling Common Scraping Pitfalls

Web scraping is inherently fragile. Here are common issues and how `WebScraper` handles them:

{: .warning }
Websites change their HTML structure without notice. Always validate scraped data before feeding it into models. A silent schema change can corrupt your pipeline.

1. **Rate limiting (HTTP 429)**: The scraper waits and retries with exponential backoff.
2. **Stale content**: Always compare scraped data against a known baseline for the first run.
3. **Incomplete tables**: The scraper returns partial results and logs warnings for malformed HTML.
4. **CAPTCHAs and bot detection**: No automated solution -- switch to a commercial API if this occurs frequently.

## Earnings Call Analysis

Earnings calls reveal management sentiment and forward guidance before it appears in financial statements. The prepared remarks section tends to be scripted and optimistic, while the Q&A session is more candid and revealing.

### Parsing Transcripts with TranscriptParser

The `TranscriptParser` class splits raw transcript text into structured sections and identifies participants:

```python
from puffin.data.transcript_parser import TranscriptParser

parser = TranscriptParser()

# Sample earnings call transcript
transcript_text = """
John Smith - CEO: Good morning. I'm pleased to report strong revenue
growth of 15% year-over-year. Our revenue was $1.2 billion this quarter,
exceeding guidance. We remain optimistic about future growth.

Operator: We'll now begin the question-and-answer session.

Jane Analyst: Can you discuss the margin outlook?

John Smith - CEO: We expect margins to improve as we scale operations.
"""

# Parse into sections
parsed = parser.parse(transcript_text)

print("Participants:", parsed["participants"])
print("\nPrepared Remarks Length:", len(parsed["prepared_remarks"]))
print("Q&A Session Length:", len(parsed["qa_session"]))
```

Output:
```
Participants: ['Jane Analyst', 'John Smith']

Prepared Remarks Length: 245
Q&A Session Length: 178
```

The parser identifies the Q&A boundary using the "Operator" keyword and extracts unique participant names from speaker labels.

### Extracting Financial Metrics

The parser identifies mentioned metrics automatically, picking up dollar amounts, percentages, and growth figures:

```python
metrics = parser.extract_metrics(parsed)

for metric in metrics:
    print(f"{metric['metric_type']}: {metric['value']} {metric['unit']}")
```

Output:
```
revenue: 1.2 billion
growth: 15.0 None
```

This extraction uses pattern matching for common financial phrasing. It recognizes revenue, earnings, margins, growth rates, and guidance figures. For more nuanced extraction (e.g., distinguishing organic vs. inorganic growth), consider fine-tuning an NLP model on financial text.

### Sentiment Analysis of Transcript Sections

Analyze management tone across the prepared remarks and Q&A sections:

```python
sentiment = parser.sentiment_sections(parsed)

for section, scores in sentiment.items():
    print(f"\n{section}:")
    print(f"  Score: {scores['score']:.2f} (-1 negative, +1 positive)")
    print(f"  Positive words: {scores['positive_count']}")
    print(f"  Negative words: {scores['negative_count']}")
```

Output:
```
prepared_remarks:
  Score: 0.60 (-1 negative, +1 positive)
  Positive words: 4
  Negative words: 1

qa_session:
  Score: 0.33 (-1 negative, +1 positive)
  Positive words: 1
  Negative words: 0
```

{: .note }
This is basic sentiment analysis using word-counting heuristics. For production, consider NLP libraries (spaCy, transformers) or sentiment models trained on financial text. The Loughran-McDonald dictionary is a widely used lexicon for financial sentiment.

### Interpreting Sentiment Signals

A few rules of thumb when turning transcript sentiment into trading signals:

- **Prepared remarks sentiment** is usually positive and less informative. Look for deviations from the company's historical baseline.
- **Q&A sentiment** tends to be more revealing. A defensive or evasive tone during Q&A can signal trouble.
- **Sentiment change** (quarter-over-quarter) is often more predictive than the absolute level.
- **CEO vs. CFO tone divergence** can indicate internal disagreement about the business outlook.

## Exercises

1. Scrape SEC filings for 3 companies and compare filing frequencies
2. Parse an earnings transcript and extract all mentioned financial metrics
3. Compare prepared remarks sentiment vs. Q&A sentiment across multiple quarters for one company

## Source Code

- [`puffin/data/scraper.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/data/scraper.py) -- WebScraper implementation
- [`puffin/data/transcript_parser.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/data/transcript_parser.py) -- TranscriptParser implementation
