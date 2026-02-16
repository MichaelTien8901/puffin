"""Tests for web scraper."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

from puffin.data.scraper import WebScraper


@pytest.fixture
def scraper():
    return WebScraper(rate_limit=0.1, max_retries=2)


@pytest.fixture
def mock_html_table():
    return """
    <html>
        <body>
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Revenue</th>
                        <th>EPS</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>2024-01-01</td>
                        <td>$1.2B</td>
                        <td>$2.50</td>
                    </tr>
                    <tr>
                        <td>2024-04-01</td>
                        <td>$1.3B</td>
                        <td>$2.75</td>
                    </tr>
                </tbody>
            </table>
        </body>
    </html>
    """


@pytest.fixture
def mock_sec_html():
    return """
    <html>
        <body>
            <table class="tableFile2">
                <tr><th>Filings</th><th>Format</th><th>Description</th><th>Filing Date</th></tr>
                <tr>
                    <td>10-K</td>
                    <td><a href="/Archives/edgar/data/0000320193/000032019324000001.txt">Documents</a></td>
                    <td>Annual Report</td>
                    <td>2024-01-15</td>
                </tr>
                <tr>
                    <td>10-Q</td>
                    <td><a href="/Archives/edgar/data/0000320193/000032019324000002.txt">Documents</a></td>
                    <td>Quarterly Report</td>
                    <td>2024-04-15</td>
                </tr>
            </table>
        </body>
    </html>
    """


def test_scraper_initialization():
    scraper = WebScraper(rate_limit=2.0, max_retries=5, timeout=30)
    assert scraper.rate_limit == 2.0
    assert scraper.max_retries == 5
    assert scraper.timeout == 30


def test_get_headers(scraper):
    headers1 = scraper._get_headers()
    headers2 = scraper._get_headers()

    # Should have User-Agent
    assert "User-Agent" in headers1
    assert "User-Agent" in headers2

    # Should rotate user agents
    assert headers1["User-Agent"] != headers2["User-Agent"]


@patch("puffin.data.scraper.requests.get")
def test_fetch_with_retry_success(mock_get, scraper):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = b"<html><body>Test</body></html>"
    mock_get.return_value = mock_response

    response = scraper._fetch_with_retry("https://example.com")

    assert response is not None
    assert response.status_code == 200
    mock_get.assert_called_once()


@patch("puffin.data.scraper.requests.get")
def test_fetch_with_retry_failure(mock_get, scraper):
    mock_get.side_effect = requests.exceptions.RequestException("Connection error")

    response = scraper._fetch_with_retry("https://example.com")

    assert response is None
    assert mock_get.call_count == scraper.max_retries


@patch("puffin.data.scraper.requests.get")
def test_scrape_earnings_transcripts_no_response(mock_get, scraper):
    mock_get.return_value = None

    result = scraper.scrape_earnings_transcripts("AAPL")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
    assert list(result.columns) == ["date", "quarter", "transcript_text", "url"]


@patch("puffin.data.scraper.requests.get")
def test_scrape_financial_tables(mock_get, scraper, mock_html_table):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = mock_html_table.encode()
    mock_get.return_value = mock_response

    result = scraper.scrape_financial_tables("https://example.com/financials")

    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    assert "Date" in result.columns
    assert "Revenue" in result.columns
    assert "EPS" in result.columns


@patch("puffin.data.scraper.requests.get")
def test_scrape_financial_tables_no_tables(mock_get, scraper):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = b"<html><body><p>No tables here</p></body></html>"
    mock_get.return_value = mock_response

    result = scraper.scrape_financial_tables("https://example.com/page")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


@patch("puffin.data.scraper.requests.get")
def test_scrape_sec_filings(mock_get, scraper, mock_sec_html):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = mock_sec_html.encode()
    mock_get.return_value = mock_response

    result = scraper.scrape_sec_filings("0000320193", form_type="10-K")

    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    assert "filing_date" in result.columns
    assert "form_type" in result.columns
    assert "url" in result.columns


@patch("puffin.data.scraper.requests.get")
def test_scrape_sec_filings_cik_padding(mock_get, scraper, mock_sec_html):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = mock_sec_html.encode()
    mock_get.return_value = mock_response

    # Test with unpadded CIK
    result = scraper.scrape_sec_filings("320193")

    # Verify the URL contains padded CIK
    called_url = mock_get.call_args[0][0]
    assert "CIK=0000320193" in called_url


@patch("puffin.data.scraper.requests.get")
def test_rate_limiting(mock_get, scraper):
    import time

    mock_response = Mock()
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    start = time.time()
    scraper._fetch_with_retry("https://example.com/1")
    scraper._fetch_with_retry("https://example.com/2")
    elapsed = time.time() - start

    # Should take at least rate_limit seconds
    assert elapsed >= scraper.rate_limit
