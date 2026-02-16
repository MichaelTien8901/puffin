"""Tests for fundamental data provider."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from puffin.data.fundamental import FundamentalDataProvider


@pytest.fixture
def provider():
    return FundamentalDataProvider(user_agent="TestAgent/1.0")


@pytest.fixture
def mock_cik_response():
    return {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
    }


@pytest.fixture
def mock_submissions_response():
    return {
        "filings": {
            "recent": {
                "form": ["10-K", "10-Q", "10-K", "8-K"],
                "filingDate": ["2025-10-30", "2025-07-30", "2024-10-30", "2025-01-15"],
                "accessionNumber": ["0000320193-25-000001", "0000320193-25-000002", "0000320193-24-000001", "0000320193-25-000003"],
            }
        }
    }


@pytest.fixture
def mock_company_facts():
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {"end": "2024-09-30", "val": 391000000000, "form": "10-K"},
                            {"end": "2023-09-30", "val": 383000000000, "form": "10-K"},
                        ]
                    }
                },
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {"end": "2024-09-30", "val": 97000000000, "form": "10-K"},
                            {"end": "2023-09-30", "val": 95000000000, "form": "10-K"},
                        ]
                    }
                },
                "Assets": {
                    "units": {
                        "USD": [
                            {"end": "2024-09-30", "val": 365000000000, "form": "10-K"},
                            {"end": "2023-09-30", "val": 352000000000, "form": "10-K"},
                        ]
                    }
                },
                "StockholdersEquity": {
                    "units": {
                        "USD": [
                            {"end": "2024-09-30", "val": 74000000000, "form": "10-K"},
                            {"end": "2023-09-30", "val": 62000000000, "form": "10-K"},
                        ]
                    }
                },
                "Liabilities": {
                    "units": {
                        "USD": [
                            {"end": "2024-09-30", "val": 291000000000, "form": "10-K"},
                            {"end": "2023-09-30", "val": 290000000000, "form": "10-K"},
                        ]
                    }
                },
            }
        }
    }


def test_get_cik(provider, mock_cik_response):
    with patch.object(provider.session, "get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = mock_cik_response
        mock_get.return_value = mock_response

        cik = provider._get_cik("AAPL")
        assert cik == 320193

        cik = provider._get_cik("MSFT")
        assert cik == 789019

        cik = provider._get_cik("INVALID")
        assert cik is None


def test_fetch_filings(provider, mock_cik_response, mock_submissions_response):
    with patch.object(provider.session, "get") as mock_get:
        # Mock CIK lookup
        cik_response = MagicMock()
        cik_response.json.return_value = mock_cik_response

        # Mock submissions
        submissions_response = MagicMock()
        submissions_response.json.return_value = mock_submissions_response

        mock_get.side_effect = [cik_response, submissions_response]

        filings = provider.fetch_filings("AAPL", filing_type="10-K", limit=10)

        assert len(filings) == 2
        assert filings[0]["filing_date"] == "2025-10-30"
        assert filings[0]["accession_number"] == "0000320193-25-000001"
        assert filings[1]["filing_date"] == "2024-10-30"


def test_fetch_filings_limit(provider, mock_cik_response, mock_submissions_response):
    with patch.object(provider.session, "get") as mock_get:
        cik_response = MagicMock()
        cik_response.json.return_value = mock_cik_response

        submissions_response = MagicMock()
        submissions_response.json.return_value = mock_submissions_response

        mock_get.side_effect = [cik_response, submissions_response]

        filings = provider.fetch_filings("AAPL", filing_type="10-K", limit=1)

        assert len(filings) == 1
        assert filings[0]["filing_date"] == "2025-10-30"


def test_get_financials_income(provider, mock_cik_response, mock_company_facts):
    with patch.object(provider.session, "get") as mock_get:
        cik_response = MagicMock()
        cik_response.json.return_value = mock_cik_response

        facts_response = MagicMock()
        facts_response.json.return_value = mock_company_facts

        mock_get.side_effect = [cik_response, facts_response]

        df = provider.get_financials("AAPL", statement="income", period="annual")

        assert not df.empty
        assert "Revenues" in df.columns
        assert "NetIncomeLoss" in df.columns
        assert len(df) == 2


def test_get_financials_balance(provider, mock_cik_response, mock_company_facts):
    with patch.object(provider.session, "get") as mock_get:
        cik_response = MagicMock()
        cik_response.json.return_value = mock_cik_response

        facts_response = MagicMock()
        facts_response.json.return_value = mock_company_facts

        mock_get.side_effect = [cik_response, facts_response]

        df = provider.get_financials("AAPL", statement="balance", period="annual")

        assert not df.empty
        assert "Assets" in df.columns
        assert "StockholdersEquity" in df.columns
        assert "Liabilities" in df.columns


def test_compute_ratios(provider, mock_cik_response, mock_company_facts):
    with patch.object(provider.session, "get") as mock_get:
        cik_response = MagicMock()
        cik_response.json.return_value = mock_cik_response

        facts_response = MagicMock()
        facts_response.json.return_value = mock_company_facts

        # Called twice - once for income, once for balance
        mock_get.side_effect = [cik_response, facts_response, cik_response, facts_response]

        ratios = provider.compute_ratios("AAPL", price=150.0)

        assert "net_margin" in ratios
        assert "roe" in ratios
        assert "roa" in ratios
        assert "debt_to_equity" in ratios

        # Check calculated values
        assert ratios["net_margin"] == pytest.approx(97000000000 / 391000000000, rel=1e-6)
        assert ratios["roe"] == pytest.approx(97000000000 / 74000000000, rel=1e-6)
        assert ratios["roa"] == pytest.approx(97000000000 / 365000000000, rel=1e-6)
        assert ratios["debt_to_equity"] == pytest.approx(291000000000 / 74000000000, rel=1e-6)


def test_compute_ratios_invalid_ticker(provider, mock_cik_response):
    with patch.object(provider.session, "get") as mock_get:
        cik_response = MagicMock()
        cik_response.json.return_value = mock_cik_response

        mock_get.return_value = cik_response

        ratios = provider.compute_ratios("INVALID")
        assert ratios == {}


def test_fetch_filings_no_cik(provider, mock_cik_response):
    with patch.object(provider.session, "get") as mock_get:
        cik_response = MagicMock()
        cik_response.json.return_value = mock_cik_response

        mock_get.return_value = cik_response

        filings = provider.fetch_filings("INVALID")
        assert filings == []
