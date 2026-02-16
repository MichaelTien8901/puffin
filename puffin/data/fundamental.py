"""Fundamental data provider using SEC EDGAR API."""

import re
from datetime import datetime
from typing import Any

import pandas as pd
import requests


class FundamentalDataProvider:
    """Provider for fundamental data from SEC EDGAR.

    Fetches company filings, parses financial statements, and computes ratios.
    """

    BASE_URL = "https://data.sec.gov"

    def __init__(self, user_agent: str = "PuffinTradingSystem/1.0"):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def fetch_filings(
        self,
        ticker: str,
        filing_type: str = "10-K",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Fetch company filings from SEC EDGAR.

        Args:
            ticker: Stock ticker symbol
            filing_type: Filing type (10-K, 10-Q, 8-K, etc.)
            limit: Maximum number of filings to return

        Returns:
            List of filing dictionaries with keys: filing_date, accession_number, url
        """
        # Get CIK (Central Index Key) for ticker
        cik = self._get_cik(ticker)
        if not cik:
            return []

        # Fetch submissions data
        url = f"{self.BASE_URL}/submissions/CIK{cik:010d}.json"
        response = self.session.get(url)
        response.raise_for_status()
        data = response.json()

        # Extract filings of requested type
        filings = []
        recent = data.get("filings", {}).get("recent", {})

        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        accession_numbers = recent.get("accessionNumber", [])

        for i, form in enumerate(forms):
            if form == filing_type and len(filings) < limit:
                accession = accession_numbers[i].replace("-", "")
                filings.append({
                    "filing_date": filing_dates[i],
                    "accession_number": accession_numbers[i],
                    "url": f"{self.BASE_URL}/Archives/edgar/data/{cik}/{accession}/{accession_numbers[i]}-index.htm",
                })

        return filings

    def get_financials(
        self,
        ticker: str,
        statement: str = "income",
        period: str = "annual",
    ) -> pd.DataFrame:
        """Get financial statement data.

        Args:
            ticker: Stock ticker symbol
            statement: Statement type ('income', 'balance', 'cashflow')
            period: 'annual' or 'quarterly'

        Returns:
            DataFrame with financial data indexed by date
        """
        cik = self._get_cik(ticker)
        if not cik:
            return pd.DataFrame()

        # Fetch company facts (XBRL data)
        url = f"{self.BASE_URL}/api/xbrl/companyfacts/CIK{cik:010d}.json"
        response = self.session.get(url)
        response.raise_for_status()
        data = response.json()

        # Map statement type to XBRL concepts
        concept_map = {
            "income": [
                "Revenues",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "CostOfRevenue",
                "GrossProfit",
                "OperatingIncomeLoss",
                "NetIncomeLoss",
            ],
            "balance": [
                "Assets",
                "AssetsCurrent",
                "Liabilities",
                "LiabilitiesCurrent",
                "StockholdersEquity",
            ],
            "cashflow": [
                "NetCashProvidedByUsedInOperatingActivities",
                "NetCashProvidedByUsedInInvestingActivities",
                "NetCashProvidedByUsedInFinancingActivities",
            ],
        }

        concepts = concept_map.get(statement, [])
        facts = data.get("facts", {}).get("us-gaap", {})

        # Extract data for each concept
        records = []
        for concept in concepts:
            if concept in facts:
                units = facts[concept].get("units", {})
                # Try USD first, then shares
                unit_data = units.get("USD", units.get("shares", []))

                for item in unit_data:
                    if item.get("form") in ("10-K", "10-Q"):
                        if period == "annual" and item.get("form") == "10-K":
                            records.append({
                                "date": item.get("end"),
                                "concept": concept,
                                "value": item.get("val"),
                            })
                        elif period == "quarterly" and item.get("form") == "10-Q":
                            records.append({
                                "date": item.get("end"),
                                "concept": concept,
                                "value": item.get("val"),
                            })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])

        # Pivot to wide format
        result = df.pivot_table(
            index="date",
            columns="concept",
            values="value",
            aggfunc="first",
        )

        return result.sort_index(ascending=False)

    def compute_ratios(
        self,
        ticker: str,
        price: float | None = None,
    ) -> dict[str, float]:
        """Compute fundamental ratios.

        Args:
            ticker: Stock ticker symbol
            price: Current stock price (for price-based ratios)

        Returns:
            Dictionary of ratio names to values
        """
        # Get latest annual financials
        income = self.get_financials(ticker, "income", "annual")
        balance = self.get_financials(ticker, "balance", "annual")

        if income.empty or balance.empty:
            return {}

        ratios: dict[str, float] = {}

        # Get latest values
        latest_income = income.iloc[0]
        latest_balance = balance.iloc[0]

        # Revenue and profit metrics
        revenue = latest_income.get("Revenues") or latest_income.get(
            "RevenueFromContractWithCustomerExcludingAssessedTax"
        )
        net_income = latest_income.get("NetIncomeLoss")
        gross_profit = latest_income.get("GrossProfit")
        cost_of_revenue = latest_income.get("CostOfRevenue")

        # Balance sheet metrics
        assets = latest_balance.get("Assets")
        equity = latest_balance.get("StockholdersEquity")
        liabilities = latest_balance.get("Liabilities")
        current_assets = latest_balance.get("AssetsCurrent")
        current_liabilities = latest_balance.get("LiabilitiesCurrent")

        # Profitability ratios
        if revenue and net_income:
            ratios["net_margin"] = net_income / revenue

        if revenue and gross_profit:
            ratios["gross_margin"] = gross_profit / revenue

        if equity and net_income:
            ratios["roe"] = net_income / equity  # Return on Equity

        if assets and net_income:
            ratios["roa"] = net_income / assets  # Return on Assets

        # Leverage ratios
        if equity and liabilities:
            ratios["debt_to_equity"] = liabilities / equity

        if assets and liabilities:
            ratios["debt_to_assets"] = liabilities / assets

        # Liquidity ratios
        if current_assets and current_liabilities:
            ratios["current_ratio"] = current_assets / current_liabilities

        # Price ratios (if price provided)
        if price and net_income and equity:
            # Estimate shares outstanding from book value per share assumption
            # This is simplified - real implementation would fetch shares outstanding
            book_value_per_share = equity / 1_000_000  # Rough approximation
            if book_value_per_share > 0:
                ratios["price_to_book"] = price / book_value_per_share

        return ratios

    def _get_cik(self, ticker: str) -> int | None:
        """Get CIK (Central Index Key) for a ticker symbol.

        Args:
            ticker: Stock ticker symbol

        Returns:
            CIK number or None if not found
        """
        url = f"{self.BASE_URL}/files/company_tickers.json"
        response = self.session.get(url)
        response.raise_for_status()
        data = response.json()

        # Search for ticker
        ticker = ticker.upper()
        for item in data.values():
            if item.get("ticker") == ticker:
                return item.get("cik_str")

        return None
